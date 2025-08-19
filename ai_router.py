# ---- in mcp_ai/ai_router.py ----
from __future__ import annotations
from typing import Any, Dict, Optional
import logging

from .utils import is_network_allowed, BudgetLedger
from .llm_agent import (
    fast_pass,
    deep_pass_local,
    deep_pass_cloud_demo,
    smart_sample,
    infer_business_domain,
)

log = logging.getLogger("ai_router")

# If you don't already have these, keep your existing implementations.
# They fill missing fields & keep the schema stable.
ALLOWED_DOMAINS = {"finance","legal","hr","it","marketing","engineering","operations","healthcare","education","general"}

def sanitize_fast(obj: Dict[str, Any], *, text: str) -> Dict[str, Any]:
    # ensure keys exist with types
    cat = obj.get("category") or "unknown"
    dom = obj.get("domain") or infer_business_domain(cat, text)
    if dom not in ALLOWED_DOMAINS:
        dom = infer_business_domain(cat, text)
    tags = obj.get("tags") or []
    if not isinstance(tags, list): tags = []
    pii = bool(obj.get("contains_pii") or False)
    conf = float(obj.get("confidence") or 0.0)
    summ = obj.get("summary") or ""
    pv = obj.get("prompt_version") or "fast-v2-rich"
    return {
        "category": str(cat),
        "domain": str(dom),
        "tags": [str(t) for t in tags[:8]],
        "contains_pii": bool(pii),
        "confidence": float(conf),
        "summary": str(summ)[:600],
        "prompt_version": pv,
    }

def sanitize_deep(obj: Dict[str, Any], *, carry_category: Optional[str]) -> Dict[str, Any]:
    cat = obj.get("category") or carry_category or "unknown"
    conf = float(obj.get("confidence") or 0.0)
    pii = obj.get("pii") or []
    if not isinstance(pii, list): pii = []
    gloss = obj.get("glossary_terms") or []
    if not isinstance(gloss, list): gloss = []
    summ = obj.get("summary") or ""
    pv = obj.get("prompt_version") or "deep-v2-rich"
    return {
        "category": str(cat),
        "confidence": float(conf),
        "pii": [str(x) for x in pii],
        "glossary_terms": [str(g) for g in gloss[:12]],
        "summary": str(summ)[:1200],
        "prompt_version": pv,
    }

_WARNED_FILE_ID_ONCE = False

def ai_enrich(
    text: str,
    cfg: Dict[str, Any],
    doc_id: Optional[str] = None,
    budget: Optional[BudgetLedger] = None,
    *,
    file_id: Optional[str] = None,   # ← back-compat
    high_value: bool = False,
    decision_trace: bool = False,
    **_future,                       # ← ignore unknown future kwargs safely
) -> Dict[str, Any]:
    """
    Orchestrates FAST + optional DEEP passes with robust, compatible signature.

    Decides 'deep' via ai.deep.policy:
      - "always": deep on every file
      - "never":  never run deep
      - "auto":   triggers based on config (confidence, PII, high_value)
    """
    global _WARNED_FILE_ID_ONCE

    # Prefer doc_id; fall back to file_id for old callers
    effective_id = doc_id or file_id or "unknown"
    if file_id and not doc_id and not _WARNED_FILE_ID_ONCE:
        log.info("ai_enrich: 'file_id' is deprecated; use 'doc_id' instead.")
        _WARNED_FILE_ID_ONCE = True

    ai_cfg = (cfg.get("ai", {}) or {})
    mode = (ai_cfg.get("mode") or "none").lower()

    deep_cfg = (ai_cfg.get("deep", {}) or {})
    policy = (deep_cfg.get("policy") or "auto").lower()
    triggers = (deep_cfg.get("auto_triggers", {}) or {})
    # Fallback to legacy key if new triggers aren’t present
    low_conf_below = float(triggers.get("low_confidence_below", deep_cfg.get("confidence_threshold", 0.75)))
    trig_contains_pii = bool(triggers.get("contains_pii", True))
    trig_high_value   = bool(triggers.get("high_value_flag", True))

    sampling = (deep_cfg.get("sampling") or "smart").lower()
    max_chars_fast  = int(deep_cfg.get("max_chars_fast", 6000))
    max_chars_local = int(deep_cfg.get("max_chars_local", 60000))

    out_fast: Dict[str, Any] = {}
    out_deep: Dict[str, Any] = {}
    ai_cost = 0.0
    trace: Dict[str, Any] = {}

    # ---------- FAST ----------
    if mode == "none" or not text:
        out_fast = sanitize_fast({}, text=text or "")
    else:
        try:
            raw_fast = fast_pass((text or "")[:max_chars_fast], cfg)
            out_fast = sanitize_fast(raw_fast or {}, text=text or "")
        except Exception as e:
            log.warning("FAST pass failed: %r", e)
            out_fast = sanitize_fast({}, text=text or "")

    # ---------- Decide DEEP ----------
    conf = float(out_fast.get("confidence") or 0.0)
    contains_pii = bool(out_fast.get("contains_pii") or False)

    run_deep = False
    reason = "n/a"
    if policy == "always":
        run_deep, reason = True, "policy=always"
    elif policy == "never":
        run_deep, reason = False, "policy=never"
    else:  # auto
        if conf < low_conf_below:
            run_deep, reason = True, f"auto: low_confidence {conf:.2f}<{low_conf_below:.2f}"
        elif trig_contains_pii and contains_pii:
            run_deep, reason = True, "auto: contains_pii"
        elif trig_high_value and high_value:
            run_deep, reason = True, "auto: high_value flag"
        else:
            run_deep, reason = False, "auto: no triggers"

    # ---------- DEEP (optional) ----------
    if run_deep and text:
        try:
            deep_text = (
                text[:max_chars_local] if sampling == "full"
                else smart_sample(text, max_chars=max_chars_local, entity_chunks=3, chunk_size=1200)
            )
            if (ai_cfg.get("mode") or "local").lower() == "local":
                raw_deep = deep_pass_local(deep_text, cfg, max_chars=max_chars_local)
            else:
                # Cloud: still safe if networking is disabled
                raw_deep = deep_pass_cloud_demo(deep_text, max_chars=max_chars_local) \
                           if not is_network_allowed(cfg) else deep_pass_cloud_demo(deep_text, max_chars=max_chars_local)  # TODO: real cloud
            out_deep = sanitize_deep(raw_deep or {}, carry_category=out_fast.get("category"))
        except Exception as e:
            log.warning("DEEP pass failed: %r", e)
            out_deep = sanitize_deep({}, carry_category=out_fast.get("category"))
    else:
        out_deep = sanitize_deep({}, carry_category=out_fast.get("category"))

    # Budget accounting hook (noop for local)
    try:
        if budget and hasattr(budget, "charge"):
            budget.charge(0.0)
    except Exception:
        pass

    if decision_trace:
        trace = {
            "doc_id": effective_id,
            "policy": policy,
            "triggers": {
                "low_confidence_below": low_conf_below,
                "contains_pii": trig_contains_pii,
                "high_value_flag": trig_high_value,
            },
            "fast_confidence": conf,
            "fast_contains_pii": contains_pii,
            "run_deep": run_deep,
            "reason": reason,
            "sampling": sampling,
        }

    return {
        "ai_fast": out_fast,
        "ai_deep_local": out_deep,
        "ai_cost_usd": float(ai_cost or 0.0),
        **({"_decision": trace} if decision_trace else {}),
    }