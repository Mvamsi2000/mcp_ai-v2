# mcp_ai/campaigns.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Mapping, List, Tuple

from .ai_router import ai_enrich, sanitize_fast, sanitize_deep
from .utils import BudgetLedger

log = logging.getLogger("campaigns")

def _as_dict(m: Mapping[str, Any] | Any) -> Dict[str, Any]:
    if isinstance(m, dict):
        return m
    try:
        return dict(m)  # sqlite3.Row and other mappings support this
    except Exception:
        # last resort: build from attributes
        d: Dict[str, Any] = {}
        for k in dir(m):
            if k.startswith("_"): 
                continue
            try:
                v = getattr(m, k)
            except Exception:
                continue
            d[k] = v
        return d

def _get(m: Mapping[str, Any], key: str, default: Any = "") -> Any:
    if hasattr(m, "get"):
        return m.get(key, default)  # type: ignore[no-any-return]
    try:
        return m[key]  # type: ignore[index]
    except Exception:
        return default

def _matches_condition(cond: str, row: Mapping[str, Any]) -> bool:
    """Very small filter mini-language: key=value comparisons, case-insensitive.
       Special case: contains_pii compares as int/bool."""
    r = _as_dict(row)
    c = (cond or "").strip()
    if not c:
        return True
    if "=" in c:
        left, right = c.split("=", 1)
        key = left.strip().lower()
        if key in ("contains_pii",):
            try:
                lhs = int(bool(_get(r, key, False)))
                rhs = int(right.strip())
                return lhs == rhs
            except Exception:
                return False
        targ = right.strip().strip("'").strip('"').lower()
        return str(_get(r, key, "")).lower() == targ
    return False

def filter_rows(rows: Iterable[Mapping[str, Any]], conditions: Iterable[str]) -> List[Dict[str, Any]]:
    conds = [c for c in (conditions or []) if c and str(c).strip()]
    out: List[Dict[str, Any]] = []
    for row in rows:
        rd = _as_dict(row)
        if all(_matches_condition(c, rd) for c in conds):
            out.append(rd)
    return out

def enrich_rows(rows: Iterable[Mapping[str, Any]], cfg: Dict[str, Any], budget: BudgetLedger) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """Run ai_enrich over each row that has 'text'."""
    results: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
    for row in rows:
        r = _as_dict(row)
        text = r.get("text") or ""
        sha1 = r.get("sha1") or r.get("doc_id") or ""
        if not text:
            continue
        try:
            out = ai_enrich(text, cfg, sha1, budget, high_value=False) or {}
            fast = out.get("ai_fast") or {}
            deep = out.get("ai_deep_local") or {}
            ai_cost = float(out.get("ai_cost_usd") or 0.0)
            results.append((sanitize_fast(fast, text=text), sanitize_deep(deep, carry_category=fast.get("category")), ai_cost))
        except Exception as e:
            log.warning("enrich failed for %s: %r", sha1, e)
            results.append((sanitize_fast({}, text=text), sanitize_deep({}, carry_category="unknown"), 0.0))
    return results