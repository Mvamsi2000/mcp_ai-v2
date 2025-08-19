# mcp_ai/llm_agent.py
from __future__ import annotations
import os, json, time, logging
from typing import Any, Dict, Optional

log = logging.getLogger("llm_agent")

# ───────────────────────────── Template loading ─────────────────────────────

def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

def _load_template(name: str, fallback: str) -> str:
    """
    Search order:
      1) Env var file override: MCPA_<FILENAME_UPPER> (abs path)
      2) Env var dir override:  MCPA_PROMPTS_DIR/<name>
      3) mcp_ai/prompts/<name>
      4) Same directory as this file
      5) Provided fallback
    """
    env_file = os.getenv(f"MCPA_{name.upper().replace('.', '_')}")
    if env_file:
        txt = _read_text(env_file)
        if txt: return txt
    env_dir = os.getenv("MCPA_PROMPTS_DIR")
    if env_dir:
        cand = os.path.join(env_dir, name)
        txt = _read_text(cand)
        if txt: return txt
    here = os.path.dirname(__file__)
    cand = os.path.join(here, "prompts", name)
    txt = _read_text(cand)
    if txt: return txt
    cand2 = os.path.join(here, name)
    txt2 = _read_text(cand2)
    if txt2: return txt2
    return fallback

DEFAULT_FAST_TMPL = """You are a data catalog analyst. Read the provided content and return a single JSON object.

HARD CONSTRAINTS
- Output ONLY a JSON object. No explanations, no markdown, no code fences.
- Use STRICT JSON (double quotes for keys/strings, no trailing commas).
- If unsure, be conservative and lower the confidence (0.0–1.0).

ALLOWED DOMAINS
["finance","legal","hr","it","marketing","engineering","operations","healthcare","education","general"]

TARGET JSON SCHEMA
{
  "category": "<short label>",
  "domain": "<one of the ALLOWED DOMAINS>",
  "tags": ["<up to 8 short tags>"],
  "contains_pii": <true|false>,
  "confidence": <float 0.0-1.0>,
  "summary": "<1–2 sentence executive summary>",
  "prompt_version": "fast-v2-rich"
}

CONTENT
<<<
{CONTENT}
>>>
"""

DEFAULT_DEEP_TMPL = """You are a senior data catalog analyst. Read the provided content and return a single JSON object.

HARD CONSTRAINTS
- Output ONLY a JSON object. No prose/markdown/code fences.
- STRICT JSON (double quotes everywhere, no comments, no trailing commas).
- If an item doesn’t apply, use [] or "".

PII TYPES
["email","phone","address","name","dob","ssn","credit_card","bank_acct","id_number","ip_address"]

TARGET JSON SCHEMA
{
  "summary": "<2–5 sentences>",
  "pii": ["<zero or more of the PII TYPES>"],
  "glossary_terms": ["<up to 12 concise terms>"],
  "prompt_version": "deep-v2-rich"
}

CONTENT
<<<
{CONTENT}
>>>
"""

_FAST_TMPL = _load_template("fast.json.tmpl", DEFAULT_FAST_TMPL)
_DEEP_TMPL = _load_template("deep.json.tmpl", DEFAULT_DEEP_TMPL)

def _render(tmpl: str, *, text: str) -> str:
    try:
        return tmpl.format(CONTENT=text, body=text)
    except Exception:
        return (tmpl or "") + "\n" + text

# ───────────────────────────── JSON extraction ─────────────────────────────

def _between_code_fences(s: str) -> Optional[str]:
    start = s.find("```")
    if start == -1: return None
    rest = s[start+3:]
    end = rest.find("```")
    if end == -1: return None
    return rest[:end]

def _stack_find_json(s: str) -> Optional[str]:
    start = None
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return s[start:i+1]
    return None

def _loose_slice(s: str) -> Optional[str]:
    a = s.find("{")
    b = s.rfind("}")
    if a != -1 and b != -1 and b > a:
        return s[a:b+1]
    return None

def _parse_json_loose(s: str) -> Optional[Dict[str, Any]]:
    if not s: return None
    cand = _between_code_fences(s) or _stack_find_json(s) or _loose_slice(s) or s.strip()
    if not cand: return None
    try:
        return json.loads(cand)
    except Exception:
        pass
    cleaned = cand.replace("\r", "").replace(",\n}", "\n}").replace(",\n]", "\n]")
    try:
        return json.loads(cleaned)
    except Exception:
        return None

# ───────────────────────────── HTTP helpers ─────────────────────────────

def _http_post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    try:
        import requests  # type: ignore
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        import urllib.request, urllib.error  # type: ignore
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            try:
                return json.loads(data)
            except Exception:
                return {"response": data}

# ───────────────────────────── LLM calls (Ollama) ─────────────────────────────

def _ollama_generate(endpoint: str, model: str, prompt: str, timeout_s: int = 60,
                     retries: int = 0, backoff_s: float = 1.0) -> str:
    url = endpoint.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    attempt = 0
    last_err: Optional[Exception] = None
    while attempt <= max(0, retries):
        try:
            data = _http_post_json(url, payload, timeout_s)
            return str(data.get("response", ""))
        except Exception as e:
            last_err = e
            if attempt == retries:
                break
            sleep = backoff_s * (2 ** attempt)
            log.warning("Ollama call failed (attempt %d/%d): %r; backing off %.2fs",
                        attempt + 1, retries + 1, e, sleep)
            time.sleep(sleep)
            attempt += 1
    raise RuntimeError(f"Ollama call failed after {retries+1} attempts: {last_err!r}")

def _get_local_settings(cfg: Optional[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    endpoint = None
    model = None
    timeout_s = None
    retries = None
    backoff_s = None
    if cfg:
        local = (cfg.get("ai", {}) or {}).get("local", {}) or {}
        endpoint = local.get("endpoint")
        model = local.get("fast_model" if kind == "fast" else "deep_model")
        timeout_s = local.get("timeout_s")
        retries = local.get("retries")
        backoff_s = local.get("backoff_s")

    endpoint = endpoint or os.getenv("OLLAMA_ENDPOINT") or "http://127.0.0.1:11434"
    model = model or os.getenv("OLLAMA_FAST_MODEL" if kind == "fast" else "OLLAMA_DEEP_MODEL") or "qwen2.5:7b-instruct"

    try:
        timeout_s = int(timeout_s) if timeout_s is not None else None
    except Exception:
        timeout_s = None
    timeout_s = timeout_s or int(os.getenv("OLLAMA_TIMEOUT_S", "60"))

    try:
        retries = int(retries) if retries is not None else 0
    except Exception:
        retries = 0

    try:
        backoff_s = float(backoff_s) if backoff_s is not None else 1.0
    except Exception:
        backoff_s = 1.0

    return {"endpoint": endpoint, "model": model, "timeout_s": timeout_s, "retries": retries, "backoff_s": backoff_s}

def _call_local_ollama(prompt: str, cfg: Optional[Dict[str, Any]], kind: str) -> str:
    s = _get_local_settings(cfg, kind)
    return _ollama_generate(
        s["endpoint"], s["model"], prompt,
        timeout_s=s["timeout_s"], retries=s["retries"], backoff_s=s["backoff_s"]
    )

def _call_cloud_demo(_: str) -> str:
    # Deterministic stub for offline mode
    return (
        '{ "category": "unknown", "domain": "general", "tags": [], '
        '"contains_pii": false, "confidence": 0.4, "summary": "" }'
    )

def _resolve_mode_provider(cfg: Optional[Dict[str, Any]]) -> tuple[str, str]:
    ai = (cfg or {}).get("ai", {}) or {}
    mode = (ai.get("mode") or "local").lower()
    provider = (ai.get("provider") or "ollama").lower()
    return mode, provider

# ───────────────────────────── Public API ─────────────────────────────

def fast_pass(text: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prompt = _render(_FAST_TMPL, text=text)
    mode, provider = _resolve_mode_provider(cfg)
    try:
        if mode == "local" and provider == "ollama":
            raw = _call_local_ollama(prompt, cfg, "fast")
        else:
            raw = _call_cloud_demo(prompt)
    except Exception as e:
        log.warning("FAST pass failed: %r", e)
        raw = _call_cloud_demo(prompt)
    return _parse_json_loose(raw) or {}

def smart_sample(text: str, max_chars: int = 60000, entity_chunks: int = 3, chunk_size: int = 1200) -> str:
    if len(text) <= max_chars: return text
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]

def deep_pass_local(text: str, cfg: Optional[Dict[str, Any]] = None, *, max_chars: int = 60000) -> Dict[str, Any]:
    prompt = _render(_DEEP_TMPL, text=text[:max_chars])
    mode, provider = _resolve_mode_provider(cfg)
    try:
        if mode == "local" and provider == "ollama":
            raw = _call_local_ollama(prompt, cfg, "deep")
        else:
            raw = _call_cloud_demo(prompt)
    except Exception as e:
        log.warning("DEEP pass failed: %r", e)
        raw = _call_cloud_demo(prompt)
    obj = _parse_json_loose(raw)
    if not obj:
        log.warning("deep_pass_local: could not parse JSON; returning heuristic summary.")
        return {
            "category": "unknown",
            "confidence": 0.4,
            "glossary_terms": [],
            "pii": [],
            "summary": text[:500] + ("..." if len(text) > 500 else ""),
            "prompt_version": "deep-fallback",
        }
    return obj

def deep_pass_cloud_demo(text: str, *, max_chars: int = 120000) -> Dict[str, Any]:
    obj = _parse_json_loose(_call_cloud_demo(text[:max_chars]))
    return obj or {
        "category": "unknown",
        "confidence": 0.4,
        "glossary_terms": [],
        "pii": [],
        "summary": text[:500] + ("..." if len(text) > 500 else ""),
        "prompt_version": "deep-cloud-fallback",
    }

def infer_business_domain(category: str, sample_text: str) -> str:
    cat = (category or "").lower()
    if any(k in cat for k in ("invoice","po","budget","ledger","finance")): return "finance"
    if any(k in cat for k in ("contract","nda","legal","agreement","clause")): return "legal"
    s = (sample_text or "").lower()
    if any(w in s for w in ("kubernetes","terraform","server","endpoint","api","database","log")): return "it"
    if any(w in s for w in ("hire","payroll","benefits","onboarding","employee")): return "hr"
    if any(w in s for w in ("campaign","creative","brand","social","cta")): return "marketing"
    if any(w in s for w in ("spec","architecture","incident","runbook","deploy","build")): return "engineering"
    if any(w in s for w in ("inventory","fulfillment","logistics","sop","quality assurance","qa")): return "operations"
    if any(w in s for w in ("diagnosis","patient","treatment","clinical","hipaa")): return "healthcare"
    if any(w in s for w in ("curriculum","syllabus","student","exam","assignment")): return "education"
    return "general"