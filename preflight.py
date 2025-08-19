# mcp_ai/preflight.py
from __future__ import annotations
import os, json, sqlite3, logging
from typing import Dict, Any, List, Tuple

log = logging.getLogger("preflight")

def _writable_dir(path: str) -> Tuple[bool, str]:
    try:
        os.makedirs(path, exist_ok=True)
        test = os.path.join(path, ".mcp_ai_write_test")
        with open(test, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test)
        return True, ""
    except Exception as e:
        return False, str(e)

def _check_roots(cfg: Dict[str, Any]) -> List[str]:
    issues = []
    roots = ((cfg.get("paths") or {}).get("roots") or [])
    if not roots:
        return ["paths.roots is empty."]
    for r in roots:
        if not os.path.exists(r):
            issues.append(f"root does not exist: {r}")
    return issues

def _check_storage(cfg: Dict[str, Any]) -> List[str]:
    issues = []
    st = cfg.get("storage") or {}
    targets = [
        st.get("out_root"),
        st.get("catalog_jsonl", "./out/metadata_catalog.jsonl"),
        st.get("catalog_csv", "./out/metadata_catalog.csv"),
        st.get("state_db", "./out/state.sqlite"),
    ]
    dirs = set()
    for p in targets:
        if not p: continue
        d = p if p.endswith(os.sep) or os.path.splitext(p)[1]=="" else os.path.dirname(p)
        if d: dirs.add(os.path.abspath(d))
    for d in dirs:
        ok, err = _writable_dir(d)
        if not ok: issues.append(f"output not writable: {d} ({err})")
    # sqlite open
    db = st.get("state_db", "./out/state.sqlite")
    try:
        parent = os.path.dirname(os.path.abspath(db))
        if parent and os.path.isdir(parent):
            con = sqlite3.connect(os.path.abspath(db))
            con.execute("PRAGMA journal_mode=WAL")
            con.close()
    except Exception as e:
        issues.append(f"sqlite open failed: {e}")
    return issues

def _check_ollama(cfg: Dict[str, Any]) -> List[str]:
    ai = cfg.get("ai") or {}
    if ai.get("mode") != "local":
        return []
    local = ai.get("local") or {}
    endpoint = local.get("endpoint") or "http://localhost:11434"
    model = local.get("fast_model") or local.get("model") or "mistral:7b-instruct"
    try:
        import requests  # type: ignore
        r = requests.get(endpoint.rstrip("/") + "/api/tags", timeout=2)
        r.raise_for_status()
        names = [m.get("name") for m in (r.json().get("models") or [])]
        if model not in names:
            return [f"Ollama model not present: {model} (available: {', '.join(names[:6])}...)"]
    except Exception as e:
        return [f"Ollama unreachable at {endpoint}: {e}"]
    return []

def validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    issues += _check_roots(cfg)
    issues += _check_storage(cfg)
    issues += _check_ollama(cfg)
    return {"ok": not issues, "issues": issues}

if __name__ == "__main__":
    import argparse
    from .utils import load_yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./config.yaml")
    a = p.parse_args()
    cfg = load_yaml(a.config)
    # resolve relative roots based on config file location
    config_dir = os.path.dirname(os.path.abspath(a.config))
    if "paths" in cfg and "roots" in cfg["paths"]:
        cfg["paths"]["roots"] = [os.path.abspath(os.path.join(config_dir, p)) for p in cfg["paths"]["roots"]]
    out = validate(cfg)
    print(json.dumps(out, indent=2))