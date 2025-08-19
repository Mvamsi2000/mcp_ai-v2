# mcp_ai/selfcheck.py
from __future__ import annotations
import argparse, json, logging, os, platform, shutil, sys
from typing import Any, Dict, List, Tuple

from .utils import load_yaml, ensure_dir

log = logging.getLogger("selfcheck")

def _ok(name: str, detail: str = "") -> Dict[str, Any]:
    return {"name": name, "status": "ok", "detail": detail}

def _warn(name: str, detail: str = "") -> Dict[str, Any]:
    return {"name": name, "status": "warn", "detail": detail}

def _fail(name: str, detail: str = "") -> Dict[str, Any]:
    return {"name": name, "status": "fail", "detail": detail}

def _try_import(mod: str) -> Tuple[bool, str]:
    try:
        __import__(mod)
        return True, ""
    except Exception as e:
        return False, repr(e)

def _http_json(method: str, url: str, payload: Dict[str, Any] | None, timeout: int = 5) -> Tuple[int, Dict[str, Any] | str]:
    try:
        import requests  # type: ignore
        if method == "GET":
            r = requests.get(url, timeout=timeout)
        else:
            r = requests.post(url, json=payload or {}, timeout=timeout)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text
    except Exception:
        # urllib fallback
        try:
            import urllib.request, json as _json  # type: ignore
            if method == "GET":
                req = urllib.request.Request(url, method="GET")
            else:
                data = _json.dumps(payload or {}).encode("utf-8")
                req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read().decode("utf-8", "ignore")
                try:
                    return resp.status, _json.loads(data)
                except Exception:
                    return resp.status, data
        except Exception as e:
            return 0, f"http error: {e!r}"

def check_python() -> Dict[str, Any]:
    ver = sys.version_info
    if ver.major >= 3 and ver.minor >= 9:
        return _ok("python", f"{platform.python_version()}")
    return _warn("python", f"Python {platform.python_version()} — recommended >= 3.9")

def check_libs() -> List[Dict[str, Any]]:
    req = ["yaml", "fitz", "pdfminer.high_level", "pdfplumber", "PIL", "pytesseract", "pdf2image", "openpyxl", "docx2txt"]
    out: List[Dict[str, Any]] = []
    for m in req:
        ok, err = _try_import(m)
        if ok:
            out.append(_ok(f"import:{m}"))
        else:
            out.append(_warn(f"import:{m}", err))
    # tesseract binary (optional but useful)
    if shutil.which("tesseract"):
        out.append(_ok("bin:tesseract"))
    else:
        out.append(_warn("bin:tesseract", "Not found in PATH (OCR still works if pytesseract can locate it)"))
    # poppler (for pdf2image)
    if shutil.which("pdftoppm"):
        out.append(_ok("bin:pdftoppm(poppler)"))
    else:
        out.append(_warn("bin:pdftoppm(poppler)", "Not found. Install poppler for faster PDF OCR."))
    return out

def check_storage(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    s = (cfg.get("storage") or {})
    out_root = s.get("out_root", "./mcp_ai/output_files")
    try:
        ensure_dir(out_root)
        test_path = os.path.join(out_root, ".writetest")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return [_ok("storage:writable", out_root)]
    except Exception as e:
        return [_fail("storage:writable", f"{out_root}: {e!r}")]

def check_ollama(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    ai = (cfg.get("ai") or {})
    local = (ai.get("local") or {})
    endpoint = (local.get("endpoint") or "http://127.0.0.1:11434").rstrip("/")
    fast_model = local.get("fast_model") or "mistral:7b-instruct"
    deep_model = local.get("deep_model") or "qwen2.5:7b-instruct"
    out: List[Dict[str, Any]] = []

    code, data = _http_json("GET", endpoint + "/api/tags", None, timeout=3)
    if code == 200:
        out.append(_ok("ollama:endpoint", endpoint))
        # see if models exist locally
        names = []
        try:
            items = (data or {}).get("models", [])  # type: ignore
            names = [x.get("name") for x in items if isinstance(x, dict)]
        except Exception:
            pass
        missing = [m for m in (fast_model, deep_model) if m not in names]
        if missing:
            out.append(_warn("ollama:models", f"Missing local models: {missing}. Tip: ollama pull <model>"))
        else:
            out.append(_ok("ollama:models", "fast/deep present"))
    else:
        out.append(_warn("ollama:endpoint", f"Unreachable at {endpoint} (status={code}, data={str(data)[:120]})"))
    return out

def check_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    profile = cfg.get("scan_profile", "deep")
    if profile not in (cfg.get("profiles") or {}):
        issues.append(_warn("config:scan_profile", f"Profile '{profile}' not found, will fallback to 'deep' behavior."))
    deep = (cfg.get("ai", {}) or {}).get("deep", {}) or {}
    if (deep.get("policy") or "auto") not in ("auto", "always", "never"):
        issues.append(_warn("config:ai.deep.policy", "Invalid policy; use auto|always|never"))
    if (deep.get("sampling") or "smart") not in ("smart", "full"):
        issues.append(_warn("config:ai.deep.sampling", "Invalid sampling; use smart|full"))
    if not issues:
        return [_ok("config")]
    return issues

def main() -> None:
    ap = argparse.ArgumentParser("mcp_ai.selfcheck")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    report = {
        "python": check_python(),
        "libs": check_libs(),
        "storage": check_storage(cfg),
        "ollama": check_ollama(cfg),
        "config": check_config(cfg),
    }
    status_counts = {"ok": 0, "warn": 0, "fail": 0}
    for section in report.values():
        if isinstance(section, dict):
            status_counts[section.get("status","ok")] = status_counts.get(section.get("status","ok"),0)+1
        else:
            for item in section:
                status_counts[item.get("status","ok")] = status_counts.get(item.get("status","ok"),0)+1

    report["summary"] = status_counts
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()