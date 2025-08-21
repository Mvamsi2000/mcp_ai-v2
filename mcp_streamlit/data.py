# mcp_ai/mcp_streamlit/data.py
from __future__ import annotations
import os, json, pathlib, tempfile
from typing import Any, Dict, List, Tuple

def list_runs(output_root: str) -> List[Tuple[str, str]]:
    """
    Return [(run_id, run_dir), ...] newest-first where items.jsonl exists.
    """
    runs_root = os.path.join(output_root, "runs")
    if not os.path.isdir(runs_root):
        return []
    entries: List[Tuple[str, str, float]] = []
    for name in os.listdir(runs_root):
        p = os.path.join(runs_root, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "items.jsonl")):
            entries.append((name, p, os.path.getmtime(p)))
    entries.sort(key=lambda t: t[2], reverse=True)
    return [(rid, p) for rid, p, _ in entries]

def load_items_jsonl(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "items.jsonl")
    if not os.path.isfile(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                # ignore malformed lines
                pass
    return out

def file_display_name(rec: Dict[str, Any]) -> str:
    fn = rec.get("filename")
    if fn: 
        return str(fn)
    p = rec.get("path") or ""
    base = pathlib.Path(p).name
    return base or p or "(unknown)"

def get_text_from_item(rec: Dict[str, Any]) -> str:
    """
    Tries common places your pipeline might stash text.
    """
    # direct
    if isinstance(rec.get("text"), str):
        return rec["text"]
    # in ai block
    ai = rec.get("ai") or {}
    if isinstance(ai.get("_raw_text"), str):
        return ai["_raw_text"]
    # sometimes in meta preview fields
    meta = rec.get("meta") or {}
    for k in ("preview", "text_preview", "ocr_text", "body"):
        if isinstance(meta.get(k), str):
            return meta[k]
    return ""

# ---------- client annotations (owners/tags/notes) ----------
def _ann_path(run_dir: str) -> str:
    return os.path.join(run_dir, "annotations.jsonl")

def load_annotations(run_dir: str) -> Dict[str, Dict[str, Any]]:
    path = _ann_path(run_dir)
    if not os.path.isfile(path):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            p = row.get("path")
            if not p:
                continue
            out[p] = {
                "owners": row.get("owners", []) or [],
                "tags": row.get("tags", []) or [],
                "notes": row.get("notes") or "",
            }
    return out

def upsert_annotation(run_dir: str, path: str, owners: List[str], tags: List[str], notes: str) -> None:
    cur = load_annotations(run_dir)
    cur[path] = {"owners": owners, "tags": tags, "notes": notes}
    out = _ann_path(run_dir)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for p, row in cur.items():
            f.write(json.dumps({"path": p, **row}, ensure_ascii=False) + "\n")
    os.replace(tmp, out)

# ---------- saved views (simple JSON file) ----------
def _views_path(run_dir: str) -> str:
    return os.path.join(run_dir, "views.json")

def load_saved_views(run_dir: str) -> Dict[str, Dict[str, Any]]:
    p = _views_path(run_dir)
    if not os.path.isfile(p):
        return {}
    try:
        return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_view(run_dir: str, name: str, config: Dict[str, Any]) -> None:
    allv = load_saved_views(run_dir)
    allv[name] = config
    pathlib.Path(_views_path(run_dir)).write_text(
        json.dumps(allv, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )