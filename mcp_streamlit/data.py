# mcp_ai/mcp_streamlit/data.py
from __future__ import annotations
import os, json, pathlib, functools
from typing import Dict, Any, List, Tuple

def list_runs(output_root: str) -> List[Tuple[str, str]]:
    """Return [(run_id, run_dir)] sorted by mtime desc."""
    runs_root = os.path.join(output_root, "runs")
    if not os.path.isdir(runs_root):
        return []
    out: List[Tuple[str, str]] = []
    for d in os.listdir(runs_root):
        p = os.path.join(runs_root, d)
        if os.path.isdir(p):
            out.append((d, p))
    out.sort(key=lambda t: os.path.getmtime(t[1]), reverse=True)
    return out

@functools.lru_cache(maxsize=32)
def load_items_jsonl(run_dir: str) -> List[Dict[str, Any]]:
    p = os.path.join(run_dir, "items.jsonl")
    items: List[Dict[str, Any]] = []
    if not os.path.isfile(p):
        return items
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

def get_text_from_item(rec: Dict[str, Any]) -> str:
    """Best-effort text: ai._raw_text, text, or fast summary."""
    ai = rec.get("ai") or {}
    txt = ai.get("_raw_text") or rec.get("text") or ""
    if not txt:
        txt = (ai.get("ai_fast") or {}).get("summary") or ""
    return txt or ""

def file_display_name(rec: Dict[str, Any]) -> str:
    return rec.get("filename") or pathlib.Path(rec.get("path", "")).name or "(unknown)"