from __future__ import annotations
import os, json, hashlib
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from config import settings

class DataStore:
    def __init__(self) -> None:
        self.output_root = settings.OUTPUT_ROOT
        self.run_id: Optional[str] = None
        self.items: List[Dict[str, Any]] = []
        self.df: pd.DataFrame | None = None
        self.chunks: List[Dict[str, Any]] = []
        self.graph_edges: List[Dict[str, Any]] = []

    def list_runs(self) -> List[str]:
        runs_root = os.path.join(self.output_root, "runs")
        if not os.path.isdir(runs_root):
            return []
        return sorted([d for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))])

    def _pick_run_dir(self) -> Tuple[Optional[str], Optional[str]]:
        target = settings.RUN_ID or self.run_id
        runs_root = os.path.join(self.output_root, "runs")
        if target:
            p = os.path.join(runs_root, target)
            if os.path.isdir(p):
                return target, p
        # latest by name (usually timestamp/uuid)
        runs = self.list_runs()
        if not runs:
            return None, None
        rid = runs[-1]
        return rid, os.path.join(runs_root, rid)

    def load(self, run_id: Optional[str]=None) -> None:
        if run_id:
            self.run_id = run_id
        rid, rdir = self._pick_run_dir()
        if not rdir:
            raise FileNotFoundError(f"No runs under {self.output_root}/runs. Set OUTPUT_ROOT or create a run.")
        self.run_id = rid

        items_path = os.path.join(rdir, "items.jsonl")
        if not os.path.isfile(items_path):
            raise FileNotFoundError(f"{items_path} not found")

        self.items.clear()
        with open(items_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                # normalize fields
                path = rec.get("path") or rec.get("file") or ""
                filename = rec.get("filename") or os.path.basename(path)
                ext = rec.get("ext") or os.path.splitext(filename)[1].lower()
                text = rec.get("ai", {}).get("_raw_text", rec.get("text", "")) or ""
                item = {
                    "run_id": rid,
                    "path": path,
                    "filename": filename,
                    "ext": ext,
                    "engine": rec.get("engine") or (rec.get("meta") or {}).get("engine"),
                    "language": rec.get("language") or (rec.get("meta") or {}).get("language"),
                    "duration": (rec.get("meta") or {}).get("duration"),
                    "elapsed_s": (rec.get("meta") or {}).get("elapsed_s"),
                    "extraction_status": rec.get("extraction_status", "metadata_only"),
                    "skipped_reason": rec.get("skipped_reason"),
                    "category": rec.get("category") or (rec.get("ai") or {}).get("ai_fast", {}).get("category"),
                    "domain": rec.get("domain") or (rec.get("ai") or {}).get("ai_fast", {}).get("domain"),
                    "tags": rec.get("tags") or (rec.get("ai") or {}).get("ai_fast", {}).get("tags") or [],
                    "summary": (rec.get("ai") or {}).get("ai_fast", {}).get("summary") or "",
                    "contains_pii": (rec.get("ai") or {}).get("ai_fast", {}).get("contains_pii"),
                    "confidence": (rec.get("ai") or {}).get("ai_fast", {}).get("confidence"),
                    "ai_cost_usd": rec.get("ai_cost_usd"),
                    "text": text,
                }
                self.items.append(item)

        self.df = pd.DataFrame(self.items)
        self._build_chunks()
        self._load_graph_edges(os.path.join(rdir, "graph_edges.jsonl"))

    def _build_chunks(self) -> None:
        self.chunks.clear()
        if not self.items:
            return
        chunk_chars = max(200, settings.RAG_CHUNK_CHARS)
        overlap = max(0, settings.RAG_OVERLAP)
        for it in self.items:
            txt = (it.get("text") or "").strip()
            if not txt:
                continue
            start = 0
            n = len(txt)
            while start < n:
                end = min(n, start + chunk_chars)
                chunk = txt[start:end]
                self.chunks.append({
                    "run_id": it["run_id"],
                    "path": it["path"],
                    "filename": it["filename"],
                    "ext": it["ext"],
                    "text": chunk,
                })
                if end == n:
                    break
                start = max(end - overlap, end)

    def _load_graph_edges(self, path: str) -> None:
        self.graph_edges.clear()
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                src = rec.get("src") or rec.get("source")
                dst = rec.get("dst") or rec.get("target")
                if not src or not dst:
                    continue
                self.graph_edges.append({
                    "src": src, "dst": dst,
                    "type": rec.get("type", "related"),
                    "weight": float(rec.get("weight", 0.5))
                })