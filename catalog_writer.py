# mcp_ai/catalog_writer.py
from __future__ import annotations

import os, json, csv, time, hashlib
from typing import Dict, Any, List, Optional

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _file_exists_and_nonempty(p: str) -> bool:
    return os.path.exists(p) and os.path.getsize(p) > 0

def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        # store lists as semicolon-separated for CSV readability
        return "; ".join(map(_safe_str, v))
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def _canonical_fields(row: Dict[str, Any]) -> List[str]:
    """
    Produce a stable, readable CSV header. We prioritize common fields
    and then include any extras (sorted) to avoid dropping information.
    """
    preferred = [
        "run_id", "path", "filename", "ext",
        "extraction_status", "skipped_reason",
        "category", "domain", "tags",
        "contains_pii", "confidence",
        "summary", "pii", "glossary_terms",
        "ai_cost_usd", "sha1", "engine", "pages", "language",
    ]
    keys = list(row.keys())
    # keep preferred order, then append any others
    out: List[str] = [k for k in preferred if k in keys]
    for k in sorted(keys):
        if k not in out:
            out.append(k)
    return out

class CatalogWriter:
    """
    Writes catalog rows to both global files and optional per-run files.
    - Global JSONL / CSV accumulate all runs.
    - Per-run JSONL / CSV live under out_root/runs/<run_id>/.
    """

    def __init__(self, cfg: Dict[str, Any], run_id: str):
        self.cfg = cfg or {}
        self.run_id = run_id

        storage = (self.cfg.get("storage") or {})
        out_root = storage.get("out_root") or "./mcp_ai/output_files"
        _ensure_dir(out_root)
        self.out_root = out_root

        # Global catalog files
        self.global_jsonl = storage.get("catalog_jsonl") or os.path.join(out_root, "metadata_catalog.jsonl")
        self.global_csv   = storage.get("catalog_csv")   or os.path.join(out_root, "metadata_catalog.csv")

        # Per-run outputs
        self.per_run = bool(storage.get("per_run_outputs", True))
        if self.per_run:
            self.run_dir = os.path.join(out_root, "runs", self.run_id)
            _ensure_dir(self.run_dir)
            self.run_jsonl = os.path.join(self.run_dir, "items.jsonl")
            self.run_csv   = os.path.join(self.run_dir, "items.csv")
            self.run_meta  = os.path.join(self.run_dir, "run_meta.json")
            # write basic run metadata up front
            self._write_run_meta()
        else:
            self.run_dir = ""
            self.run_jsonl = ""
            self.run_csv = ""
            self.run_meta = ""

        # counters
        self._count = 0

    def _write_run_meta(self) -> None:
        meta = {
            "run_id": self.run_id,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "config_hash": self._config_hash(),
            "out_root": self.out_root,
        }
        with open(self.run_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _config_hash(self) -> str:
        try:
            s = json.dumps(self.cfg, sort_keys=True, ensure_ascii=False)
        except Exception:
            s = str(self.cfg)
        return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

    def _append_jsonl(self, path: str, row: Dict[str, Any]) -> None:
        _ensure_dir(os.path.dirname(path))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _append_csv(self, path: str, row: Dict[str, Any]) -> None:
        _ensure_dir(os.path.dirname(path))
        write_header = not _file_exists_and_nonempty(path)
        # choose a stable header set
        fieldnames = _canonical_fields(row)
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            # normalize row to header fields
            flat = {k: _safe_str(row.get(k)) for k in fieldnames}
            w.writerow(flat)

    def write_item(self, row: Dict[str, Any]) -> None:
        """
        row should already contain file/extraction + AI fields.
        We add run_id and then write to run/global sinks.
        """
        row = dict(row or {})
        row.setdefault("run_id", self.run_id)

        # JSONL: global + per-run
        self._append_jsonl(self.global_jsonl, row)
        if self.per_run and self.run_jsonl:
            self._append_jsonl(self.run_jsonl, row)

        # CSV: global + per-run
        self._append_csv(self.global_csv, row)
        if self.per_run and self.run_csv:
            self._append_csv(self.run_csv, row)

        self._count += 1

    def finalize(self) -> None:
        # update run_meta with count + finished_at
        if self.per_run and self.run_meta:
            try:
                with open(self.run_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {"run_id": self.run_id}
            meta["items_written"] = self._count
            meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            with open(self.run_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)