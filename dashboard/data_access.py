# mcp_ai/dashboard/data_access.py
from __future__ import annotations
import json, os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_dir(out_root: str, run_id: str) -> str:
    return os.path.join(out_root, "runs", run_id)


def find_runs(out_root: str) -> List[Dict[str, Any]]:
    runs_root = os.path.join(out_root, "runs")
    if not os.path.isdir(runs_root):
        return []
    out = []
    for name in sorted(os.listdir(runs_root), reverse=True):
        rd = os.path.join(runs_root, name)
        if not os.path.isdir(rd):
            continue
        csvp = os.path.join(rd, "items.csv")
        jsonlp = os.path.join(rd, "items.jsonl")
        files = 0
        if os.path.isfile(csvp):
            try:
                files = sum(1 for _ in open(csvp, "r", encoding="utf-8")) - 1  # minus header
                files = max(0, files)
            except Exception:
                files = 0
        elif os.path.isfile(jsonlp):
            try:
                files = sum(1 for _ in open(jsonlp, "r", encoding="utf-8"))
            except Exception:
                files = 0
        out.append({"run_id": name, "files": files})
    return out


def _flatten_ai(ai: Dict[str, Any]) -> Dict[str, Any]:
    if not ai:
        return {}
    fast = ai.get("ai_fast") or {}
    deep = ai.get("ai_deep_local") or ai.get("ai_deep") or {}
    out = {
        # fast
        "ai_fast_category": fast.get("category"),
        "ai_fast_domain": fast.get("domain"),
        "ai_fast_tags": fast.get("tags"),
        "ai_fast_contains_pii": fast.get("contains_pii"),
        "ai_fast_confidence": fast.get("confidence"),
        "ai_fast_summary": fast.get("summary"),
        # deep
        "ai_deep_summary": deep.get("summary"),
        "ai_deep_glossary_terms": deep.get("glossary_terms"),
        "ai_deep_pii": deep.get("pii"),
        "ai_deep_confidence": deep.get("confidence"),
    }
    return out


def _flatten_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    if not meta:
        return {}
    return {
        "meta_ext": meta.get("ext"),
        "meta_engine": meta.get("engine"),
        "meta_pages": meta.get("pages"),
        "meta_dpi": meta.get("dpi"),
        "meta_language": meta.get("language"),
    }


def flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    # If CSVs are already flattened we still harmonize keys
    base = {
        "run_id": rec.get("run_id"),
        "path": rec.get("path"),
        "size_bytes": rec.get("size_bytes"),
        "sha1": rec.get("sha1"),
        "extraction_status": rec.get("extraction_status"),
        "skipped_reason": rec.get("skipped_reason"),
        "timestamp": rec.get("timestamp"),
    }
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    ai = rec.get("ai") if isinstance(rec.get("ai"), dict) else {}

    base.update(_flatten_meta(meta))
    base.update(_flatten_ai(ai))
    return base


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # skip malformed lines
                continue
    return rows


def load_run_df(out_root: str, run_id: str) -> pd.DataFrame:
    rd = _run_dir(out_root, run_id)
    csvp = os.path.join(rd, "items.csv")
    jsonlp = os.path.join(rd, "items.jsonl")

    if os.path.isfile(csvp):
        try:
            df = pd.read_csv(csvp, low_memory=False)
            # If CSV is already flattened, return as is
            # Ensure known columns exist even if missing
            for c in ["timestamp", "path"]:
                if c not in df.columns:
                    df[c] = None
            return df
        except Exception:
            pass

    # fallback to JSONL
    if os.path.isfile(jsonlp):
        rows = _read_jsonl(jsonlp)
        flat = [flatten_record(r) for r in rows]
        if not flat:
            return pd.DataFrame(columns=["path"])
        return pd.DataFrame(flat)

    # nothing
    return pd.DataFrame(columns=["path"])


def _contains_any(s: str, needles: List[str]) -> bool:
    s2 = (s or "").lower()
    return any(n.lower() in s2 for n in needles)


def filter_df(
    df: pd.DataFrame,
    *,
    query: str = "",
    domains: Optional[List[str]] = None,
    min_conf: float = 0.0,
    pii_only: bool = False,
    only_needs_deep: bool = False,
    needs_deep_mask: Optional[pd.Series] = None,
    top_folder: Optional[str] = None,
    common_root: Optional[str] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    work = df.copy()

    # folder filter
    if top_folder:
        base = common_root or ""
        def _rel_top(p: str) -> str:
            try:
                rel = os.path.relpath(p, base) if base else p
            except Exception:
                rel = p
            parts = [pp for pp in rel.split(os.sep) if pp and pp != "."]
            return parts[0] if parts else ""
        work = work[work["path"].astype(str).apply(_rel_top) == top_folder]

    # domains
    if domains:
        work = work[work.get("ai_fast_domain", "").astype(str).isin(domains)]

    # confidence
    if min_conf > 0:
        conf = pd.to_numeric(work.get("ai_fast_confidence"), errors="coerce").fillna(0.0)
        work = work[conf >= float(min_conf)]

    # pii
    if pii_only:
        pii = work.get("ai_fast_contains_pii")
        if pii is not None:
            work = work[pii.fillna(False)]
        else:
            work = work.iloc[0:0]

    # query
    if query and query.strip():
        q = query.strip()
        # basic OR support: split by ' OR ' or '|'
        ors = [x.strip() for x in (q.replace("|", " OR ").split(" OR ")) if x.strip()]
        if ors:
            mask = pd.Series(False, index=work.index)
            searchable_cols = ["path", "ai_fast_summary", "ai_fast_category", "ai_fast_tags"]
            def row_text(row) -> str:
                parts: List[str] = []
                for c in searchable_cols:
                    val = row.get(c)
                    if isinstance(val, list):
                        parts.extend([str(v) for v in val])
                    elif pd.notna(val):
                        parts.append(str(val))
                return " ".join(parts)
            texts = work.apply(row_text, axis=1)
            for tok in ors:
                mask = mask | texts.str.lower().str.contains(tok.lower(), na=False)
            work = work[mask]

    # needs deep
    if only_needs_deep and needs_deep_mask is not None:
        work = work[needs_deep_mask.reindex(work.index, fill_value=False)]

    return work.reset_index(drop=True)


def stage_manifest_path(run_dir: str) -> str:
    return os.path.join(run_dir, "stage_manifest.jsonl")


def write_manifest(path: str, items: List[str]) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps({"path": it}) + "\n")
        return True
    except Exception:
        return False