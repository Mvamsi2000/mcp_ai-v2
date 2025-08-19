# mcp_ai/metrics.py
from __future__ import annotations
import argparse, collections, json, os, statistics, sys
from typing import Any, Dict, Iterable, List, Tuple

from .utils import load_yaml, now_iso

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def summarize(run_id: str | None, catalog_path: str) -> Dict[str, Any]:
    records = [r for r in iter_jsonl(catalog_path) if not run_id or r.get("run_id")==run_id]
    if not records:
        return {"error":"no records found", "run_id": run_id}

    total = len(records)
    by_status = collections.Counter([r.get("extraction_status","") for r in records])
    domains = collections.Counter([((r.get("ai",{}) or {}).get("ai_fast",{}) or {}).get("domain","") for r in records])
    contains_pii = sum(1 for r in records if ((r.get("ai",{}) or {}).get("ai_fast",{}) or {}).get("contains_pii"))
    confs = [float(((r.get("ai",{}) or {}).get("ai_fast",{}) or {}).get("confidence",0.0)) for r in records]
    avg_conf = round(statistics.fmean(confs), 3) if confs else 0.0
    deep_used = sum(1 for r in records if (r.get("ai",{}) or {}).get("ai_deep_local"))
    top_tags = collections.Counter()
    for r in records:
        tags = ((r.get("ai",{}) or {}).get("ai_fast",{}) or {}).get("tags",[]) or []
        for t in tags:
            if t:
                top_tags[t] += 1
    top_tags_list = top_tags.most_common(15)
    size_bytes = sum(int(r.get("size_bytes") or 0) for r in records)

    return {
        "run_id": run_id or "all",
        "total_files": total,
        "size_bytes": size_bytes,
        "by_status": dict(by_status),
        "domains": dict(domains),
        "contains_pii_files": contains_pii,
        "avg_fast_confidence": avg_conf,
        "deep_used_files": deep_used,
        "top_tags": top_tags_list,
        "generated_at": now_iso(),
    }

def main() -> None:
    ap = argparse.ArgumentParser("mcp_ai.metrics")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None, help="Optional run id to summarize; default aggregates all")
    ap.add_argument("--out", default=None, help="Optional output JSON path")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    catalog = (cfg.get("storage") or {}).get("catalog_jsonl","./mcp_ai/output_files/metadata_catalog.jsonl")
    rep = summarize(args.run_id, catalog)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()