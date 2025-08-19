# mcp_ai/main.py
from __future__ import annotations

import argparse, json, logging, os, time, uuid
from typing import Any, Dict

from .utils import load_yaml, setup_logging, BudgetLedger, now_iso, ensure_dir
from .scanner import scan_files                # assumes it yields dicts with "path", "size_bytes", "sha1"
from .extractor import extract_content         # returns dict with "extraction_status", "skipped_reason", "text", "meta"
from .ai_router import ai_enrich               # ai_enrich(text, cfg, doc_id, budget, high_value=False, decision_trace=False)
from .catalog_writer import CatalogWriter      # per-run + global outputs

log = logging.getLogger("main")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("mcp_ai")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--profile", default=None, help="Override scan_profile (basic|standard|deep)")
    p.add_argument("--always-deep", action="store_true", help="Force deep pass on every file (ai.deep.policy=always)")
    p.add_argument("--dry-run", action="store_true", help="Extract only; skip AI and writing")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N files")
    return p.parse_args()


def _new_run_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def main() -> None:
    args = _parse_args()
    cfg = load_yaml(args.config)

    # Apply simple overrides
    if args.profile:
        cfg["scan_profile"] = args.profile
    if args.always_deep:
        cfg.setdefault("ai", {}).setdefault("deep", {})["policy"] = "always"

    setup_logging(cfg)
    log.info("Config loaded from %s", args.config)

    # Prepare run context
    run_id = _new_run_id()
    storage = (cfg.get("storage") or {})
    out_root = storage.get("out_root") or "./mcp_ai/output_files"
    ensure_dir(out_root)

    # Budgets
    bcfg = (cfg.get("budgets") or {})
    ledger = BudgetLedger(
        float(bcfg.get("run_usd", 25.0)),
        float(bcfg.get("per_file_usd", 0.03)),
    )

    # Writer (per-run + global)
    writer = CatalogWriter(cfg, run_id)

    # Choose extraction profile & type policies
    profile_name = (cfg.get("scan_profile") or "deep").lower()
    profiles = (cfg.get("profiles") or {})
    profile = profiles.get(profile_name, {}) or {}
    type_policies = (cfg.get("type_policies") or {})

    log.info("Run %s started (profile=%s, always_deep=%s, dry_run=%s)", run_id, profile_name, args.always_deep, args.dry_run)

    processed = 0
    t0 = time.time()

    for f in scan_files(cfg):
        if args.limit and processed >= args.limit:
            break
        processed += 1

        path = f.get("path")
        log.info("→ [%d] %s", processed, path)

        # 1) EXTRACT
        try:
            extres = extract_content(path, profile, type_policies)
        except Exception as e:
            log.warning("Extraction failed on %s: %r", path, e)
            # Write a minimal error row so the run is traceable
            writer.write_item({
                "run_id": run_id,
                "path": path,
                "filename": os.path.basename(path or ""),
                "extraction_status": "error",
                "skipped_reason": str(e),
                "timestamp": now_iso(),
            })
            continue

        text = extres.get("text") or ""
        meta = extres.get("meta") or {}

        # 2) AI (unless dry-run)
        ai: Dict[str, Any] = {}
        if not args.dry_run:
            try:
                # Capture decision trace to understand why deep ran
                ai = ai_enrich(text, cfg, doc_id=path, budget=ledger, high_value=False, decision_trace=True)
            except Exception as e:
                log.warning("AI enrich failed on %s: %r", path, e)
                ai = {}

        # Optional: print why deep ran (if ai_router returns a _decision field)
        if ai.get("_decision"):
            log.info("   deep decision: %s", ai["_decision"])

        # 3) WRITE (flatten key fields for CSV, store full ai/json for JSONL)
        row: Dict[str, Any] = {
            "run_id": run_id,
            "path": path,
            "filename": os.path.basename(path or ""),
            "size_bytes": f.get("size_bytes"),
            "sha1": f.get("sha1") or meta.get("sha1"),
            "ext": meta.get("ext"),
            "engine": meta.get("engine"),
            "pages": meta.get("pages"),
            "language": meta.get("language"),

            "extraction_status": extres.get("extraction_status"),
            "skipped_reason": extres.get("skipped_reason"),

            # Flattened FAST fields (for easy CSV analytics)
            "category": (ai.get("ai_fast") or {}).get("category"),
            "domain": (ai.get("ai_fast") or {}).get("domain"),
            "tags": (ai.get("ai_fast") or {}).get("tags"),
            "contains_pii": (ai.get("ai_fast") or {}).get("contains_pii"),
            "confidence": (ai.get("ai_fast") or {}).get("confidence"),
            "summary": (ai.get("ai_fast") or {}).get("summary"),

            # Flattened DEEP fields
            "pii": (ai.get("ai_deep_local") or {}).get("pii"),
            "glossary_terms": (ai.get("ai_deep_local") or {}).get("glossary_terms"),

            # Cost (if your router tracks it)
            "ai_cost_usd": ai.get("ai_cost_usd"),

            # Full AI blob (handy in JSONL)
            "ai": ai,

            "timestamp": now_iso(),
        }

        writer.write_item(row)

    # Finalize run
    writer.finalize()
    dt = time.time() - t0
    log.info("Run %s completed. Files: %d, elapsed: %.2fs", run_id, processed, dt)

    # Print machine-readable summary (shows per-run dir if enabled)
    out = {
        "run_id": run_id,
        "files": processed,
        "elapsed_s": round(dt, 2),
        "global_jsonl": writer.global_jsonl,
        "global_csv": writer.global_csv,
    }
    if writer.per_run:
        out["run_dir"] = os.path.join(writer.out_root, "runs", run_id)
        out["run_jsonl"] = writer.run_jsonl
        out["run_csv"] = writer.run_csv
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()