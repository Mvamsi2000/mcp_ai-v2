# mcp_ai/debug_extract.py
from __future__ import annotations
import argparse, json, os
from typing import Any, Dict

from .utils import load_yaml, BudgetLedger
from .extractor import extract_content
from .ai_router import ai_enrich

def main() -> None:
    ap = argparse.ArgumentParser("mcp_ai.debug_extract")
    ap.add_argument("path", help="File to extract")
    ap.add_argument("--config", default="./mcp_ai/config.yaml")
    ap.add_argument("--ai", action="store_true", help="Run AI enrichment after extraction")
    ap.add_argument("--print-text", action="store_true", help="Print extracted text to stdout")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    prof = (cfg.get("profiles", {}) or {}).get(cfg.get("scan_profile","deep"), {}) or {}
    tpol = (cfg.get("type_policies", {}) or {})

    ex = extract_content(args.path, prof, tpol)
    print(json.dumps({
        "extraction_status": ex.get("extraction_status"),
        "skipped_reason": ex.get("skipped_reason"),
        "meta": ex.get("meta", {}),
        "text_len": len(ex.get("text","")),
    }, indent=2))

    if args.print_text:
        print("\n--- TEXT START ---\n")
        print(ex.get("text",""))
        print("\n--- TEXT END ---\n")

    if args.ai:
        text = ex.get("text","") or ""
        ai = ai_enrich(text, cfg, "debug", BudgetLedger(5.0, 0.1))
        print(json.dumps({"ai": ai}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()