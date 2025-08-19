# mcp_ai/cloud_stage.py
from __future__ import annotations
import argparse, json, os, shutil
from typing import Any, Dict, Iterable, List, Tuple

from .utils import load_yaml, ensure_dir

DEFAULT_RULES = {
    "min_confidence": 0.75,     # stage if fast_conf < this
    "include_pii": True,        # stage if contains_pii
    "include_high_value": True, # stage if text looks like high-value (set by ai_router heuristics)
    "max_files": 10_000,        # safety cap
    "action": "copy",           # copy | move | symlink
}

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

def should_stage(rec: Dict[str, Any], rules: Dict[str, Any]) -> Tuple[bool, str]:
    ai = (rec.get("ai") or {})
    fast = (ai.get("ai_fast") or {})
    deep = (ai.get("ai_deep_local") or {})
    conf = float(fast.get("confidence") or 0.0)
    contains_pii = bool(fast.get("contains_pii"))
    # when ai_router flagged high_value it decided to run deep; we approximate via presence of deep
    high_value = bool(deep)

    reasons: List[str] = []
    if conf < float(rules.get("min_confidence", 0.75)):
        reasons.append(f"low_conf({conf:.2f})")
    if rules.get("include_pii", True) and contains_pii:
        reasons.append("pii")
    if rules.get("include_high_value", True) and high_value:
        reasons.append("high_value")

    return (len(reasons) > 0, ",".join(reasons))

def stage_path(dst_root: str, src_path: str) -> str:
    # preserve filename; flatten dirs by mirroring relative path under a hashed bucket if needed
    base = os.path.basename(src_path)
    return os.path.join(dst_root, base)

def perform_stage(src: str, dst: str, action: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if action == "copy":
        shutil.copy2(src, dst)
    elif action == "move":
        shutil.move(src, dst)
    elif action == "symlink":
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
        except OSError:
            # fallback to copy on platforms without symlink permissions
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown action: {action}")

def main() -> None:
    ap = argparse.ArgumentParser("mcp_ai.cloud_stage")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dst", default="./mcp_ai/output_files/cloud_stage", help="Destination staging folder")
    ap.add_argument("--rules", default=None, help="JSON string or file path with rules")
    ap.add_argument("--run-id", default=None, help="Stage from a specific run; default includes all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    rules: Dict[str, Any] = dict(DEFAULT_RULES)
    if args.rules:
        if os.path.exists(args.rules):
            with open(args.rules, "r", encoding="utf-8") as f:
                rules.update(json.load(f))
        else:
            rules.update(json.loads(args.rules))

    catalog = (cfg.get("storage") or {}).get("catalog_jsonl","./mcp_ai/output_files/metadata_catalog.jsonl")
    ensure_dir(args.dst)

    manifest: List[Dict[str, Any]] = []
    count = 0
    for rec in iter_jsonl(catalog):
        if args.run-id and rec.get("run_id") != args.run_id:  # noqa: E999 (dash in attribute)
            continue
        ok, why = should_stage(rec, rules)
        if not ok:
            continue
        src = rec.get("path","")
        if not src or not os.path.exists(src):
            continue
        dst = stage_path(args.dst, src)
        manifest.append({"src": src, "dst": dst, "reasons": why})
        count += 1
        if not args.dry_run:
            perform_stage(src, dst, rules.get("action","copy"))
        if count >= int(rules.get("max_files", 10_000)):
            break

    man_path = os.path.join(args.dst, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump({"rules": rules, "items": manifest}, f, indent=2, ensure_ascii=False)

    print(json.dumps({"staged": len(manifest), "dst": args.dst, "manifest": man_path, "dry_run": args.dry_run}, indent=2))

if __name__ == "__main__":
    main()