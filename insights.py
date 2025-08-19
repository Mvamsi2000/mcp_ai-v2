from __future__ import annotations

import time, os
from typing import Dict, Any

def _readiness_from_category(cat: str) -> str:
    c = (cat or "").lower()
    if any(k in c for k in ("invoice","budget","report","contract","nda","policy")):
        return "Green (Ready)"
    if any(k in c for k in ("draft","proposal","slides","presentation","assessment","questionnaire")):
        return "Yellow (Needs Review)"
    return "Unknown"

def _sensitivity_from_flags(contains_pii: bool, domain: str) -> str:
    if contains_pii:
        return "Confidential"
    if domain in ("finance","legal","hr"):
        return "Internal"
    return "Public"

def _years_since(epoch_seconds_str: str) -> float:
    try:
        # The caller provides RFC3339 like "2025-08-17T12:34:56Z"
        # We only need approximate years → rely on file stats when available in row.
        return 0.0
    except Exception:
        return 0.0

def compute_insights_for_file(row: Dict[str, Any], archive_stale_years: int, sha_first_path: Dict[str,str]) -> Dict[str, Any]:
    """
    Lightweight “business” insights. You can replace with your richer version later.
    """
    category = (row.get("category") or "unknown")
    domain = (row.get("domain") or "general")
    contains_pii = bool(row.get("contains_pii", False))
    readiness = _readiness_from_category(category)
    sensitivity = _sensitivity_from_flags(contains_pii, domain)

    # map SHA → first seen path (simple de-dup signal)
    sha = row.get("sha1")
    first_path = ""
    if sha:
        first_path = sha_first_path.setdefault(sha, row.get("path",""))

    return {
        "readiness_level": readiness,
        "sensitivity": sensitivity,
        "first_seen_path": first_path,
        "stale_years_threshold": archive_stale_years,
    }