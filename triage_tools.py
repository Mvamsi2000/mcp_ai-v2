# mcp_ai/triage_tools.py
from __future__ import annotations
import re
from typing import Any, Dict, Tuple

PII_EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PII_PHONE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")
PII_ADDRESS_HINTS = ("street", "ave", "avenue", "road", "rd", "lane", "ln", "boulevard", "blvd", "way", "apt", "suite", "po box")

def quick_pii_flags(text: str) -> Dict[str, bool]:
    t = text or ""
    flags = {
        "email": bool(PII_EMAIL.search(t)),
        "phone": bool(PII_PHONE.search(t)),
        "address": any(word in t.lower() for word in PII_ADDRESS_HINTS),
    }
    return flags

def is_high_value(text: str) -> bool:
    """
    Mark as high-value if doc looks like a structured business artifact.
    """
    if not text:
        return False
    t = text.lower()
    hits = 0
    for kw in ("invoice", "purchase order", "contract", "nda", "balance sheet", "p&l", "budget", "roadmap", "architecture", "runbook"):
        if kw in t:
            hits += 1
    return hits >= 1