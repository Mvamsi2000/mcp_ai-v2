# mcp_ai/redaction.py
from __future__ import annotations
import re
from typing import Dict, Any, Tuple

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4})")
ID_RE = re.compile(r"\b(?:SSN|SIN|Tax ID|Passport|DL)[^\n:]*:\s*([A-Za-z0-9\-]+)\b", re.IGNORECASE)

def mask(text: str) -> Tuple[str, Dict[str, int]]:
    counts = {"email": 0, "phone": 0, "id": 0}
    def email_sub(m):
        counts["email"] += 1
        return "[EMAIL_REDACTED]"
    def phone_sub(m):
        counts["phone"] += 1
        return "[PHONE_REDACTED]"
    def id_sub(m):
        counts["id"] += 1
        return "ID:[REDACTED]"
    text = EMAIL_RE.sub(email_sub, text)
    text = PHONE_RE.sub(phone_sub, text)
    text = ID_RE.sub(id_sub, text)
    return text, counts
