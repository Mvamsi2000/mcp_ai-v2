from __future__ import annotations
from datetime import datetime

def human_bytes(n: int | None) -> str:
    if n is None:
        return "-"
    units = ["B","KB","MB","GB","TB"]
    s = float(n)
    for u in units:
        if s < 1024.0:
            return f"{s:.1f}{u}"
        s /= 1024.0
    return f"{s:.1f}PB"

def ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")