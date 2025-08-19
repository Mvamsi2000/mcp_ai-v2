# mcp_ai/chunker.py
from __future__ import annotations
from typing import List

def chunk_text(text: str, max_chars: int = 8000, overlap: int = 200) -> List[str]:
    if max_chars <= 0:
        return [text]
    chunks = []
    i = 0
    n = len(text)
    step = max(1, max_chars - overlap)
    while i < n:
        chunks.append(text[i : i + max_chars])
        i += step
    return chunks