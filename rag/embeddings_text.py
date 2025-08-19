# mcp_ai/embeddings_text.py
from __future__ import annotations
from typing import List
import hashlib

def cheap_hash_embed(texts: List[str]) -> List[List[float]]:
    """
    Very cheap, deterministic "embedding" so other code can run without FAISS or cloud.
    DO NOT use in production for semantic search.
    """
    out: List[List[float]] = []
    for t in texts:
        h = hashlib.sha256((t or "").encode("utf-8")).digest()
        # Map 32 bytes -> 8 floats
        vec = [int.from_bytes(h[i:i+4], "big") / 2**32 for i in range(0, 32, 4)]
        out.append(vec)
    return out