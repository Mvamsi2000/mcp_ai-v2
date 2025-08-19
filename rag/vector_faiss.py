# mcp_ai/vector_faiss.py
from __future__ import annotations
from typing import List, Tuple
from .embeddings_text import cheap_hash_embed

class InMemoryANN:
    def __init__(self):
        self.vecs: List[List[float]] = []
        self.meta: List[Tuple[int, str]] = []

    def add(self, docs: List[str], ids: List[int]) -> None:
        self.vecs.extend(cheap_hash_embed(docs))
        self.meta.extend(list(zip(ids, docs)))

    def topk(self, query: str, k: int = 5) -> List[Tuple[int, float, str]]:
        q = cheap_hash_embed([query])[0]
        def dot(a, b): return sum(x*y for x, y in zip(a, b))
        scores = [(i, dot(q, v), m[1]) for i, (v, m) in enumerate(zip(self.vecs, self.meta))]
        scores.sort(key=lambda x: x[1], reverse=True)
        out = []
        for i, s, _ in scores[:k]:
            out.append((self.meta[i][0], float(s), self.meta[i][1]))
        return out