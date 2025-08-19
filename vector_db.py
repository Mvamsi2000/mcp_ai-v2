# mcp_ai/vector_db.py
from __future__ import annotations
from typing import List, Tuple
from .vector_faiss import InMemoryANN

class VectorDB:
    def __init__(self):
        self.idx = InMemoryANN()

    def add_texts(self, texts: List[str], ids: List[int]) -> None:
        self.idx.add(texts, ids)

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float, str]]:
        return self.idx.topk(query, k=k)