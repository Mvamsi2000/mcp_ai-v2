# mcp_ai/mcp_streamlit/graph.py
from __future__ import annotations
import json, os, pathlib
from typing import Any, Dict, List, Tuple

import numpy as np

from .config import settings
from .data import load_items_jsonl, file_display_name, get_text_from_item
from .ai import embed_texts

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a)); bn = float(np.linalg.norm(b))
    if an <= 1e-9 or bn <= 1e-9: return 0.0
    return float(np.dot(a / an, b / bn))

async def ensure_graph_edges(
    run_dir: str,
    *,
    k: int = 3,
    min_sim: float = 0.32,
    use_ollama: bool,
    ollama_url: str,
    ollama_embed_model: str,
    use_cloud: bool,
    cloud_provider: str,
    openai_api_key: str | None,
    openai_embed_model: str,
) -> Tuple[str, int]:
    """
    Build graph_edges.jsonl (src,dst,weight) if missing; return (path, n_edges).
    """
    out_path = os.path.join(run_dir, "graph_edges.jsonl")
    if os.path.isfile(out_path):
        # count lines
        n = sum(1 for _ in open(out_path, "r", encoding="utf-8", errors="ignore"))
        return out_path, n

    items = load_items_jsonl(run_dir)
    docs: List[Tuple[str, str]] = []  # (path, text)
    for r in items:
        p = r.get("path") or ""
        txt = get_text_from_item(r)
        if not txt:
            # fall back to summary if text missing
            ai = (r.get("ai") or {}).get("ai_fast") or {}
            sm = ai.get("summary") or ""
            if sm:
                txt = sm
        if txt.strip():
            docs.append((p, txt.strip()))
    if not docs:
        pathlib.Path(out_path).write_text("", encoding="utf-8")
        return out_path, 0

    vecs = await embed_texts(
        [t for _, t in docs],
        use_ollama=use_ollama,
        ollama_url=ollama_url,
        ollama_model=ollama_embed_model,
        use_cloud=use_cloud,
        cloud_provider=cloud_provider,
        openai_api_key=openai_api_key,
        openai_embed_model=openai_embed_model,
    )

    # Build symmetric KNN edges with threshold
    edges: List[Tuple[int, int, float]] = []
    n = len(vecs)
    for i in range(n):
        sims: List[Tuple[int, float]] = []
        for j in range(n):
            if i == j: 
                continue
            s = _cos_sim(vecs[i], vecs[j])
            if s >= min_sim:
                sims.append((j, s))
        sims.sort(key=lambda t: t[1], reverse=True)
        for j, s in sims[:k]:
            if i < j:  # prevent duplicates; we’ll add undirected once
                edges.append((i, j, s))

    with open(out_path, "w", encoding="utf-8") as f:
        for i, j, s in edges:
            f.write(json.dumps({"src": docs[i][0], "dst": docs[j][0], "weight": round(float(s), 4)}) + "\n")

    return out_path, len(edges)