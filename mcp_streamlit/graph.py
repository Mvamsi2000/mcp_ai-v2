# mcp_ai/mcp_streamlit/graph.py
from __future__ import annotations
import os, json, pathlib, asyncio
from typing import Dict, Any, List, Tuple
import numpy as np
from .data import load_items_jsonl, get_text_from_item, file_display_name
from .ai import embed_texts, cosine_sim

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
    If graph_edges.jsonl doesn't exist, build it using lightweight semantic similarity.
    Returns (path, num_edges).
    """
    out_path = os.path.join(run_dir, "graph_edges.jsonl")
    if os.path.isfile(out_path):
        # Already present: count edges
        n = sum(1 for _ in open(out_path, "r", encoding="utf-8"))
        return out_path, n

    items = load_items_jsonl(run_dir)
    if not items:
        return out_path, 0

    nodes = []
    texts = []
    for rec in items:
        nid = rec.get("path") or rec.get("filename")
        if not nid:
            continue
        nodes.append(nid)
        texts.append(get_text_from_item(rec) or file_display_name(rec))

    if not nodes:
        return out_path, 0

    vecs = await embed_texts(
        texts,
        use_ollama=use_ollama,
        ollama_url=ollama_url,
        ollama_model=ollama_embed_model,
        use_cloud=use_cloud,
        cloud_provider=cloud_provider,
        openai_api_key=openai_api_key,
        openai_embed_model=openai_embed_model,
    )

    # kNN (symmetric, without self-edges)
    edges: List[Dict[str, Any]] = []
    for i in range(len(nodes)):
        sims: List[Tuple[int, float]] = []
        for j in range(len(nodes)):
            if i == j: 
                continue
            s = cosine_sim(vecs[i], vecs[j])
            if s >= min_sim:
                sims.append((j, s))
        sims.sort(key=lambda t: t[1], reverse=True)
        for j, s in sims[:k]:
            edges.append({"src": nodes[i], "dst": nodes[j], "type": "related", "weight": round(float(s), 3)})

    # Write edges
    with open(out_path, "w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return out_path, len(edges)