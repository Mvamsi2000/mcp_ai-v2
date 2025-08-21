# mcp_ai/mcp_streamlit/graph.py
from __future__ import annotations
import os, json, pathlib
from typing import List, Tuple, Dict, Any

import numpy as np

# absolute (no leading dot) so it works when app.py is run as a script
from config import settings
from ai import embed_texts


def _load_item_summaries(run_dir: str) -> List[Tuple[str, str]]:
    """
    Return [(path, text_for_embedding), ...] using lightweight text so we
    don't OOM on very large files. Prefer ai.fast.summary + filename.
    """
    items_path = os.path.join(run_dir, "items.jsonl")
    if not os.path.isfile(items_path):
        return []
    rows: List[Tuple[str, str]] = []
    with open(items_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            path = rec.get("path") or rec.get("filename") or "unknown"
            filename = pathlib.Path(path).name
            ai_fast = ((rec.get("ai") or {}).get("ai_fast") or {})
            summary = ai_fast.get("summary") or ""
            domain = ai_fast.get("domain") or rec.get("domain") or ""
            # small, stable embedding string (title + domain + summary)
            text = f"{filename}\nDomain: {domain}\n{summary}".strip()
            if not text:
                # fallback to any available text
                raw = (rec.get("ai") or {}).get("_raw_text") or rec.get("text") or ""
                text = f"{filename}\n{raw[:2000]}"
            rows.append((path, text))
    return rows


def ensure_graph_edges(
    run_dir: str,
    k: int = 3,
    min_sim: float = 0.32,
    *,
    use_ollama: bool = True,
    ollama_url: str | None = None,
    ollama_embed_model: str | None = None,
    use_cloud: bool = False,
    cloud_provider: str = "none",
    openai_api_key: str | None = None,
    openai_embed_model: str | None = None,
) -> Tuple[str, int]:
    """
    Build k-NN edges over per-file embeddings (one vector per file) and
    write JSONL edges file. Returns (edges_path, n_edges).
    """
    edges_path = os.path.join(run_dir, "graph_edges.jsonl")
    docs = _load_item_summaries(run_dir)
    if not docs:
        # still create an empty file to satisfy the UI
        with open(edges_path, "w", encoding="utf-8") as f:
            pass
        return edges_path, 0

    labels = [d[0] for d in docs]
    corpus = [d[1] for d in docs]

    vecs = embed_texts(
        corpus,
        use_ollama=use_ollama,
        ollama_url=ollama_url or settings.OLLAMA_URL,
        ollama_model=ollama_embed_model or settings.OLLAMA_EMBED_MODEL,
        use_cloud=use_cloud,
        cloud_provider=cloud_provider,
        openai_api_key=openai_api_key or getattr(settings, "OPENAI_API_KEY", None),
        openai_embed_model=openai_embed_model or settings.OPENAI_EMBED_MODEL,
    )
    # (N, D)
    V = np.array(vecs, dtype=float)
    # Normalize
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    VN = V / norms

    # Cosine similarity matrix (N x N)
    S = VN @ VN.T
    np.fill_diagonal(S, -1.0)  # exclude self

    # For each i, take top-k neighbors j with S[i,j] >= min_sim
    edges: List[Dict[str, Any]] = []
    N = S.shape[0]
    for i in range(N):
        # argsort descending
        idxs = np.argsort(-S[i])[: max(8, k + 2)]
        added = 0
        for j in idxs:
            if j < 0 or j >= N: 
                continue
            sim = float(S[i, j])
            if sim < min_sim:
                continue
            # avoid duplicating both (i->j) and (j->i); only keep i<j
            if i < j:
                edges.append({"src": labels[i], "dst": labels[j], "type": "related", "weight": round(sim, 4)})
                added += 1
            if added >= k:
                break

    # Write JSONL
    with open(edges_path, "w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return edges_path, len(edges)


def build_graph_html(edges_path: str) -> str:
    """
    Small helper for legacy pages/** imports, and also handy for previews.
    """
    from pyvis.network import Network

    net = Network(height="600px", width="100%", bgcolor="#0b1220", font_color="#e2e8f0", directed=False)
    edges: List[Dict[str, Any]] = []
    nodes: set[str] = set()
    if os.path.isfile(edges_path):
        with open(edges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                edges.append(e)
                nodes.add(e["src"]); nodes.add(e["dst"])

    for n in nodes:
        label = pathlib.Path(n).name
        net.add_node(n, label=label, title=n, shape="dot", size=12, color="#1f6feb")

    for e in edges:
        w = e.get("weight", 0.3)
        net.add_edge(e["src"], e["dst"], value=max(1, int(float(w) * 10)), color="#94a3b8")

    net.toggle_physics(True)
    return net.generate_html(notebook=False)