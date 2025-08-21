# mcp_ai/mcp_streamlit/ai.py
from __future__ import annotations
import os, hashlib, math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import httpx

try:
    from openai import OpenAI  # optional
except Exception:
    OpenAI = None  # type: ignore

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_norm(a), _norm(b)))

# -------------------- Embeddings --------------------

async def embed_texts(
    texts: List[str],
    *,
    use_ollama: bool,
    ollama_url: str,
    ollama_model: str,
    use_cloud: bool,
    cloud_provider: str,
    openai_api_key: Optional[str],
    openai_embed_model: str,
) -> np.ndarray:
    """
    Returns 2D numpy array [n, d] embeddings.
    Prefers Ollama (local). If unavailable and cloud enabled, use OpenAI.
    If neither works, falls back to a deterministic hash-embedding baseline.
    """
    if use_ollama:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{ollama_url.rstrip('/')}/api/embeddings",
                    json={"model": ollama_model, "prompt": "\n\n".join(texts)},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # Ollama returns a single embedding for the whole prompt; we want per-chunk.
                    # So do it per-text if n>1:
                    if len(texts) == 1:
                        vec = np.array(data.get("embedding") or [], dtype=np.float32)
                        return vec.reshape(1, -1)
                    vecs: List[np.ndarray] = []
                    for t in texts:
                        r = await client.post(
                            f"{ollama_url.rstrip('/')}/api/embeddings",
                            json={"model": ollama_model, "prompt": t},
                        )
                        e = (r.json() if r.status_code == 200 else {}).get("embedding") or []
                        vecs.append(np.array(e, dtype=np.float32))
                    return np.vstack(vecs) if vecs else _hash_embed_batch(texts)
        except Exception:
            pass

    if use_cloud and cloud_provider == "openai" and openai_api_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=openai_api_key)
            # OpenAI v1 embeddings are batched; we make a single call
            resp = client.embeddings.create(model=openai_embed_model, input=texts)  # type: ignore
            vecs = [np.array(r.embedding, dtype=np.float32) for r in resp.data]  # type: ignore
            return np.vstack(vecs)
        except Exception:
            pass

    # Deterministic hash fallback — always works (lower quality but robust).
    return _hash_embed_batch(texts)

def _hash_embed_batch(texts: List[str], dim: int = 384) -> np.ndarray:
    mat = []
    for t in texts:
        h = hashlib.sha256((t or "").encode("utf-8", errors="ignore")).digest()
        rnd = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        # tile to desired dim and add small bag-of-words term
        base = np.tile(rnd, int(math.ceil(dim / rnd.shape[0])))[:dim]
        bonus = np.zeros(dim, dtype=np.float32)
        for tok in (t or "").split():
            bonus[hash(tok) % dim] += 1.0
        mat.append(_norm(base + 0.01 * bonus))
    return np.vstack(mat)

# -------------------- Chat / Generate --------------------

async def chat_answer(
    question: str,
    *,
    system: str,
    docs: List[Tuple[str, str]] | None,  # [(doc_id, snippet)]
    use_ollama: bool,
    ollama_url: str,
    ollama_model: str,
    use_cloud: bool,
    cloud_provider: str,
    openai_api_key: Optional[str],
    openai_model: str,
) -> str:
    context = ""
    if docs:
        # Keep prompt compact for small models
        ctx_parts = [f"[{i+1}] {doc_id}\n{snippet}" for i, (doc_id, snippet) in enumerate(docs[:8])]
        context = "\n\n".join(ctx_parts)

    if use_ollama:
        try:
            payload = {
                "model": ollama_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            }
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(f"{ollama_url.rstrip('/')}/api/chat", json=payload)
                if r.status_code == 200:
                    j = r.json()
                    # Newer Ollama returns {message:{content:"..."}}
                    msg = (j.get("message") or {}).get("content") or j.get("response") or ""
                    if msg:
                        return msg.strip()
        except Exception:
            pass

    if use_cloud and OpenAI is not None and openai_api_key:
        try:
            client = OpenAI(api_key=openai_api_key)
            resp = client.chat.completions.create(  # type: ignore
                model=openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()  # type: ignore
        except Exception:
            pass

    # Last-ditch: return a template
    return "I could not reach a language model. Please ensure Ollama is running or provide cloud credentials."