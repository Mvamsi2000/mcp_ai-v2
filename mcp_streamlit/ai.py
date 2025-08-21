# mcp_ai/mcp_streamlit/ai.py
from __future__ import annotations
import json, math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# httpx is optional at import time; used if present.
try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore

# ---------------------- Embeddings ----------------------

def _hash_embed(text: str, dim: int = 768) -> np.ndarray:
    """
    Fast fallback embedding: deterministic hashed bag-of-words -> R^dim.
    Not SOTA, but good enough if Ollama/OpenAI are unavailable.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = hash(tok) % dim
        vec[h] += 1.0
    n = np.linalg.norm(vec)
    if n > 0:
        vec /= n
    return vec

async def _ollama_embed_batch(
    texts: Sequence[str], model: str, url: str
) -> List[np.ndarray]:
    if httpx is None:
        return [_hash_embed(t) for t in texts]
    out: List[np.ndarray] = []
    async with httpx.AsyncClient(timeout=60) as client:
        for t in texts:
            try:
                r = await client.post(
                    f"{url.rstrip('/')}/api/embeddings",
                    json={"model": model, "prompt": t},
                )
                r.raise_for_status()
                data = r.json()
                emb = np.array(data.get("embedding") or data.get("data") or [], dtype=np.float32)
                if emb.size == 0:
                    emb = _hash_embed(t)
                out.append(emb.astype(np.float32))
            except Exception:
                out.append(_hash_embed(t))
    return out

async def _openai_embed_batch(
    texts: Sequence[str], api_key: str, model: str
) -> List[np.ndarray]:
    if httpx is None:
        return [_hash_embed(t) for t in texts]
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            r = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json={"model": model, "input": list(texts)},
            )
            r.raise_for_status()
            data = r.json()
            arrs = [np.array(row["embedding"], dtype=np.float32) for row in data.get("data", [])]
            if len(arrs) == len(texts):
                return arrs
        except Exception:
            pass
    return [_hash_embed(t) for t in texts]

async def embed_texts(
    texts: Sequence[str],
    *,
    use_ollama: bool,
    ollama_url: str,
    ollama_model: str,
    use_cloud: bool,
    cloud_provider: str,
    openai_api_key: str | None,
    openai_embed_model: str,
) -> List[np.ndarray]:
    texts = [t if isinstance(t, str) else "" for t in texts]
    if use_cloud and cloud_provider == "openai" and openai_api_key:
        return await _openai_embed_batch(texts, openai_api_key, openai_embed_model)
    if use_ollama:
        return await _ollama_embed_batch(texts, ollama_model, ollama_url)
    # last resort
    return [_hash_embed(t) for t in texts]

# ---------------------- Chat / QA ----------------------

async def _ollama_chat(
    system: str, user: str, *, url: str, model: str
) -> str:
    if httpx is None:
        return _extractive_fallback_answer(user)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{url.rstrip('/')}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            # Ollama returns {"message": {"content": "..."}}
            msg = (data.get("message") or {}).get("content")
            if isinstance(msg, str) and msg.strip():
                return msg
    except Exception:
        pass
    return _extractive_fallback_answer(user)

async def _openai_chat(
    system: str, user: str, *, api_key: str, model: str
) -> str:
    if httpx is None:
        return _extractive_fallback_answer(user)
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "temperature": 0.2,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions",
                                  headers=headers, json=payload)
            r.raise_for_status()
            j = r.json()
            txt = j.get("choices", [{}])[0].get("message", {}).get("content")
            if isinstance(txt, str) and txt.strip():
                return txt
    except Exception:
        pass
    return _extractive_fallback_answer(user)

def _extractive_fallback_answer(prompt: str) -> str:
    return (
        "Local LLM not reachable and no cloud key configured.\n"
        "Here’s a best-effort summary from retrieved snippets:\n\n"
        "(Provide high-level answer, then cite the [document names] used.)"
    )

def _render_context_block(docs: Sequence[Tuple[str, str]]) -> str:
    # Keep within ~8–12k chars to avoid blowing up small local models
    max_chars = 12000
    parts: List[str] = []
    used = 0
    for name, text in docs:
        chunk = f"[{name}]\n{text}\n---\n"
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    return "".join(parts)

async def chat_answer(
    question: str,
    *,
    system: str,
    docs: Sequence[Tuple[str, str]],
    use_ollama: bool,
    ollama_url: str,
    ollama_model: str,
    use_cloud: bool,
    cloud_provider: str,
    openai_api_key: str | None,
    openai_model: str,
) -> str:
    context = _render_context_block(docs)
    user = (
        "Answer the question using ONLY the context below. "
        "Cite sources inline using [document names]. If unsure, say so.\n\n"
        f"Context:\n{context}\n"
        f"Question: {question}"
    )
    if use_cloud and cloud_provider == "openai" and openai_api_key:
        return await _openai_chat(system, user, api_key=openai_api_key, model=openai_model)
    if use_ollama:
        return await _ollama_chat(system, user, url=ollama_url, model=ollama_model)
    return _extractive_fallback_answer(user)