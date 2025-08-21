from __future__ import annotations
import streamlit as st
import asyncio
from ai import topk_dense, topk_bm25, build_prompt_rag, build_prompt_catalog, generate_answer
from config import settings

DATA = st.session_state["DATA"]

st.title("🤖 Ask AI")
if DATA.df is None or DATA.df.empty:
    st.info("No data. Load a run.")
    st.stop()

mode = st.radio("Mode", ["Catalog Q&A","Content RAG"], horizontal=True)
q = st.text_input("Ask a question about your data/files...", placeholder="e.g., Summarize the finance reports about CHS meeting")
k = st.slider("Top-K", min_value=3, max_value=20, value=settings.RAG_TOP_K, step=1)

if st.button("Run"):
    if not q.strip():
        st.warning("Please enter a question.")
        st.stop()

    if mode == "Catalog Q&A":
        rows = DATA.items
        # Simple BM25 over lightweight fields (filename + summary + tags + domain)
        cat_chunks = []
        for r in rows:
            text = " ".join([
                r.get("filename",""),
                str(r.get("summary","")),
                " ".join(r.get("tags") or []),
                str(r.get("domain",""))
            ])
            cat_chunks.append({"text": text, "filename": r.get("filename",""), "path": r.get("path","")})
        # Prefer dense; fallback to bm25
        top, dbg = topk_dense(q, cat_chunks, k)
        if not top:
            top, dbg = topk_bm25(q, cat_chunks, k)
        prompt = build_prompt_catalog(q, [
            {"filename": c["filename"], "summary": "", "tags": "", "domain":"", "owners": "", "path": c.get("path","")}
            for c in top
        ])
    else:
        # Content RAG over text chunks
        chunks = DATA.chunks
        top, dbg = topk_dense(q, chunks, k)
        if not top:
            top, dbg = topk_bm25(q, chunks, k)
        prompt = build_prompt_rag(q, top)

    with st.spinner(f"Asking LLM ({'cloud' if settings.cloud_ready() else 'ollama'})..."):
        ans, meta = asyncio.get_event_loop().run_until_complete(generate_answer(prompt))

    st.subheader("Answer")
    st.write(ans)

    st.divider()
    st.subheader("Citations")
    if mode == "Catalog Q&A":
        for c in top:
            st.write(f"- **{c.get('filename','?')}** — {c.get('path','')}")
    else:
        for c in top:
            st.write(f"- **{c.get('filename','?')}** — score {c.get('score',0):.3f}")

    with st.expander("Debug"):
        st.json({"retriever": dbg, "llm": meta})