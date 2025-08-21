# mcp_ai/mcp_streamlit/app.py
from __future__ import annotations
import os, sys, subprocess, asyncio, json, pathlib, textwrap
from typing import Dict, Any, List, Tuple
import streamlit as st
import numpy as np

# Local imports
HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from config import settings
from data import list_runs, load_items_jsonl, file_display_name, get_text_from_item
from ai import embed_texts, chat_answer, cosine_sim
from graph import ensure_graph_edges

# ---------------- UI Setup ----------------
st.set_page_config(
    page_title="MCP Dash",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
try:
    with open(HERE / "styles.css", "r", encoding="utf-8") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ---------------- Sidebar ----------------
st.sidebar.title("MCP Dash")
st.sidebar.caption("Runs • Catalog • Graph • Ask AI")

with st.sidebar.expander("AI Settings", expanded=False):
    st.write("**Local** (default): Ollama")
    st.code(f"{settings.OLLAMA_URL} · {settings.OLLAMA_MODEL}", language="bash")
    st.write("**Cloud** (toggle in Config.yaml → ai.cloud.enabled)")

runs = list_runs(settings.OUTPUT_ROOT)
if not runs:
    st.sidebar.error("No runs found under OUTPUT_ROOT. Make sure a scan has produced `runs/<id>/items.jsonl`.")
    st.stop()

run_ids = [r[0] for r in runs]
default_index = 0
if settings.RUN_ID and settings.RUN_ID in run_ids:
    default_index = run_ids.index(settings.RUN_ID)
run_id = st.sidebar.selectbox("Run", run_ids, index=default_index)
run_dir = dict(runs)[run_id]

# ---------------- Load Items ----------------
items = load_items_jsonl(run_dir)
if not items:
    st.warning("This run has no items.jsonl or it is empty.")
    st.stop()

# ---------------- Header ----------------
st.markdown(
    f"""
    <h1 style="display:flex;align-items:center;gap:.6rem;">
      <span>📁 {run_id}</span>
      <span style="font-size:0.9rem;color:#94a3b8;">{len(items)} files</span>
    </h1>
    """,
    unsafe_allow_html=True,
)

# ----------- KPIs -----------
ok = sum(1 for r in items if r.get("extraction_status") == "ok")
mo = sum(1 for r in items if r.get("extraction_status") == "metadata_only")
err = sum(1 for r in items if r.get("extraction_status") == "error")
ok_ratio = (ok / len(items)) if items else 0.0
latencies = [r.get("meta", {}).get("elapsed_s") for r in items if isinstance(r.get("meta", {}).get("elapsed_s"), (int, float))]
median_latency = float(np.median(latencies)) if latencies else None

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**Success**")
    st.markdown(f"<div class='metric-card'><h2>{ok}/{len(items)}</h2><div>({ok_ratio:.0%})</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("**Metadata-only**")
    st.markdown(f"<div class='metric-card'><h2>{mo}</h2></div>", unsafe_allow_html=True)
with c3:
    st.markdown("**Errors**")
    st.markdown(f"<div class='metric-card'><h2>{err}</h2></div>", unsafe_allow_html=True)
with c4:
    st.markdown("**Median latency (s)**")
    st.markdown(f"<div class='metric-card'><h2>{median_latency if median_latency is not None else '—'}</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- Tabs ----------------
tab_overview, tab_catalog, tab_file, tab_graph, tab_ask, tab_scan = st.tabs(
    ["Overview", "Catalog", "File Inspector", "Graph", "Ask AI", "Scan"]
)

# -------- Overview ----------
with tab_overview:
    st.subheader("Overview")
    st.write("Quick view of the run quality, domains, and types.")
    # Simple domain histogram
    domains: Dict[str, int] = {}
    for r in items:
        d = (r.get("domain") or (r.get("ai", {}).get("ai_fast") or {}).get("domain") or "general")
        domains[d] = domains.get(d, 0) + 1
    if domains:
        domain_rows = sorted(domains.items(), key=lambda t: t[1], reverse=True)
        st.table({"Domain": [d for d,_ in domain_rows], "Files": [c for _,c in domain_rows]})
    else:
        st.info("No domain info available.")

# -------- Catalog ----------
with tab_catalog:
    st.subheader("Catalog")
    q = st.text_input("Filter by filename/path/tag/summary:")
    def visible(rec: Dict[str, Any]) -> bool:
        if not q:
            return True
        hay = " ".join([
            rec.get("filename") or "",
            rec.get("path") or "",
            " ".join((rec.get("tags") or [])) if isinstance(rec.get("tags"), list) else "",
            (rec.get("ai", {}).get("ai_fast") or {}).get("summary") or ""
        ]).lower()
        return q.lower() in hay

    show = [rec for rec in items if visible(rec)]
    st.caption(f"Showing {len(show)} / {len(items)}")
    st.dataframe(
        [
            {
                "filename": file_display_name(rec),
                "status": rec.get("extraction_status"),
                "engine": rec.get("engine") or (rec.get("meta") or {}).get("engine"),
                "ext": rec.get("ext"),
                "lang": rec.get("language"),
                "domain": rec.get("domain") or (rec.get("ai", {}).get("ai_fast") or {}).get("domain"),
                "text_len": len(get_text_from_item(rec)),
            }
            for rec in show
        ],
        use_container_width=True,
        hide_index=True
    )

# -------- File Inspector ----------
with tab_file:
    st.subheader("File Inspector")
    names = [file_display_name(r) for r in items]
    idx = st.selectbox("Select file", list(range(len(items))), format_func=lambda i: names[i])
    rec = items[int(idx)]
    cols = st.columns(3)
    with cols[0]:
        st.write("**Path**"); st.code(rec.get("path",""))
    with cols[1]:
        st.write("**Status**"); st.code(rec.get("extraction_status",""))
    with cols[2]:
        st.write("**Engine**"); st.code(rec.get("engine") or (rec.get("meta") or {}).get("engine") or "")
    st.markdown("**Summary (fast)**")
    st.write((rec.get("ai", {}).get("ai_fast") or {}).get("summary") or "—")
    st.markdown("**Detected Language**")
    st.code(rec.get("language") or ((rec.get("meta") or {}).get("language")) or "—")
    with st.expander("Raw meta"):
        st.json(rec.get("meta") or {})
    with st.expander("AI block"):
        st.json(rec.get("ai") or {})
    with st.expander("Text (first 2000 chars)"):
        txt = get_text_from_item(rec)
        st.code(txt[:2000] + ("..." if len(txt) > 2000 else ""))

# -------- Graph ----------
with tab_graph:
    st.subheader("Relationship Graph")
    edges_path = os.path.join(run_dir, "graph_edges.jsonl")
    exists = os.path.isfile(edges_path)

    if not exists:
        st.info("No graph_edges.jsonl found for this run.")
        if st.button("Generate a relationship graph (semantic similarity)"):
            with st.spinner("Building graph (embedding + kNN)…"):
                out_path, n_edges = asyncio.run(ensure_graph_edges(
                    run_dir,
                    k=3,
                    min_sim=0.32,
                    use_ollama=True,
                    ollama_url=settings.OLLAMA_URL,
                    ollama_embed_model=settings.OLLAMA_EMBED_MODEL,
                    use_cloud=settings.cloud_ready(),
                    cloud_provider=("openai" if settings.CLOUD_LLM_PROVIDER == "openai" else "none"),
                    openai_api_key=settings.OPENAI_API_KEY,
                    openai_embed_model=settings.OPENAI_EMBED_MODEL,
                ))
            st.success(f"Graph created: {out_path} ({n_edges} edges). Reopen this tab to view.")
    else:
        # Render a simple adjacency listing (keeps deps light).
        edges = []
        with open(edges_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    edges.append(json.loads(line))
                except Exception:
                    pass
        st.caption(f"{len(edges)} edges")
        # Group by src
        by_src: Dict[str, List[Tuple[str, float]]] = {}
        for e in edges:
            by_src.setdefault(e["src"], []).append((e["dst"], float(e.get("weight", 0.0))))
        for src, lst in list(by_src.items())[:50]:
            lst.sort(key=lambda t: t[1], reverse=True)
            st.markdown(f"**{pathlib.Path(src).name}**")
            st.write(", ".join(f"{pathlib.Path(d).name} ({w:.2f})" for d, w in lst[:8]))

# -------- Ask AI ----------
with tab_ask:
    st.subheader("Ask AI")
    st.caption("Local by default (Ollama). Enable cloud in Config.yaml to use OpenAI/Azure.")
    mode = st.radio("Mode", ["Catalog Q&A", "Content Q&A"], horizontal=True)
    q = st.text_area("Your question", placeholder="Ask about the catalog or file contents…", height=120)
    topk = st.slider("Top-K passages", min_value=3, max_value=12, value=int(settings.RAG_TOP_K))

    if st.button("Ask"):
        if not q.strip():
            st.warning("Enter a question.")
        else:
            with st.spinner("Thinking…"):
                # Build retrieval corpus
                docs: List[Tuple[str, str]] = []
                if mode == "Catalog Q&A":
                    for r in items:
                        name = file_display_name(r)
                        summary = (r.get("ai", {}).get("ai_fast") or {}).get("summary") or ""
                        meta = r.get("meta") or {}
                        row = f"{name}\nstatus={r.get('extraction_status')} domain={r.get('domain')}\nsummary={summary}\nengine={r.get('engine') or meta.get('engine')}"
                        docs.append((name, row))
                else:
                    # Content Q&A: chunk available text
                    CH = max(300, settings.RAG_CHUNK_CHARS)
                    OV = max(0, settings.RAG_OVERLAP)
                    for r in items:
                        name = file_display_name(r)
                        txt = get_text_from_item(r)
                        if not txt:
                            continue
                        for i in range(0, len(txt), CH - OV):
                            snippet = txt[i:i+CH]
                            if snippet.strip():
                                docs.append((name, snippet))

                # Embed & rank
                corpus_texts = [d[1] for d in docs]
                vecs = asyncio.run(embed_texts(
                    corpus_texts,
                    use_ollama=True,
                    ollama_url=settings.OLLAMA_URL,
                    ollama_model=settings.OLLAMA_EMBED_MODEL,
                    use_cloud=settings.cloud_ready(),
                    cloud_provider=("openai" if settings.CLOUD_LLM_PROVIDER == "openai" else "none"),
                    openai_api_key=settings.OPENAI_API_KEY,
                    openai_embed_model=settings.OPENAI_EMBED_MODEL,
                ))
                # Query vector (simple)
                qvec = asyncio.run(embed_texts(
                    [q],
                    use_ollama=True,
                    ollama_url=settings.OLLAMA_URL,
                    ollama_model=settings.OLLAMA_EMBED_MODEL,
                    use_cloud=settings.cloud_ready(),
                    cloud_provider=("openai" if settings.CLOUD_LLM_PROVIDER == "openai" else "none"),
                    openai_api_key=settings.OPENAI_API_KEY,
                    openai_embed_model=settings.OPENAI_EMBED_MODEL,
                ))[0]

                sims = [(i, float(np.dot(vecs[i] / (np.linalg.norm(vecs[i])+1e-9), qvec / (np.linalg.norm(qvec)+1e-9)))) for i in range(len(docs))]
                sims.sort(key=lambda t: t[1], reverse=True)
                picks = [docs[i] for i, _ in sims[:topk]]

                # Ask LLM
                sys_prompt = "You are an expert data assistant. Answer clearly and cite the document names in [brackets] where relevant."
                ans = asyncio.run(chat_answer(
                    q,
                    system=sys_prompt,
                    docs=picks,
                    use_ollama=True,
                    ollama_url=settings.OLLAMA_URL,
                    ollama_model=settings.OLLAMA_MODEL,
                    use_cloud=settings.cloud_ready(),
                    cloud_provider=("openai" if settings.CLOUD_LLM_PROVIDER == "openai" else "none"),
                    openai_api_key=settings.OPENAI_API_KEY,
                    openai_model=settings.OPENAI_MODEL,
                ))
                st.markdown("**Answer**")
                st.write(ans)
                with st.expander("Context used"):
                    st.code("\n\n".join([f"[{i+1}] {d[0]}\n{d[1][:400]}" for i, d in enumerate(picks)]))

# -------- Scan ----------
with tab_scan:
    st.subheader("Run a Scan")
    cmd = settings.SCAN_COMMAND or "python -m mcp_ai.main --config ./mcp_ai/config.yaml"
    st.code(cmd, language="bash")
    if st.button("Run"):
        with st.spinner("Launching scan…"):
            try:
                # Non-blocking; tail logs in your terminal
                subprocess.Popen(cmd, shell=True)
                st.success("Scan started. Watch your terminal logs. Refresh runs after it finishes.")
            except Exception as e:
                st.error(f"Failed to start: {e}")