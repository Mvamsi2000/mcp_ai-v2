# mcp_ai/mcp_streamlit/app.py
from __future__ import annotations

import os
import sys
import json
import time
import pathlib
import subprocess
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
from streamlit.components.v1 import html as st_html

# ──────────────────────────────────────────────────────────────────────────────
# Path + Config (absolute imports so Streamlit can run as a script)
# ──────────────────────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

try:
    from config import settings  # type: ignore
except Exception:
    # ultra-safe defaults if config import fails
    class _S:
        OUTPUT_ROOT = "./mcp_ai/output_files"
        RUN_ID = None
        OLLAMA_URL = "http://127.0.0.1:11434"
        OLLAMA_MODEL = "llama3.1"
        OLLAMA_EMBED_MODEL = "nomic-embed-text"
        OPENAI_API_KEY = None
        OPENAI_MODEL = "gpt-4o-mini"
        OPENAI_EMBED_MODEL = "text-embedding-3-small"
        RAG_CHUNK_CHARS = 1200
        RAG_OVERLAP = 120
        RAG_TOP_K = 6
        SCAN_COMMAND = "python -m mcp_ai.main --config ./mcp_ai/config.yaml"

        def cloud_ready(self) -> bool:
            return False

        CLOUD_LLM_PROVIDER = "none"

    settings = _S()  # type: ignore

# Optional fuzzy search (nice-to-have)
try:
    from rapidfuzz import process as rf_process  # type: ignore
except Exception:
    rf_process = None  # type: ignore

# Optional graph utilities
try:
    from graph import ensure_graph_edges, build_graph_html  # type: ignore
except Exception:
    ensure_graph_edges = None  # type: ignore
    build_graph_html = None  # type: ignore

# Optional AI utilities
try:
    from ai import embed_texts, chat_answer  # type: ignore
except Exception:
    embed_texts = None  # type: ignore
    chat_answer = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (placed BEFORE usage to avoid NameError on reruns)
# ──────────────────────────────────────────────────────────────────────────────

def _to_csv(rows: List[Dict[str, Any]]) -> str:
    """Convert list[dict] to CSV text."""
    if not rows:
        return ""
    import io, csv
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()

def _render_pyvis(edges_path: str) -> str:
    """Render a pyvis HTML from graph_edges.jsonl."""
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

def _tick_every(secs: int) -> bool:
    """Return True roughly once per `secs` seconds per user session."""
    key = "_last_tick"
    now = time.time()
    if key not in st.session_state:
        st.session_state[key] = now
        return True
    if now - st.session_state[key] >= secs:
        st.session_state[key] = now
        return True
    return False

def list_runs(output_root: str) -> List[Tuple[str, str]]:
    """Return [(run_id, run_dir)] by scanning OUTPUT_ROOT/runs/*."""
    runs_root = pathlib.Path(output_root) / "runs"
    out: List[Tuple[str, str]] = []
    if not runs_root.exists():
        return out
    for d in sorted(runs_root.iterdir(), key=lambda p: p.name, reverse=True):
        if d.is_dir():
            out.append((d.name, str(d)))
    return out

def load_items_jsonl(run_dir: str) -> List[Dict[str, Any]]:
    p = pathlib.Path(run_dir) / "items.jsonl"
    rows: List[Dict[str, Any]] = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def file_display_name(rec: Dict[str, Any]) -> str:
    return rec.get("filename") or pathlib.Path(rec.get("path", "")).name or rec.get("path", "unknown")

def get_text_from_item(rec: Dict[str, Any]) -> str:
    ai = rec.get("ai") or {}
    raw = ai.get("_raw_text")
    if isinstance(raw, str) and raw.strip():
        return raw
    if isinstance(rec.get("text"), str) and rec["text"].strip():
        return rec["text"]
    return ""

def load_annotations(run_dir: str) -> Dict[str, Dict[str, Any]]:
    """annotations.jsonl is a map by path; if present, last write wins."""
    p = pathlib.Path(run_dir) / "annotations.jsonl"
    m: Dict[str, Dict[str, Any]] = {}
    if not p.exists():
        return m
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                k = row.get("path")
                if not k:
                    continue
                m[k] = row
            except Exception:
                pass
    return m

def upsert_annotation(run_dir: str, path: str, owners: List[str], tags: List[str], notes: str) -> None:
    """Append an annotation row; simple last-write-wins semantics."""
    p = pathlib.Path(run_dir) / "annotations.jsonl"
    row = {"path": path, "owners": owners, "tags": tags, "notes": notes, "ts": int(time.time())}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_saved_views(run_dir: str) -> Dict[str, Dict[str, Any]]:
    p = pathlib.Path(run_dir) / "saved_views.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_view(run_dir: str, name: str, data: Dict[str, Any]) -> None:
    p = pathlib.Path(run_dir) / "saved_views.json"
    cur = load_saved_views(run_dir)
    cur[name] = data
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# UI config + theme
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MCP Dash",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS (safe if missing)
try:
    with open(HERE / "styles.css", "r", encoding="utf-8") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — pick run
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("MCP Dash")
st.sidebar.caption("client portal")

runs = list_runs(settings.OUTPUT_ROOT)
if not runs:
    st.sidebar.error("No runs found under OUTPUT_ROOT. Create a run at mcp_ai/output_files/runs/<id>/items.jsonl.")
    st.stop()

run_ids = [r[0] for r in runs]
default_index = 0
if getattr(settings, "RUN_ID", None) and settings.RUN_ID in run_ids:
    default_index = run_ids.index(settings.RUN_ID)

if "run_id" not in st.session_state:
    st.session_state["run_id"] = run_ids[default_index]

picked = st.sidebar.selectbox("Run", run_ids, index=run_ids.index(st.session_state["run_id"]))
if picked != st.session_state["run_id"]:
    st.session_state["run_id"] = picked
    st.rerun()

run_id = st.session_state["run_id"]
run_dir = dict(runs)[run_id]

# Load dataset
items = load_items_jsonl(run_dir)
if not items:
    st.warning("This run has no items.jsonl or it is empty.")
    st.stop()

annotations = load_annotations(run_dir)
views = load_saved_views(run_dir)

# ──────────────────────────────────────────────────────────────────────────────
# Header + KPIs
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <h1 style="display:flex;align-items:center;gap:.6rem;">
      <span>📁 {run_id}</span>
      <span class="badge">{len(items)} files</span>
    </h1>
    """,
    unsafe_allow_html=True,
)

ok = sum(1 for r in items if r.get("extraction_status") == "ok")
mo = sum(1 for r in items if r.get("extraction_status") == "metadata_only")
err = sum(1 for r in items if r.get("extraction_status") == "error")
latencies = [r.get("meta", {}).get("elapsed_s") for r in items if isinstance(r.get("meta", {}).get("elapsed_s"), (int, float))]
median_latency = float(np.median(latencies)) if latencies else None

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**Success**")
    st.markdown(f"<div class='metric-card'><h2>{ok}</h2><div class='badge ok'>ok</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("**Metadata-only**")
    st.markdown(f"<div class='metric-card'><h2>{mo}</h2><div class='badge mo'>metadata_only</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown("**Errors**")
    st.markdown(f"<div class='metric-card'><h2>{err}</h2><div class='badge err'>error</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown("**Median latency (s)**")
    st.markdown(f"<div class='metric-card'><h2>{median_latency if median_latency is not None else '—'}</h2></div>", unsafe_allow_html=True)

st.markdown("<hr class='sep'/>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_runs, tab_catalog, tab_file, tab_graph, tab_ask, tab_logs = st.tabs(
    ["Runs", "Catalog", "Inspector", "Graph", "Ask AI", "Logs / Scan"]
)

# -------- Runs tab --------
with tab_runs:
    st.subheader("All runs under OUTPUT_ROOT")
    run_rows = [{"run_id": rid, "path": rdir} for rid, rdir in runs]
    st.dataframe(run_rows, use_container_width=True, hide_index=True, height=300)
    new_pick = st.selectbox("Switch run", run_ids, index=run_ids.index(run_id))
    if st.button("Activate selected run"):
        if new_pick != run_id:
            st.session_state["run_id"] = new_pick
            st.success(f"Switched to run {new_pick}")
            st.rerun()
    st.caption(f"Root: {settings.OUTPUT_ROOT}")

# -------- Catalog tab --------
with tab_catalog:
    st.subheader("Catalog")
    left, right = st.columns([3, 1])
    with left:
        q = st.text_input("Global search", placeholder="Filename, path, tags, summary, owner…")
    with right:
        picked_view = st.selectbox("Saved views", ["(none)"] + list(views.keys()))
        if picked_view != "(none)":
            vconf = views[picked_view]
            q = vconf.get("query", q)

    def visible(rec: Dict[str, Any]) -> bool:
        if not q:
            return True
        hay = " ".join([
            rec.get("filename") or "",
            rec.get("path") or "",
            " ".join((rec.get("tags") or [])) if isinstance(rec.get("tags"), list) else "",
            (rec.get("ai", {}).get("ai_fast") or {}).get("summary") or "",
            " ".join(annotations.get(rec.get("path",""),{}).get("owners",[])),
            " ".join(annotations.get(rec.get("path",""),{}).get("tags",[])),
        ]).lower()
        if rf_process:
            score = rf_process.extractOne(q, [hay])[1]  # type: ignore
            return score >= 60
        return q.lower() in hay

    filtered = [rec for rec in items if visible(rec)]

    # Build table rows with main fields
    export_rows: List[Dict[str, Any]] = []
    for rec in filtered:
        ann = annotations.get(rec.get("path", ""), {})
        ai_fast = (rec.get("ai", {}) or {}).get("ai_fast", {}) or {}
        export_rows.append({
            "file": file_display_name(rec),
            "path": rec.get("path", ""),
            "status": rec.get("extraction_status", ""),
            "engine": rec.get("engine") or (rec.get("meta") or {}).get("engine"),
            "ext": rec.get("ext"),
            "lang": rec.get("language") or (rec.get("meta") or {}).get("language"),
            "domain": rec.get("domain") or ai_fast.get("domain"),
            "owners": ",".join(ann.get("owners", [])),
            "tags": ",".join(ann.get("tags", [])),
            "text_len": len(get_text_from_item(rec)),
            "confidence": ai_fast.get("confidence"),
            "contains_pii": ai_fast.get("contains_pii"),
            "summary_fast": (ai_fast.get("summary") or "")[:160],
        })

    st.caption(f"Showing {len(filtered)} / {len(items)}")
    st.dataframe(export_rows, use_container_width=True, hide_index=True, height=420)

    # Selection + detail panel
    names = [file_display_name(r) for r in filtered]
    if names:
        idx = st.selectbox("Select a file to inspect", list(range(len(filtered))), format_func=lambda i: names[i])
        sel = filtered[int(idx)]
        path_sel = sel.get("path", "")
        ann = annotations.get(path_sel, {})
        ai = sel.get("ai") or {}
        ai_fast = ai.get("ai_fast") or {}
        ai_deep_local = ai.get("ai_deep_local") or {}
        pii = ai_deep_local.get("pii")
        glossary = ai_deep_local.get("glossary_terms")

        st.markdown("### File details")
        cA, cB, cC = st.columns([2, 1, 1])
        with cA:
            st.write("**Path**"); st.code(path_sel)
            st.write("**Engine**"); st.code(sel.get("engine") or (sel.get("meta") or {}).get("engine") or "—")
            st.write("**Owners**"); st.code(", ".join(ann.get("owners", [])) or "—")
            st.write("**Tags**"); st.code(", ".join(ann.get("tags", [])) or "—")
        with cB:
            st.write("**Status**"); st.code(sel.get("extraction_status", ""))
            st.write("**Language**"); st.code(sel.get("language") or (sel.get("meta") or {}).get("language") or "—")
        with cC:
            st.write("**Domain**"); st.code(sel.get("domain") or ai_fast.get("domain") or "—")
            st.write("**Confidence**"); st.code(ai_fast.get("confidence", "—"))

        st.markdown("**AI Fast — Summary**")
        st.write(ai_fast.get("summary") or "—")
        st.markdown("**AI Fast — Contains PII?**")
        st.code(str(ai_fast.get("contains_pii")) if "contains_pii" in ai_fast else "—")

        st.markdown("**AI Deep — Summary**")
        st.write((ai_deep_local.get("summary") or "—"))
        with st.expander("AI Deep — PII"):
            st.json(pii or {})
        with st.expander("AI Deep — Glossary Terms"):
            st.json(glossary or {})

        with st.expander("Raw meta"): st.json(sel.get("meta") or {})
        with st.expander("AI block"):  st.json(ai)

    # Save current view
    with st.expander("Save current view"):
        name = st.text_input("Name")
        if st.button("Save view"):
            if not name.strip():
                st.warning("Provide a name.")
            else:
                save_view(run_dir, name.strip(), {"query": q})
                st.success("Saved. Reopen the dropdown to see it.")

    # Export CSV
    st.download_button(
        "Download CSV",
        data=_to_csv(export_rows),
        file_name=f"{run_id}_catalog.csv",
        mime="text/csv",
    )

# -------- Inspector tab --------
with tab_file:
    st.subheader("Inspector")
    names_all = [file_display_name(r) for r in items]
    idx_all = st.selectbox("Select file", list(range(len(items))), format_func=lambda i: names_all[i])
    rec = items[int(idx_all)]
    path = rec.get("path", "")
    ann = annotations.get(path, {})
    owners = ann.get("owners", [])
    tags = ann.get("tags", [])
    notes = ann.get("notes", "")

    cols = st.columns(3)
    with cols[0]: st.write("**Path**"); st.code(path)
    with cols[1]: st.write("**Status**"); st.code(rec.get("extraction_status", ""))
    with cols[2]: st.write("**Engine**"); st.code(rec.get("engine") or (rec.get("meta") or {}).get("engine") or "")
    st.markdown("**Summary (fast)**")
    st.write((rec.get("ai", {}).get("ai_fast") or {}).get("summary") or "—")
    st.markdown("**Detected Language**")
    st.code(rec.get("language") or ((rec.get("meta") or {}).get("language")) or "—")

    with st.expander("Owners & tags (client-facing)"):
        owners_str = st.text_input("Owners (comma-separated)", value=",".join(owners))
        tags_str   = st.text_input("Tags (comma-separated)", value=",".join(tags))
        notes_new  = st.text_area("Notes", value=notes, height=100)
        if st.button("Save annotations"):
            upsert_annotation(
                run_dir,
                path,
                [o.strip() for o in owners_str.split(",") if o.strip()],
                [t.strip() for t in tags_str.split(",") if t.strip()],
                notes_new.strip(),
            )
            st.success("Saved (annotations.jsonl).")

    with st.expander("Raw meta"): st.json(rec.get("meta") or {})
    with st.expander("AI block"):  st.json(rec.get("ai") or {})
    with st.expander("Text (first 2000 chars)"):
        txt = get_text_from_item(rec)
        st.code(txt[:2000] + ("..." if len(txt) > 2000 else ""))

# -------- Graph tab --------
with tab_graph:
    st.subheader("Relationship Graph")
    edges_path = os.path.join(run_dir, "graph_edges.jsonl")
    if not os.path.isfile(edges_path):
        st.info("No graph_edges.jsonl found for this run.")
        if st.button("Generate relationship graph"):
            if ensure_graph_edges is None:
                st.error("Graph generator not available (graph.py not importable).")
            else:
                with st.spinner("Building graph (embedding + kNN)…"):
                    out_path, n_edges = ensure_graph_edges(
                        run_dir,
                        k=3,
                        min_sim=0.32,
                        use_ollama=True,
                        ollama_url=settings.OLLAMA_URL,
                        ollama_embed_model=settings.OLLAMA_EMBED_MODEL,
                        use_cloud=(settings.cloud_ready() if hasattr(settings, "cloud_ready") else False),
                        cloud_provider=("openai" if getattr(settings, "CLOUD_LLM_PROVIDER", "none") == "openai" else "none"),
                        openai_api_key=getattr(settings, "OPENAI_API_KEY", None),
                        openai_embed_model=getattr(settings, "OPENAI_EMBED_MODEL", None),
                    )
                st.success(f"Graph created: {out_path} ({n_edges} edges).")
                st.experimental_rerun()
    else:
        st.caption("Interactive view (drag, zoom, select)")
        # Prefer build_graph_html if provided, else our local renderer
        if build_graph_html:
            html_str = build_graph_html(edges_path)
        else:
            html_str = _render_pyvis(edges_path)
        st_html(html_str, height=600, scrolling=True)

# -------- Ask AI tab --------
with tab_ask:
    st.subheader("Ask AI")
    st.caption("Local by default (Ollama). Enable cloud in Config.yaml to use OpenAI/Azure.")
    if embed_texts is None or chat_answer is None:
        st.warning("AI backends not available. Ensure ai.py exports embed_texts and chat_answer.")
    else:
        mode = st.radio("Mode", ["Catalog Q&A", "Content Q&A"], horizontal=True)
        q = st.text_area("Your question", placeholder="Ask about the catalog or contents…", height=120)
        topk = st.slider("Top-K passages", min_value=3, max_value=12, value=int(getattr(settings, "RAG_TOP_K", 6)))

        if st.button("Ask"):
            if not q.strip():
                st.warning("Enter a question.")
            else:
                with st.spinner("Thinking…"):
                    docs: List[Tuple[str, str]] = []
                    if mode == "Catalog Q&A":
                        for r in items:
                            name = file_display_name(r)
                            summary = (r.get("ai", {}).get("ai_fast") or {}).get("summary") or ""
                            ann = annotations.get(r.get("path",""), {})
                            owners = ",".join(ann.get("owners",[]))
                            tags   = ",".join(ann.get("tags",[]))
                            row = f"{name}\nstatus={r.get('extraction_status')} domain={r.get('domain')}\nowners={owners} tags={tags}\nsummary={summary}\nengine={r.get('engine') or (r.get('meta') or {}).get('engine')}"
                            docs.append((name, row))
                    else:
                        CH = max(300, int(getattr(settings, "RAG_CHUNK_CHARS", 1200)))
                        OV = max(0, int(getattr(settings, "RAG_OVERLAP", 120)))
                        for r in items:
                            name = file_display_name(r)
                            txt = get_text_from_item(r)
                            if not txt:
                                continue
                            for i in range(0, len(txt), CH - OV if CH - OV > 0 else CH):
                                snippet = txt[i:i+CH]
                                if snippet.strip():
                                    docs.append((name, snippet))

                    corpus_texts = [d[1] for d in docs]
                    vecs = embed_texts(
                        corpus_texts,
                        use_ollama=True,
                        ollama_url=settings.OLLAMA_URL,
                        ollama_model=settings.OLLAMA_EMBED_MODEL,
                        use_cloud=(settings.cloud_ready() if hasattr(settings, "cloud_ready") else False),
                        cloud_provider=("openai" if getattr(settings, "CLOUD_LLM_PROVIDER", "none") == "openai" else "none"),
                        openai_api_key=getattr(settings, "OPENAI_API_KEY", None),
                        openai_embed_model=getattr(settings, "OPENAI_EMBED_MODEL", None),
                    )
                    qvec = embed_texts(
                        [q],
                        use_ollama=True,
                        ollama_url=settings.OLLAMA_URL,
                        ollama_model=settings.OLLAMA_EMBED_MODEL,
                        use_cloud=(settings.cloud_ready() if hasattr(settings, "cloud_ready") else False),
                        cloud_provider=("openai" if getattr(settings, "CLOUD_LLM_PROVIDER", "none") == "openai" else "none"),
                        openai_api_key=getattr(settings, "OPENAI_API_KEY", None),
                        openai_embed_model=getattr(settings, "OPENAI_EMBED_MODEL", None),
                    )[0]

                    V = np.array(vecs, dtype=float)
                    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
                    qv = qvec / (np.linalg.norm(qvec) + 1e-9)
                    sims = (V @ qv).tolist()
                    idxs = list(range(len(docs)))
                    idxs.sort(key=lambda i: sims[i], reverse=True)
                    picks = [docs[i] for i in idxs[:topk]]

                    sys_prompt = "You are an expert data assistant. Answer clearly and cite [document names] inline."
                    ans = chat_answer(
                        q,
                        system=sys_prompt,
                        docs=picks,
                        use_ollama=True,
                        ollama_url=settings.OLLAMA_URL,
                        ollama_model=settings.OLLAMA_MODEL,
                        use_cloud=(settings.cloud_ready() if hasattr(settings, "cloud_ready") else False),
                        cloud_provider=("openai" if getattr(settings, "CLOUD_LLM_PROVIDER", "none") == "openai" else "none"),
                        openai_api_key=getattr(settings, "OPENAI_API_KEY", None),
                        openai_model=getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
                    )
                    st.markdown("**Answer**")
                    st.write(ans)
                    with st.expander("Citations / context used"):
                        st.code("\n\n".join([f"[{i+1}] {d[0]}\n{d[1][:400]}" for i, d in enumerate(picks)]))

# -------- Logs / Scan tab --------
with tab_logs:
    st.subheader("Logs & Scan")
    log_path = pathlib.Path(getattr(settings, "OUTPUT_ROOT", "./mcp_ai/output_files")) / "mcp_ai.log"

    cols = st.columns([1,1,2])
    with cols[0]:
        if st.button("Refresh"):
            st.experimental_rerun()
    with cols[1]:
        auto = st.checkbox("Auto-refresh (5s)")
        if auto and _tick_every(5):
            st.experimental_rerun()

    if log_path.exists():
        text = "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-400:])
        st.text_area("mcp_ai.log (tail)", value=text, height=300)
    else:
        st.info("No log file found yet.")

    st.markdown("**Run a Scan**")
    cmd = getattr(settings, "SCAN_COMMAND", None) or "python -m mcp_ai.main --config ./mcp_ai/config.yaml"
    st.code(cmd, language="bash")
    if st.button("Start Scan"):
        try:
            subprocess.Popen(cmd, shell=True)
            st.success("Scan started. Watch logs above.")
        except Exception as e:
            st.error(f"Failed to start: {e}")