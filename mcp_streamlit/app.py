# mcp_ai/mcp_streamlit/app.py
from __future__ import annotations

import asyncio
import io
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
from streamlit.components.v1 import html as st_html

# ──────────────────────────────────────────────────────────────────────────────
# Local imports (add directory to sys.path so absolute imports work)
# ──────────────────────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

# These modules must live alongside app.py: config.py, data.py, ai.py, graph.py
from config import settings  # type: ignore
from data import (          # type: ignore
    list_runs,
    load_items_jsonl,
    load_annotations,
    load_saved_views,
    save_view,
    upsert_annotation,
    file_display_name,
    get_text_from_item,
)
from ai import embed_texts, chat_answer  # type: ignore
try:
    from graph import ensure_graph_edges  # type: ignore
except Exception:
    ensure_graph_edges = None  # type: ignore

# Optional fuzzy search
try:
    from rapidfuzz import process as rf_process  # type: ignore
except Exception:
    rf_process = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page configuration + optional CSS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MCP Dash",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS (if present)
try:
    with open(HERE / "styles.css", "r", encoding="utf-8") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Data source: discover runs & selected run
# ──────────────────────────────────────────────────────────────────────────────

def _discover_runs(root: str) -> List[Tuple[str, str]]:
    """Return [(run_id, run_dir), ...], newest first."""
    try:
        runs = list_runs(root)
        # list_runs should already be sorted; otherwise enforce reverse chrono by dir name
        return runs
    except Exception:
        # Fallback simple discover
        rr = []
        runs_root = os.path.join(root, "runs")
        if os.path.isdir(runs_root):
            for d in sorted(os.listdir(runs_root), reverse=True):
                p = os.path.join(runs_root, d)
                if os.path.isdir(p) and os.path.isfile(os.path.join(p, "items.jsonl")):
                    rr.append((d, p))
        return rr

def _summarize_run(run_dir: str) -> Dict[str, Any]:
    """Fast one-pass summary of a run from items.jsonl (no pandas)."""
    items_path = os.path.join(run_dir, "items.jsonl")
    ok = mo = er = 0
    lat = []
    total = 0
    if not os.path.isfile(items_path):
        return {"files_total": 0, "ok": 0, "metadata_only": 0, "error": 0, "median_latency": None}
    with open(items_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            total += 1
            stt = rec.get("extraction_status") or ""
            if stt == "ok":
                ok += 1
            elif stt == "metadata_only":
                mo += 1
            elif stt == "error":
                er += 1
            el = (rec.get("meta") or {}).get("elapsed_s")
            if isinstance(el, (int, float)):
                lat.append(float(el))
    med = float(np.median(lat)) if lat else None
    return {"files_total": total, "ok": ok, "metadata_only": mo, "error": er, "median_latency": med}

def _load_run_items(run_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    items = load_items_jsonl(run_dir)
    annotations = load_annotations(run_dir)
    views = load_saved_views(run_dir)
    return items, annotations, views

# Discover runs at start
ALL_RUNS: List[Tuple[str, str]] = _discover_runs(settings.OUTPUT_ROOT)
if not ALL_RUNS:
    st.sidebar.error(
        "No runs found under OUTPUT_ROOT. "
        "Create runs with items.jsonl at: output_files/runs/<run_id>/items.jsonl"
    )
    st.stop()

# Selected run id: from state, or env settings.RUN_ID, or latest
if "selected_run_id" not in st.session_state:
    cand = settings.RUN_ID or ALL_RUNS[0][0]
    st.session_state["selected_run_id"] = cand

def _set_run(run_id: str) -> None:
    st.session_state["selected_run_id"] = run_id
    # Safe rerun (newer Streamlit has st.rerun; older has experimental_rerun)
    rerun = getattr(st, "rerun", getattr(st, "experimental_rerun", None))
    if callable(rerun):
        rerun()

# Sidebar run picker (always available)
st.sidebar.title("MCP Dash")
st.sidebar.caption("Client-facing portal for MCP runs")
run_ids = [r[0] for r in ALL_RUNS]
current_run_id = st.sidebar.selectbox(
    "Run", run_ids, index=run_ids.index(st.session_state["selected_run_id"])
)
if current_run_id != st.session_state["selected_run_id"]:
    _set_run(current_run_id)

RUN_DIR = dict(ALL_RUNS)[st.session_state["selected_run_id"]]
ITEMS, ANNOT, VIEWS = _load_run_items(RUN_DIR)
if not ITEMS:
    st.warning("This run has no items.jsonl entries.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Header + KPI cards
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <h1 style="display:flex;align-items:center;gap:.6rem;">
      <span>📁 {st.session_state['selected_run_id']}</span>
      <span class="badge">{len(ITEMS)} files</span>
    </h1>
    """,
    unsafe_allow_html=True,
)

ok = sum(1 for r in ITEMS if r.get("extraction_status") == "ok")
mo = sum(1 for r in ITEMS if r.get("extraction_status") == "metadata_only")
er = sum(1 for r in ITEMS if r.get("extraction_status") == "error")
latencies = []
for r in ITEMS:
    el = (r.get("meta") or {}).get("elapsed_s")
    if isinstance(el, (int, float)):
        latencies.append(float(el))
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
    st.markdown(f"<div class='metric-card'><h2>{er}</h2><div class='badge err'>error</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown("**Median latency (s)**")
    st.markdown(f"<div class='metric-card'><h2>{median_latency if median_latency is not None else '—'}</h2></div>", unsafe_allow_html=True)

st.markdown("<hr class='sep'/>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_runs, tab_overview, tab_catalog, tab_file, tab_graph, tab_ask, tab_logs = st.tabs(
    ["Runs", "Overview", "Catalog", "Inspector", "Graph", "Ask AI", "Logs / Scan"]
)

# ================================ Runs (picker) ================================
with tab_runs:
    st.subheader("All Runs")
    rows = []
    for rid, rdir in ALL_RUNS:
        s = _summarize_run(rdir)
        rows.append(
            {
                "run_id": rid,
                "files": s["files_total"],
                "ok": s["ok"],
                "metadata_only": s["metadata_only"],
                "error": s["error"],
                "median_latency_s": s["median_latency"],
                "dir": rdir,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    sel = st.selectbox("Switch to run", [r[0] for r in ALL_RUNS], index=[r[0] for r in ALL_RUNS].index(st.session_state["selected_run_id"]))
    if st.button("Use selected run"):
        _set_run(sel)

# ================================== Overview ===================================
with tab_overview:
    st.subheader("Overview")
    domains: Dict[str, int] = {}
    for r in ITEMS:
        ai_fast = (r.get("ai") or {}).get("ai_fast") or {}
        d = r.get("domain") or ai_fast.get("domain") or "general"
        domains[d] = domains.get(d, 0) + 1

    if domains:
        rows = sorted(domains.items(), key=lambda t: t[1], reverse=True)
        st.table({"Domain": [d for d, _ in rows], "Files": [c for _, c in rows]})
    else:
        st.info("No domain information available.")

# ================================== Catalog ====================================
with tab_catalog:
    st.subheader("Catalog")

    left, right = st.columns([3, 1])
    with left:
        q = st.text_input("Global search", placeholder="Filename, path, tags, summary, owner…")
    with right:
        picked_view = st.selectbox("Saved views", ["(none)"] + list(VIEWS.keys()))
        if picked_view != "(none)":
            vconf = VIEWS[picked_view]
            q = vconf.get("query", q)

    def visible(rec: Dict[str, Any]) -> bool:
        if not q:
            return True
        ann = ANNOT.get(rec.get("path", ""), {}) or {}
        hay = " ".join(
            [
                rec.get("filename") or "",
                rec.get("path") or "",
                " ".join((rec.get("tags") or [])) if isinstance(rec.get("tags"), list) else "",
                ((rec.get("ai") or {}).get("ai_fast") or {}).get("summary") or "",
                " ".join(ann.get("owners", [])),
                " ".join(ann.get("tags", [])),
            ]
        ).lower()
        if rf_process:
            score = rf_process.extractOne(q, [hay])[1]  # type: ignore[index]
            return score >= 60
        return q.lower() in hay

    show = [rec for rec in ITEMS if visible(rec)]
    st.caption(f"Showing {len(show)} / {len(ITEMS)}")

    export_rows: List[Dict[str, Any]] = []
    for rec in show:
        ann = ANNOT.get(rec.get("path", ""), {})
        owners = ",".join(ann.get("owners", []))
        tags = ",".join(ann.get("tags", []))
        export_rows.append(
            {
                "file": file_display_name(rec),
                "path": rec.get("path"),
                "status": rec.get("extraction_status"),
                "engine": rec.get("engine") or (rec.get("meta") or {}).get("engine"),
                "ext": rec.get("ext"),
                "lang": rec.get("language")
                or (rec.get("meta") or {}).get("language")
                or ((rec.get("ai") or {}).get("ai_fast") or {}).get("language"),
                "domain": rec.get("domain")
                or ((rec.get("ai") or {}).get("ai_fast") or {}).get("domain"),
                "owners": owners,
                "tags": tags,
                "text_len": len(get_text_from_item(rec)),
            }
        )

    st.dataframe(export_rows, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=_to_csv(export_rows),
        file_name=f"{st.session_state['selected_run_id']}_catalog.csv",
        mime="text/csv",
    )

    with st.expander("Save current view"):
        name = st.text_input("Name")
        if st.button("Save view"):
            if not name.strip():
                st.warning("Provide a name.")
            else:
                save_view(RUN_DIR, name.strip(), {"query": q})
                st.success("Saved. Reopen the dropdown to see it.")

# ================================= Inspector ===================================
with tab_file:
    st.subheader("Inspector")
    names = [file_display_name(r) for r in ITEMS]
    idx = st.selectbox("Select file", list(range(len(ITEMS))), format_func=lambda i: names[i])
    rec = ITEMS[int(idx)]
    path = rec.get("path", "")

    ann = ANNOT.get(path, {}) or {}
    owners = ann.get("owners", [])
    tags = ann.get("tags", [])
    notes = ann.get("notes", "")

    cols = st.columns(3)
    with cols[0]:
        st.write("**Path**")
        st.code(path)
    with cols[1]:
        st.write("**Status**")
        st.code(rec.get("extraction_status", ""))
    with cols[2]:
        st.write("**Engine**")
        st.code(rec.get("engine") or (rec.get("meta") or {}).get("engine") or "")

    st.markdown("**Summary (fast)**")
    st.write(((rec.get("ai") or {}).get("ai_fast") or {}).get("summary") or "—")

    st.markdown("**Detected Language**")
    st.code(
        rec.get("language")
        or (rec.get("meta") or {}).get("language")
        or ((rec.get("ai") or {}).get("ai_fast") or {}).get("language")
        or "—"
    )

    with st.expander("Owners & tags (client-facing)"):
        owners_str = st.text_input("Owners (comma-separated)", value=",".join(owners))
        tags_str = st.text_input("Tags (comma-separated)", value=",".join(tags))
        notes_new = st.text_area("Notes", value=notes, height=100)
        if st.button("Save annotations"):
            upsert_annotation(
                RUN_DIR,
                path,
                [o.strip() for o in owners_str.split(",") if o.strip()],
                [t.strip() for t in tags_str.split(",") if t.strip()],
                notes_new.strip(),
            )
            st.success("Saved (annotations.jsonl).")

    with st.expander("Raw meta"):
        st.json(rec.get("meta") or {})
    with st.expander("AI block"):
        st.json(rec.get("ai") or {})
    with st.expander("Text (first 2000 chars)"):
        txt = get_text_from_item(rec)
        st.code(txt[:2000] + ("..." if len(txt) > 2000 else ""))

# =================================== Graph ====================================
with tab_graph:
    st.subheader("Relationship Graph")
    edges_path = os.path.join(RUN_DIR, "graph_edges.jsonl")
    if not os.path.isfile(edges_path):
        st.info("No graph_edges.jsonl found for this run.")
        gen = st.button("Generate relationship graph")
        if gen:
            with st.spinner("Building graph…"):
                if ensure_graph_edges is None:
                    st.error("graph.ensure_graph_edges is missing. Make sure graph.py exists next to app.py.")
                else:
                    # call advanced signature if available; else fallback
                    try:
                        result = ensure_graph_edges(  # type: ignore[call-arg]
                            run_dir=RUN_DIR,
                            k=3,
                            min_sim=0.32,
                            use_ollama=True,
                            ollama_url=getattr(settings, "OLLAMA_URL", "http://127.0.0.1:11434"),
                            ollama_embed_model=getattr(settings, "OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                            use_cloud=getattr(settings, "cloud_ready", lambda: False)(),
                            cloud_provider=("openai" if getattr(settings, "CLOUD_LLM_PROVIDER", "none") == "openai" else "none"),
                            openai_api_key=getattr(settings, "OPENAI_API_KEY", None),
                            openai_embed_model=getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                        )
                    except TypeError:
                        result = ensure_graph_edges(RUN_DIR)  # type: ignore[misc,call-arg]

                    if asyncio.iscoroutine(result):
                        out_path, n_edges = asyncio.run(result)  # type: ignore[misc]
                    else:
                        if isinstance(result, tuple):
                            out_path, n_edges = result
                        else:
                            out_path, n_edges = result, None

                    msg = f"Graph created: {out_path}"
                    if n_edges is not None:
                        msg += f" ({n_edges} edges)"
                    st.success(msg)
                    edges_path = out_path

    if os.path.isfile(edges_path):
        st.caption("Interactive view (drag, zoom, select)")
        html_str = _render_pyvis(edges_path)
        if html_str:
            st_html(html_str, height=600, scrolling=True)
        else:
            st.info("Install `pyvis` to render the graph: `pip install pyvis`")

# ==================================== Ask AI ===================================
with tab_ask:
    st.subheader("Ask AI")
    st.caption("Local by default (Ollama). Enable cloud in Config.yaml to use OpenAI/Azure.")

    mode = st.radio("Mode", ["Catalog Q&A", "Content Q&A"], horizontal=True)
    q = st.text_area("Your question", placeholder="Ask about the catalog or contents…", height=120)
    topk_default = int(getattr(settings, "RAG_TOP_K", 6))
    topk = st.slider("Top-K passages", min_value=3, max_value=12, value=topk_default)

    if st.button("Ask"):
        if not q.strip():
            st.warning("Enter a question.")
        else:
            with st.spinner("Thinking…"):
                # Build corpus
                docs: List[Tuple[str, str]] = []
                if mode == "Catalog Q&A":
                    for r in ITEMS:
                        name = file_display_name(r)
                        summary = ((r.get("ai") or {}).get("ai_fast") or {}).get("summary") or ""
                        ann = ANNOT.get(r.get("path", ""), {})
                        owners = ",".join(ann.get("owners", []))
                        tgs = ",".join(ann.get("tags", []))
                        row = (
                            f"{name}\n"
                            f"status={r.get('extraction_status')} domain={r.get('domain')}\n"
                            f"owners={owners} tags={tgs}\n"
                            f"summary={summary}\n"
                            f"engine={r.get('engine') or (r.get('meta') or {}).get('engine')}"
                        )
                        docs.append((name, row))
                else:
                    CH = max(300, getattr(settings, "RAG_CHUNK_CHARS", 1200))
                    OV = max(0, getattr(settings, "RAG_OVERLAP", 120))
                    step = max(1, CH - OV)
                    for r in ITEMS:
                        name = file_display_name(r)
                        txt = get_text_from_item(r)
                        if not txt:
                            continue
                        for i in range(0, len(txt), step):
                            snippet = txt[i : i + CH]
                            if snippet.strip():
                                docs.append((name, snippet))

                # Embed corpus + query
                corpus_texts = [d[1] for d in docs]
                vecs = asyncio.run(
                    embed_texts(
                        corpus_texts,
                        use_ollama=True,
                        ollama_url=getattr(settings, "OLLAMA_URL", "http://127.0.0.1:11434"),
                        ollama_model=getattr(settings, "OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                        use_cloud=getattr(settings, "cloud_ready", lambda: False)(),
                        cloud_provider=("openai" if getattr(settings, "CLOUD_LLM_PROVIDER", "none") == "openai" else "none"),
                        openai_api_key=getattr(settings, "OPENAI_API_KEY", None),
                        openai_embed_model=getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                    )
                )
                qvec = asyncio.run(
                    embed_texts(
                        [q],
                        use_ollama=True,
                        ollama_url=getattr(settings, "OLLAMA_URL", "http://127.0.0.1:11434"),
                        ollama_model=getattr(settings, "OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                        use_cloud=getattr(settings, "cloud_ready", lambda: False)(),
                        cloud_provider=("openai" if getattr(settings, "CLOUD_LLM_PROVIDER", "none") == "openai" else "none"),
                        openai_api_key=getattr(settings, "OPENAI_API_KEY", None),
                        openai_embed_model=getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                    )
                )[0]

                # Cosine similarity (manual; avoids extra deps)
                def _norm(v: np.ndarray) -> float:
                    n = float(np.linalg.norm(v))
                    return n if n > 1e-9 else 1e-9

                sims = [
                    (i, float(np.dot(vecs[i] / _norm(vecs[i]), qvec / _norm(qvec))))
                    for i in range(len(docs))
                ]
                sims.sort(key=lambda t: t[1], reverse=True)
                picks = [docs[i] for i, _ in sims[:max(1, topk)]]

                sys_prompt = "You are an expert data assistant. Answer clearly and cite [document names] inline."
                ans = asyncio.run(
                    chat_answer(
                        q,
                        system=sys_prompt,
                        docs=picks,
                        use_ollama=True,
                        ollama_url=getattr(settings, "OLLAMA_URL", "http://127.0.0.1:11434"),
                        ollama_model=getattr(settings, "OLLAMA_MODEL", "llama3.1"),
                        use_cloud=getattr(settings, "cloud_ready", lambda: False)(),
                        cloud_provider=("openai" if getattr(settings, "CLOUD_LLM_PROVIDER", "none") == "openai" else "none"),
                        openai_api_key=getattr(settings, "OPENAI_API_KEY", None),
                        openai_model=getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
                    )
                )
                st.markdown("**Answer**")
                st.write(ans)

                with st.expander("Citations / context used"):
                    st.code("\n\n".join([f"[{i+1}] {d[0]}\n{d[1][:400]}" for i, d in enumerate(picks)]))

# ================================= Logs / Scan =================================
with tab_logs:
    st.subheader("Logs & Scan")
    log_path = pathlib.Path(getattr(settings, "OUTPUT_ROOT", "./mcp_ai/output_files")) / "mcp_ai.log"

    auto = st.checkbox("Auto-refresh (5s)")
    if auto and _tick_every(5):
        rerun = getattr(st, "rerun", getattr(st, "experimental_rerun", None))
        if callable(rerun):
            rerun()

    if log_path.exists():
        text = "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-400:])
        st.text_area("mcp_ai.log (tail)", value=text, height=300)
    else:
        st.info("No log file found yet.")

    st.markdown("**Run a Scan**")
    cmd = getattr(settings, "SCAN_COMMAND", "python -m mcp_ai.main --config ./mcp_ai/config.yaml")
    st.code(cmd, language="bash")
    if st.button("Start Scan"):
        try:
            subprocess.Popen(cmd, shell=True)
            st.success("Scan started. Watch logs above.")
        except Exception as e:
            st.error(f"Failed to start: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_csv(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    import csv
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()

def _render_pyvis(edges_path: str) -> str | None:
    try:
        from pyvis.network import Network  # type: ignore
    except Exception:
        return None
    net = Network(height="600px", width="100%", bgcolor="#0b1220", font_color="#e2e8f0", directed=False)
    edges = []
    nodes = set()
    with open(edges_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                e = json.loads(s)
                edges.append(e)
                nodes.add(e["src"])
                nodes.add(e["dst"])
            except Exception:
                pass
    for n in nodes:
        label = pathlib.Path(n).name or n
        net.add_node(n, label=label, title=n, shape="dot", size=12, color="#1f6feb")
    for e in edges:
        weight = float(e.get("weight", 0.3))
        net.add_edge(e["src"], e["dst"], value=max(1, int(weight * 10)), color="#94a3b8")
    net.toggle_physics(True)
    return net.generate_html(notebook=False)

def _tick_every(secs: int) -> bool:
    """Return True roughly once per `secs` seconds per user session."""
    key = "_last_tick"
    now = time.time()
    if key not in st.session_state:
        st.session_state[key] = now
        return True
    if now - float(st.session_state[key]) >= secs:
        st.session_state[key] = now
        return True
    return False