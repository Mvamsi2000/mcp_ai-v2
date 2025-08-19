# mcp_ai/dashboard/app.py
from __future__ import annotations
import argparse, os, sys
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd

# ---- robust local imports (package or script) ----
try:
    # package-style
    from .data_access import (
        find_runs, load_run_df, filter_df, load_config,
        stage_manifest_path, write_manifest,
    )
    from .charts import (
        chart_confidence_hist, chart_domain_bar, chart_pii_bar, chart_volume_over_time,
        chart_filetype_bar, chart_top_owners,
    )
    from .export_utils import render_export_buttons
    from .config_ui import render_config_inspector
    from .ask_llm import answer_question
except Exception:
    # script-style
    _PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    from mcp_ai.dashboard.data_access import (
        find_runs, load_run_df, filter_df, load_config,
        stage_manifest_path, write_manifest,
    )
    from mcp_ai.dashboard.charts import (
        chart_confidence_hist, chart_domain_bar, chart_pii_bar, chart_volume_over_time,
        chart_filetype_bar, chart_top_owners,
    )
    from mcp_ai.dashboard.export_utils import render_export_buttons
    from mcp_ai.dashboard.config_ui import render_config_inspector
    from mcp_ai.dashboard.ask_llm import answer_question

APP_TITLE = "MCP-AI • Catalog Dashboard"


# ---------------- helpers ----------------

def _badge(label: str, color: str = "gray"):
    st.markdown(
        f"""
        <span style="
          display:inline-block;background:{color};color:white;
          padding:2px 8px;border-radius:12px;font-size:12px;margin-right:6px;">
          {label}
        </span>
        """,
        unsafe_allow_html=True,
    )

def _pill(label: str):
    st.markdown(
        f"<span style='display:inline-block;background:#eef;padding:2px 8px;border-radius:10px;margin:2px'>{label}</span>",
        unsafe_allow_html=True,
    )

def _pick_first(vals: List[str]) -> Optional[str]:
    for v in vals:
        if v:
            return v
    return None

@st.cache_data(show_spinner=False)
def _cached_load_run_df(out_root: str, run_id: str) -> pd.DataFrame:
    return load_run_df(out_root, run_id)


def _calc_common_root(paths: List[str]) -> str:
    if not paths:
        return ""
    try:
        return os.path.commonpath(paths)
    except Exception:
        return ""


def _relative_top_folders(paths: List[str], root: str) -> List[str]:
    out = []
    for p in paths:
        try:
            rel = os.path.relpath(p, root) if root else p
        except Exception:
            rel = p
        parts = [pp for pp in rel.split(os.sep) if pp and pp != "."]
        if parts:
            out.append(parts[0])
    return sorted(list(dict.fromkeys(out)))  # unique, keep order


# ---------------- main ----------------

def main():
    # CLI args (streamlit passes unknown args after "--")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Professional triage • smart staging • clear insight")

    # ----- config / storage -----
    cfg = load_config(args.config)
    out_root = (cfg.get("storage", {}) or {}).get("out_root", "./mcp_ai/output_files")
    runs = find_runs(out_root)

    if not runs:
        st.error(f"No runs found under {out_root}/runs")
        st.stop()

    # sidebar: run + filters
    with st.sidebar:
        st.header("Run")
        run_ids = [r["run_id"] for r in runs]
        run_label_map = {r["run_id"]: f'{r["run_id"]}  ·  {r["files"]} files' for r in runs}
        sel_run = st.selectbox("Choose a run", run_ids, format_func=lambda rid: run_label_map.get(rid, rid))

        st.divider()
        st.header("Filters")

        # Load base df for folder computation
        base_df = _cached_load_run_df(out_root, sel_run)
        all_paths = base_df["path"].dropna().astype(str).tolist()
        common_root = _calc_common_root(all_paths)
        top_folders = _relative_top_folders(all_paths, common_root)

        folder = st.selectbox("Top folder", options=["(all)"] + top_folders, index=0)

        q = st.text_input("Search in summary/path/tags", "", placeholder="invoice OR 'order id' OR customer")
        domain = st.multiselect("Domain", options=[
            "finance","legal","hr","it","marketing","engineering","operations","healthcare","education","general"
        ])
        min_conf = st.slider("Min FAST confidence", 0.0, 1.0, 0.0, 0.05)
        pii_only = st.checkbox("PII only")
        only_needs_deep = st.checkbox("Auto-triage: Needs deep")

        st.divider()
        st.header("Actions")
        stage_btn = st.button("Create staging manifest from selected rows")
        st.caption("Writes JSONL of file paths inside the selected run folder.")

    # tabs
    tab_overview, tab_catalog, tab_assistant, tab_settings = st.tabs(
        ["Overview", "Catalog", "Assistant", "Settings"]
    )

    # -------------- data --------------
    df = base_df.copy()

    # triage thresholds from config
    deep_cfg = ((cfg.get("ai", {}) or {}).get("deep", {}) or {})
    low_conf_thr = float(((deep_cfg.get("auto_triggers", {}) or {}).get("low_confidence_below", 0.75)))

    # compute needs_deep mask once on base df
    needs_deep_mask = (
        (df["ai_fast_confidence"].fillna(0.0) < low_conf_thr) |
        (df["ai_fast_contains_pii"].fillna(False))
    )

    # apply UI filters
    df_filtered = filter_df(
        df,
        query=q,
        domains=domain,
        min_conf=min_conf,
        pii_only=pii_only,
        only_needs_deep=only_needs_deep,
        needs_deep_mask=needs_deep_mask,
        top_folder=(None if folder == "(all)" else folder),
        common_root=_calc_common_root(df["path"].dropna().astype(str).tolist()),
    )

    # -------- Overview --------
    with tab_overview:
        total = len(df)
        shown = len(df_filtered)
        pii_ct = int(df["ai_fast_contains_pii"].fillna(False).sum())
        lowconf_ct = int((df["ai_fast_confidence"].fillna(0.0) < low_conf_thr).sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Files (run)", total)
        c2.metric("Shown (filtered)", shown)
        c3.metric("PII suspected", pii_ct)
        c4.metric(f"Below conf < {low_conf_thr:.2f}", lowconf_ct)

        st.subheader("Overview")

        cc1, cc2, cc3 = st.columns([1,1,1])
        with cc1:
            ch = chart_confidence_hist(df)
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)
            else:
                st.info("No confidence data available.")
        with cc2:
            ch = chart_domain_bar(df)
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)
            else:
                st.info("No domain data available.")
        with cc3:
            ch = chart_pii_bar(df)
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)
            else:
                st.info("No PII flags to chart.")

        fig = chart_volume_over_time(df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        row2a, row2b = st.columns([1,1])
        with row2a:
            ch = chart_filetype_bar(df)
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)
            else:
                st.info("No file-type information to show.")
        with row2b:
            ch = chart_top_owners(df)
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)
            else:
                st.info("No owner information available for this run.")

    # -------- Catalog --------
    with tab_catalog:
        st.subheader("Catalog")
        view_cols = [
            "path","ai_fast_category","ai_fast_domain","ai_fast_confidence",
            "ai_fast_contains_pii","ai_fast_tags","ai_fast_summary",
            "ai_deep_summary","ai_deep_glossary_terms","ai_deep_pii",
            "meta_engine","meta_ext","extraction_status",
        ]
        fdf = df_filtered.copy()
        for c in view_cols:
            if c not in fdf.columns:
                fdf[c] = None

        sel_col = "_sel"
        fdf[sel_col] = False
        _tbl = st.data_editor(
            fdf[[sel_col] + view_cols],
            hide_index=True,
            height=420,
            column_config={
                sel_col: st.column_config.CheckboxColumn("Select"),
                "ai_fast_confidence": st.column_config.NumberColumn(format="%.2f"),
                "ai_fast_tags": st.column_config.ListColumn(width="medium"),
                "ai_deep_glossary_terms": st.column_config.ListColumn(width="medium"),
                "ai_deep_pii": st.column_config.ListColumn(width="small"),
            },
            use_container_width=True,
            key="catalog_table",
        )
        selected_paths = set(_tbl.loc[_tbl[sel_col] == True, "path"].dropna().astype(str).tolist())

        # Detail pane
        st.subheader("Detail")
        detail_path = _pick_first(list(selected_paths)) or (fdf.iloc[0]["path"] if len(fdf) else None)
        if not detail_path:
            st.info("Select a row to see details.")
        else:
            row = df.loc[df["path"] == detail_path]
            if row.empty:
                st.info("Select a valid row.")
            else:
                r = row.iloc[0].to_dict()
                meta = {
                    "ext": r.get("meta_ext"),
                    "engine": r.get("meta_engine"),
                    "pages": r.get("meta_pages"),
                    "dpi": r.get("meta_dpi"),
                    "language": r.get("meta_language"),
                }
                fast = {
                    "category": r.get("ai_fast_category"),
                    "domain": r.get("ai_fast_domain"),
                    "tags": r.get("ai_fast_tags") or [],
                    "contains_pii": r.get("ai_fast_contains_pii"),
                    "confidence": r.get("ai_fast_confidence"),
                    "summary": r.get("ai_fast_summary"),
                }
                deep = {
                    "summary": r.get("ai_deep_summary"),
                    "glossary_terms": r.get("ai_deep_glossary_terms") or [],
                    "pii": r.get("ai_deep_pii") or [],
                    "confidence": r.get("ai_deep_confidence"),
                }

                cA, cB = st.columns([1.6, 1])
                with cA:
                    st.write(f"**Path**: `{r['path']}`")
                    st.write(f"**Extraction**: `{r.get('extraction_status','')}`  ·  **Engine**: `{meta.get('engine')}`  ·  **Ext**: `{meta.get('ext')}`")
                    st.write(f"**FAST**: {fast['category'] or '—'}  ·  {fast['domain'] or '—'}  ·  conf={fast['confidence'] if fast['confidence'] is not None else '—'}")
                    if fast["contains_pii"]:
                        _badge("PII suspected", "#c0392b")
                    else:
                        _badge("No PII", "#7f8c8d")
                    st.write(f"**FAST summary**: {fast['summary'] or '—'}")
                    if deep["summary"]:
                        st.markdown("**DEEP summary**")
                        st.write(deep["summary"])
                    if deep["pii"]:
                        st.markdown("**PII types**")
                        for p in deep["pii"]:
                            _pill(p)
                    if deep["glossary_terms"]:
                        st.markdown("**Glossary terms**")
                        for g in deep["glossary_terms"]:
                            _pill(g)

                with cB:
                    st.markdown("**Tags**")
                    for t in fast["tags"]:
                        _pill(t)
                    st.markdown("**Meta**")
                    st.json({k: v for k, v in meta.items() if v not in (None, "")})

        # Staging manifest action
        if stage_btn:
            if not selected_paths:
                st.warning("Select one or more rows first.")
            else:
                run_dir = os.path.join(str(out_root), "runs", str(sel_run))
                mpath = stage_manifest_path(run_dir)
                wrote = write_manifest(mpath, sorted(selected_paths))
                if wrote:
                    st.success(f"Staging manifest written: {mpath}")
                else:
                    st.error("Could not write manifest (check permissions).")

        st.divider()
        st.subheader("Export (filtered view)")
        render_export_buttons(df_filtered)

    # -------- Assistant --------
    with tab_assistant:
        st.subheader("Ask about this run")
        q = st.text_input("Question", "", placeholder="Examples: how many files total? which invoices mention 'VINET'? files tagged legal?")
        if q.strip():
            ans, preview_df = answer_question(df, q, selection_df=df_filtered)
            st.write(ans)
            if preview_df is not None and not preview_df.empty:
                st.dataframe(preview_df, use_container_width=True, height=360)

    # -------- Settings --------
    with tab_settings:
        render_config_inspector(cfg_path=args.config)


if __name__ == "__main__":
    main()