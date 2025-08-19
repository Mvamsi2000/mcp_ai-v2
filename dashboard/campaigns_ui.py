# mcp_ai/dashboard/campaigns_ui.py
from __future__ import annotations
import os
import streamlit as st

from .data_access import stage_manifest_path

def _code(cmd: str):
    st.code(cmd, language="bash")

def render_campaigns_panel(cfg: dict, out_root: str, run_id: str):
    st.subheader("Campaigns")
    st.caption("Copy-and-run commands for triage and targeted deep runs.")

    cfg_path = os.path.join(os.getcwd(), "mcp_ai", "config.yaml")
    run_dir = os.path.join(out_root, "runs", str(run_id))
    manifest_path = stage_manifest_path(run_dir)

    st.markdown("### 1) Fast triage across configured roots")
    _code(f"python -m mcp_ai.main --config {cfg_path} --profile basic")

    st.markdown("### 2) Deep scan (local) on staged manifest")
    _code(f"python -m mcp_ai.run --config {cfg_path} --manifest {manifest_path} --mode deep-local")

    st.markdown("### 3) Deep scan (cloud) on staged manifest")
    _code(f"python -m mcp_ai.run --config {cfg_path} --manifest {manifest_path} --mode deep-cloud")

    st.divider()
    st.markdown("""
- The **Catalog** tab writes `stage_manifest.jsonl` in the run folder.
- Tune `ai.deep.policy` and `auto_triggers` in config for cost/speed control.
""")