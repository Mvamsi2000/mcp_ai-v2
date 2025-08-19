# mcp_ai/dashboard/config_ui.py
from __future__ import annotations
import os

import streamlit as st


def render_config_inspector(*, cfg_path: str):
    st.caption("Config file")
    st.code(os.path.abspath(cfg_path), language=None)
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="yaml")
    except Exception as e:
        st.error(f"Could not read config: {e}")