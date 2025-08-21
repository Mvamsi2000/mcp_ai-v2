from __future__ import annotations
import streamlit as st
from streamlit.components.v1 import html as st_html
from graph import build_graph_html

DATA = st.session_state["DATA"]

st.title("🕸️ Relationship Graph")
if not DATA.graph_edges:
    st.info("No graph_edges.jsonl found in this run.")
    st.stop()

html = build_graph_html(DATA.graph_edges)
st_html(html, height=720, scrolling=True)