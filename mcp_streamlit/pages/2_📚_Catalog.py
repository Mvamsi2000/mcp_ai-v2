from __future__ import annotations
import streamlit as st
import pandas as pd

DATA = st.session_state["DATA"]

st.title("📚 Catalog")
if DATA.df is None or DATA.df.empty:
    st.info("No data. Load a run in the sidebar.")
    st.stop()

df = DATA.df.copy()
with st.expander("Filter"):
    c1, c2, c3 = st.columns(3)
    ext = c1.text_input("Ext filter (e.g. .pdf)")
    dom = c2.text_input("Domain contains")
    tag = c3.text_input("Tag contains")
    if ext:
        df = df[df["ext"] == ext]
    if dom:
        df = df[df["domain"].astype(str).str.contains(dom, case=False, na=False)]
    if tag:
        df = df[df["tags"].astype(str).str.contains(tag, case=False, na=False)]

st.dataframe(df.fillna(""), use_container_width=True, height=650)