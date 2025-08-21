from __future__ import annotations
import streamlit as st
import pandas as pd

DATA = st.session_state["DATA"]

st.title("📊 Runs Summary")
if not DATA.df is None and not DATA.df.empty:
    df = DATA.df
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        ok = (df["extraction_status"] == "ok").sum()
        mo = (df["extraction_status"] == "metadata_only").sum()
        er = (df["extraction_status"] == "error").sum() if "error" in df["extraction_status"].values else 0
        col1.metric("Files", len(df))
        col2.metric("OK", int(ok))
        col3.metric("Metadata Only", int(mo))
        col4.metric("Errors", int(er))
    st.dataframe(df.fillna(""), use_container_width=True, height=600)
else:
    st.info("Load a run from the sidebar.")