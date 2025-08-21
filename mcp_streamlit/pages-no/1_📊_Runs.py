# /Users/vamsi_mure/Documents/mcp_ai-v2/mcp_ai/mcp_streamlit/pages/1_📊_Runs.py
from __future__ import annotations
import streamlit as st
import pandas as pd

from data import get_data, ensure_session_data

# Make sure DATA exists for this page
DATA = get_data() or ensure_session_data()

st.title("📊 Runs Summary")

if DATA and isinstance(DATA.df, pd.DataFrame) and not DATA.df.empty:
    df = DATA.df

    # Defensive access (column might not exist in some runs)
    col1, col2, col3, col4 = st.columns(4)
    if "extraction_status" in df.columns:
        ok = (df["extraction_status"] == "ok").sum()
        mo = (df["extraction_status"] == "metadata_only").sum()
        er = (df["extraction_status"] == "error").sum()
    else:
        ok = mo = er = 0

    col1.metric("Files", int(len(df)))
    col2.metric("OK", int(ok))
    col3.metric("Metadata Only", int(mo))
    col4.metric("Errors", int(er))

    st.dataframe(df.fillna(""), use_container_width=True, height=600)
else:
    st.info("Load a run from the sidebar in the main page.")