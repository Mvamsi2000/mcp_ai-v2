# mcp_ai/dashboard/export_utils.py
from __future__ import annotations
import io, json
from typing import Optional

import pandas as pd
import streamlit as st


def _to_jsonl_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    for rec in df.to_dict(orient="records"):
        buf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return buf.getvalue().encode("utf-8")


def render_export_buttons(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("Nothing to export.")
        return
    # CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="catalog_filtered.csv",
        mime="text/csv",
        use_container_width=False,
    )
    # JSONL
    jsonl_bytes = _to_jsonl_bytes(df)
    st.download_button(
        "Download JSONL",
        data=jsonl_bytes,
        file_name="catalog_filtered.jsonl",
        mime="application/x-ndjson",
        use_container_width=False,
    )