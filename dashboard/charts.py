# mcp_ai/dashboard/charts.py
from __future__ import annotations
import os
from typing import Optional

import pandas as pd
import altair as alt
import plotly.express as px
from plotly.graph_objs import Figure

# avoid 5k-row Altair limit
alt.data_transformers.disable_max_rows()


def _is_empty(df: pd.DataFrame) -> bool:
    return (df is None) or df.empty


def chart_confidence_hist(df: pd.DataFrame) -> Optional[alt.Chart]:
    if _is_empty(df) or "ai_fast_confidence" not in df.columns:
        return None
    work = df.copy()
    work["ai_fast_confidence"] = pd.to_numeric(work["ai_fast_confidence"], errors="coerce")
    work = work.dropna(subset=["ai_fast_confidence"])
    if work.empty:
        return None
    return (
        alt.Chart(work)
        .mark_bar()
        .encode(
            x=alt.X("ai_fast_confidence:Q", bin=alt.Bin(maxbins=20), title="FAST confidence"),
            y=alt.Y("count()", title="Files"),
            tooltip=[alt.Tooltip("count()"), alt.Tooltip("ai_fast_confidence:Q", bin=True)],
        )
        .properties(height=220)
    )


def chart_domain_bar(df: pd.DataFrame) -> Optional[alt.Chart]:
    if _is_empty(df) or "ai_fast_domain" not in df.columns:
        return None
    work = df.copy()
    work["ai_fast_domain"] = work["ai_fast_domain"].fillna("unknown")
    agg = (
        work.groupby("ai_fast_domain", dropna=False)
        .size()
        .reset_index(name="files")
        .sort_values("files", ascending=False)
    )
    if agg.empty:
        return None
    return (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("files:Q", title="Files"),
            y=alt.Y("ai_fast_domain:N", sort="-x", title="Domain"),
            tooltip=["ai_fast_domain", "files"],
        )
        .properties(height=220)
    )


def chart_pii_bar(df: pd.DataFrame) -> Optional[alt.Chart]:
    if _is_empty(df) or "ai_fast_contains_pii" not in df.columns:
        return None
    work = df.copy()
    work["has_pii"] = work["ai_fast_contains_pii"].fillna(False).astype(bool)
    agg = (
        work.groupby("has_pii")
        .size()
        .reset_index(name="files")
        .replace({True: "PII suspected", False: "No PII"})
        .rename(columns={"has_pii": "status"})
        .sort_values("files", ascending=False)
    )
    if agg.empty:
        return None
    return (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("files:Q", title="Files"),
            y=alt.Y("status:N", sort="-x", title="PII"),
            tooltip=["status", "files"],
        )
        .properties(height=220)
    )


def chart_volume_over_time(df: pd.DataFrame) -> Optional[Figure]:
    if _is_empty(df) or "timestamp" not in df.columns:
        return None
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"])
    if work.empty:
        return None
    work["date"] = work["timestamp"].dt.date
    agg = work.groupby("date").size().reset_index(name="files")
    if agg.empty:
        return None
    fig = px.bar(agg, x="date", y="files", labels={"date": "Date", "files": "Files"})
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ----- File-type mix -----

def _ensure_ext_series(df: pd.DataFrame) -> pd.Series:
    if "meta_ext" in df.columns and df["meta_ext"].notna().any():
        ext = df["meta_ext"].fillna("")
    else:
        paths = df.get("path", pd.Series([], dtype=str)).fillna("")
        ext = paths.apply(lambda p: (os.path.splitext(p)[1] or "").lower())
    ext = ext.astype(str).str.lower().str.replace(".", "", regex=False)
    ext = ext.where(ext != "", other="unknown")
    return ext


def chart_filetype_bar(df: pd.DataFrame, top_n: int = 15) -> Optional[alt.Chart]:
    if _is_empty(df):
        return None
    work = df.copy()
    work["ext"] = _ensure_ext_series(work)
    agg = (
        work.groupby("ext")
        .size()
        .reset_index(name="files")
        .sort_values("files", ascending=False)
        .head(top_n)
    )
    if agg.empty:
        return None
    return (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("files:Q", title="Files"),
            y=alt.Y("ext:N", sort="-x", title="File type (ext)"),
            tooltip=["ext", "files"],
        )
        .properties(height=240)
    )


# ----- Top owners (if present) -----

def _owner_series(df: pd.DataFrame) -> pd.Series:
    for col in ("owner", "meta_owner", "file_owner"):
        if col in df.columns and df[col].notna().any():
            return df[col].astype(str)
    return pd.Series([], dtype=str)


def chart_top_owners(df: pd.DataFrame, top_n: int = 15) -> Optional[alt.Chart]:
    if _is_empty(df):
        return None
    owners = _owner_series(df)
    if owners.empty:
        return None
    work = pd.DataFrame({"owner": owners.fillna("").replace("", pd.NA)}).dropna()
    if work.empty:
        return None
    agg = (
        work.groupby("owner")
        .size()
        .reset_index(name="files")
        .sort_values("files", ascending=False)
        .head(top_n)
    )
    if agg.empty:
        return None
    return (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("files:Q", title="Files"),
            y=alt.Y("owner:N", sort="-x", title="Owner"),
            tooltip=["owner", "files"],
        )
        .properties(height=240)
    )