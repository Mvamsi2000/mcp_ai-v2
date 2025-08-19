# mcp_ai/dashboard/ask_llm.py
from __future__ import annotations
import re
from typing import Optional, Tuple

import pandas as pd


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    for c in ["ai_fast_confidence"]:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")
    for c in ["ai_fast_contains_pii"]:
        if c in w.columns:
            w[c] = w[c].fillna(False).astype(bool)
    for c in ["ai_fast_tags", "ai_deep_glossary_terms", "ai_deep_pii"]:
        if c in w.columns:
            w[c] = w[c].apply(lambda v: v if isinstance(v, list) else ([] if pd.isna(v) else [v]))
    for c in ["ai_fast_summary", "ai_fast_category", "ai_fast_domain", "path"]:
        if c in w.columns:
            w[c] = w[c].astype(str)
    return w


def _contains_ci(s: str, needle: str) -> bool:
    return needle.lower() in (s or "").lower()


def _search_any(row: pd.Series, token: str) -> bool:
    if _contains_ci(row.get("path", ""), token): return True
    if _contains_ci(row.get("ai_fast_summary", ""), token): return True
    for lst_col in ("ai_fast_tags", "ai_deep_glossary_terms"):
        arr = row.get(lst_col) or []
        if any(_contains_ci(str(x), token) for x in arr):
            return True
    return False


def answer_question(
    df: pd.DataFrame,
    question: str,
    *,
    selection_df: Optional[pd.DataFrame] = None,
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Simple rule-based QA over the run DataFrame.
    Returns: (answer_text, optional_preview_df)
    """
    if df is None or df.empty:
        return ("I don't see any items in this run.", None)

    w = _normalize(df)
    q = (question or "").strip()

    # 1) total count
    if re.search(r"\bhow many files\b", q, re.I):
        scope = selection_df if (selection_df is not None and not selection_df.empty) else w
        return (f"Files in scope: {len(scope)}.", None)

    # 2) count by folder (top folder)
    m = re.search(r"how many files in (?:the )?(.+?) folder", q, re.I)
    if m:
        folder = m.group(1).strip().strip("'\"")
        def _top(p: str) -> str:
            parts = [pp for pp in str(p).split("/") if pp and pp != "."]
            return parts[0] if parts else ""
        mask = w["path"].astype(str).apply(_top).str.lower() == folder.lower()
        return (f"Files in folder '{folder}': {int(mask.sum())}.", w.loc[mask, ["path"]].head(50))

    # 3) by domain (e.g., legal, finance)
    m = re.search(r"how many files (?:are )?tagged as (\w+)", q, re.I)
    if m:
        dom = m.group(1).lower()
        mask = w.get("ai_fast_domain", "").str.lower() == dom
        return (f"Files tagged as {dom}: {int(mask.sum())}.", w.loc[mask, ["path","ai_fast_category","ai_fast_domain"]].head(50))

    # 4) invoices related to PERSON/COMPANY/DATE
    m = re.search(r"(?:which|show)\s+invoices?.*?(?:related to|for|mention(?:ing)?)\s+(.+)", q, re.I)
    if m:
        token = m.group(1).strip().strip("'\"")
        mask = (
            w.get("ai_fast_category","").str.contains("invoice", case=False, na=False) &
            w.apply(lambda r: _search_any(r, token), axis=1)
        )
        preview = w.loc[mask, ["path","ai_fast_summary"]].head(50)
        return (f"Invoices mentioning '{token}': {len(preview)} (showing up to 50).", preview)

    # 5) owners (if owner columns exist)
    for col in ("owner","meta_owner","file_owner"):
        if col in w.columns:
            m = re.search(r"(?:which|show)\s+files\s+(?:owned by|for owner)\s+(.+)", q, re.I)
            if m:
                who = m.group(1).strip().strip("'\"")
                mask = w[col].astype(str).str.contains(who, case=False, na=False)
                preview = w.loc[mask, ["path", col]].head(100)
                return (f"Files owned by '{who}': {len(preview)} (showing up to 100).", preview)
            if re.search(r"what (?:are|are the) (?:different )?owners", q, re.I):
                owners = (w[col].dropna().astype(str).value_counts().head(50)).reset_index()
                owners.columns = ["owner","files"]
                return ("Top owners:", owners)

    # 6) fallback keyword search
    if q:
        ors = [tok.strip() for tok in re.split(r"\s+OR\s+|\|", q, flags=re.I) if tok.strip()]
        if ors:
            mask = pd.Series(False, index=w.index)
            for tok in ors:
                mask = mask | w.apply(lambda r: _search_any(r, tok), axis=1)
            preview = w.loc[mask, ["path","ai_fast_category","ai_fast_summary"]].head(100)
            return (f"Matches for {ors}: {len(preview)} (showing up to 100).", preview)

    return ("I can answer counts by domain/folder/PII; find invoices by keyword; list owners; and do keyword searches. Try: `how many files tagged as legal?`, `which invoices mention 'VINET'?`, `what are the different owners?`", None)