from __future__ import annotations
import streamlit as st

DATA = st.session_state["DATA"]

st.title("🔎 File Inspector")
if DATA.df is None or DATA.df.empty:
    st.info("No data. Load a run in the sidebar.")
    st.stop()

filenames = [f"{i['filename']} — {i['path']}" for i in DATA.items]
sel = st.selectbox("Choose a file", filenames, index=0)
idx = filenames.index(sel)
rec = DATA.items[idx]

c1, c2 = st.columns([2,1])
with c1:
    st.subheader(rec["filename"])
    st.caption(rec["path"])
    st.write(f"**Status:** {rec['extraction_status']}")
    st.write(f"**Engine:** {rec.get('engine','-')}, **Lang:** {rec.get('language','-')}")
    st.write(f"**Summary:** {rec.get('summary','') or '-'}")
with c2:
    st.write("**Meta**")
    st.json({k:v for k,v in rec.items() if k not in ("text",)})

st.divider()
st.subheader("Extracted Text")
txt = rec.get("text","")
if txt:
    st.code(txt[:20000], language="markdown")
else:
    st.info("No text available for this item.")