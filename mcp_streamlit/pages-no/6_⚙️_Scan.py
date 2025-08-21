from __future__ import annotations
import streamlit as st
import subprocess
import shlex
import threading

from config import settings
from data_store import DataStore

DATA = st.session_state["DATA"]

st.title("⚙️ Trigger Scan / Live Logs")

if not settings.SCAN_COMMAND:
    st.info("Set SCAN_COMMAND in environment or Config.yaml to enable one-click scan.\n\nExample:\n\n`SCAN_COMMAND=\"python -m mcp_ai.cli.scan --input ./mcp_ai/input_files --output ./mcp_ai/output_files\"`")
    st.stop()

cmd = st.text_input("Scan command", value=settings.SCAN_COMMAND)

log_area = st.empty()
btn = st.button("Run Scan")

def run_and_stream(command: str):
    try:
        proc = subprocess.Popen(
            shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in iter(proc.stdout.readline, ''):
            log_area.text(line.rstrip())
        proc.stdout.close()
        proc.wait()
        log_area.text(f"[exit {proc.returncode}]")
    except Exception as e:
        log_area.text(f"scan failed: {e}")

if btn and cmd.strip():
    threading.Thread(target=run_and_stream, args=(cmd,), daemon=True).start()
    st.success("Scan started. Logs will stream above.")

st.divider()
if st.button("Reload latest run"):
    try:
        DATA.load()  # latest
        st.success(f"Reloaded run {DATA.run_id} with {len(DATA.items)} items.")
    except Exception as e:
        st.error(str(e))