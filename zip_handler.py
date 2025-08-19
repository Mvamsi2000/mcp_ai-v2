# mcp_ai/zip_handler.py
from __future__ import annotations
import zipfile, os
from typing import Optional

def safe_unzip(zip_path: str, out_dir: str) -> Optional[str]:
    """
    Minimal unzip helper if you later want to expand archives.
    Returns the directory it extracted to, or None on failure.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        return out_dir
    except Exception:
        return None