# mcp_ai/scanner.py
from __future__ import annotations
import fnmatch, os
from typing import Dict, Generator, Iterable, List, Optional

_DEFAULT_EXCLUDES = [
    "**/.git/**",
    "**/tmp/**",
    "**/__MACOSX/**",
    "**/.DS_Store",
    "**/Thumbs.db",
    "**/desktop.ini",
    "**/.*",           # any hidden file/dir
]

_SYSTEM_BASENAMES = {
    ".DS_Store", "Thumbs.db", "desktop.ini",
}

def _norm(p: str) -> str:
    return os.path.normpath(p)

def _is_hidden(path: str) -> bool:
    # treat any path component that starts with '.' as hidden
    for comp in _norm(path).split(os.sep):
        if comp.startswith(".") and comp not in (".", ".."):
            return True
    return False

def _is_system_file(path: str) -> bool:
    base = os.path.basename(path)
    return base in _SYSTEM_BASENAMES

def _compile_excludes(cfg_excludes: Optional[List[str]]) -> List[str]:
    pats = list(_DEFAULT_EXCLUDES)
    if cfg_excludes:
        pats.extend(cfg_excludes)
    # dedupe while preserving order
    seen = set()
    out: List[str] = []
    for p in pats:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    norm = _norm(path)
    for pat in patterns:
        if fnmatch.fnmatch(norm, pat):
            return True
    return False

def scan_files(cfg: Dict) -> Generator[Dict, None, None]:
    """
    Yields dicts like:
      { "path": str, "size_bytes": int, "sha1": None }
    Notes:
      - filters hidden/OS system files
      - honors paths.roots and paths.exclude_globs
    """
    roots = (cfg.get("paths", {}) or {}).get("roots", []) or []
    excl = _compile_excludes((cfg.get("paths", {}) or {}).get("exclude_globs", []))

    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # prune hidden and excluded dirs in-place for performance
            pruned = []
            for d in list(dirnames):
                full = os.path.join(dirpath, d)
                if _is_hidden(full) or _matches_any(full, excl):
                    pruned.append(d)
            for d in pruned:
                dirnames.remove(d)

            for fn in filenames:
                path = os.path.join(dirpath, fn)
                if _is_hidden(path) or _is_system_file(path) or _matches_any(path, excl):
                    continue

                try:
                    size_bytes = os.path.getsize(path)
                except OSError:
                    size_bytes = None

                yield {
                    "path": path,
                    "size_bytes": size_bytes,
                    "sha1": None,    # left None here; compute later if you need it
                }