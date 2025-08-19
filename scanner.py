# mcp_ai/scanner.py
from __future__ import annotations
import logging, os
from typing import Any, Dict, Generator, Iterable, List, Tuple

from .utils import iter_paths, sha1_of_file, human_bytes, ext_lower

log = logging.getLogger("scanner")

def scan_files(cfg: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Yields file dicts with basic FS metadata.
    Respects include_types, max_file_mb in the chosen profile, and exclude_globs.
    """
    roots = (cfg.get("paths", {}) or {}).get("roots", []) or []
    exclude = (cfg.get("paths", {}) or {}).get("exclude_globs", []) or []
    profile_name = cfg.get("scan_profile", "deep")
    profile = (cfg.get("profiles", {}) or {}).get(profile_name, {}) or {}

    include_types = profile.get("include_types", ["*"])
    include_all = "*" in include_types
    include_exts = set([e.lower() for e in include_types if e != "*"])

    max_mb = float(profile.get("max_file_mb", 100))

    count = 0
    for path in iter_paths(roots, exclude):
        try:
            st = os.stat(path)
            size = st.st_size
            ext = ext_lower(path)
            if not include_all and ext not in include_exts:
                continue
            if size > max_mb * 1024 * 1024:
                log.info("skip (size>%sMB): %s", int(max_mb), path)
                continue
            count += 1
            yield {
                "path": path,
                "size_bytes": size,
                "ext": ext,
                "sha1": try_sha1(path),
                "created": iso_time(st.st_ctime),
                "modified": iso_time(st.st_mtime),
            }
        except Exception as e:
            log.warning("scan error on %s: %r", path, e)
            continue
    log.info("scanner: yielded %s files", count)

def try_sha1(path: str) -> str:
    try:
        return sha1_of_file(path)
    except Exception:
        return ""

def iso_time(ts: float) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()