# mcp_ai/utils.py
from __future__ import annotations
import csv, fnmatch, glob, hashlib, io, json, logging, os, sys, time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

log = logging.getLogger("utils")

# ───────────────────────────── YAML loader ─────────────────────────────

def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data

# ───────────────────────────── Filesystem helpers ─────────────────────────────

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def sha1_of_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = 0
    v = float(n)
    while v >= 1024 and s < len(units) - 1:
        v /= 1024.0
        s += 1
    return f"{v:.1f} {units[s]}"

def ext_lower(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def match_any(path: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(path, pat):
            return True
    return False

# ───────────────────────────── Logging ─────────────────────────────

def setup_logging(cfg: Dict[str, Any]) -> None:
    lg = cfg.get("logging") or {}
    level = getattr(logging, str(lg.get("level", "INFO")).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if lg.get("to_file"):
        file_path = lg.get("file_path") or "./mcp_ai/output_files/mcp_ai.log"
        ensure_dir(os.path.dirname(file_path))
        rotate_bytes = int(lg.get("rotate_bytes", 10_485_760))
        backups = int(lg.get("backups", 3))
        fh = RotatingFileHandler(file_path, maxBytes=rotate_bytes, backupCount=backups, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)

# ───────────────────────────── Budget tracking ─────────────────────────────

@dataclass
class BudgetLedger:
    run_limit_usd: float
    per_file_limit_usd: float
    spent_usd: float = 0.0

    def charge(self, amount: float) -> bool:
        """Return True if charge accepted, False if budget exceeded."""
        if amount < 0:
            amount = 0.0
        if self.spent_usd + amount > self.run_limit_usd:
            return False
        self.spent_usd += amount
        return True

# ───────────────────────────── Writers ─────────────────────────────

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def append_csv_row(path: str, header: List[str], row: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: _csv_cell(row.get(k)) for k in header})

def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

# ───────────────────────────── Iteration ─────────────────────────────

def iter_paths(roots: List[str], exclude_globs: List[str]) -> Generator[str, None, None]:
    for root in roots:
        root = os.path.abspath(root)
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            norm = dirpath.replace("\\", "/")
            # apply exclude patterns on directory path
            if match_any(norm, exclude_globs):
                dirnames[:] = []  # do not descend
                continue
            for name in filenames:
                p = os.path.join(dirpath, name)
                p_norm = p.replace("\\", "/")
                if match_any(p_norm, exclude_globs):
                    continue
                yield p

# ───────────────────────────── Small utilities ─────────────────────────────
# --- Networking guard ---------------------------------------------------------

def is_network_allowed(cfg: dict | None) -> bool:
    """
    Returns True if outbound network calls are allowed by config.
    Used to decide whether to attempt real cloud requests.
    """
    try:
        return bool(((cfg or {}).get("safety") or {}).get("allow_outbound_network", False))
    except Exception:
        return False

def now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat()

def coalesce(*vals: Any, default: Any = None) -> Any:
    for v in vals:
        if v is not None:
            return v
    return default