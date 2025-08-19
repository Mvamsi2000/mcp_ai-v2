# mcp_ai/ask_mcp.py
from __future__ import annotations
import sqlite3, csv, json, os, io, re, datetime
from typing import Dict, Any, Tuple, List
import argparse, sqlite3, json
EXAMPLES = {
    "how many files have pii": "SELECT COUNT(*) AS count FROM ai_fast WHERE contains_pii=1",
    "show duplicates": "SELECT sha1, COUNT(*) as c FROM files GROUP BY sha1 HAVING c>1 ORDER BY c DESC LIMIT 50",
    "top categories": "SELECT category, COUNT(*) as c FROM ai_fast GROUP BY category ORDER BY c DESC LIMIT 20",
    "largest files": "SELECT path, size_bytes FROM files ORDER BY size_bytes DESC LIMIT 20",
    "stale files": "SELECT f.path, i.stale_years FROM files f JOIN insights i ON f.file_id=i.file_id WHERE i.stale_years >= 6 ORDER BY i.stale_years DESC LIMIT 50",
}

def nl_to_sql(question: str) -> str:
    q = question.strip().lower()
    # simple rule-based mapping
    for k, sql in EXAMPLES.items():
        if all(w in q for w in k.split()):
            return sql
    if "pii" in q and "list" in q:
        return "SELECT f.path, af.contains_pii FROM files f JOIN ai_fast af ON f.file_id=af.file_id WHERE af.contains_pii=1 LIMIT 100"
    if "recent" in q and "files" in q:
        return "SELECT path, last_accessed FROM files ORDER BY last_accessed DESC LIMIT 50"
    # default fallback
    return "SELECT path, size_bytes FROM files ORDER BY size_bytes DESC LIMIT 10"

def run_query(db_path: str, sql: str) -> Tuple[List[str], List[Any]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = cur.execute(sql).fetchall()
    cols = [d[0] for d in cur.description]
    con.close()
    return cols, rows

def export_rows(cols, rows, fmt: str = "csv") -> bytes:
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(cols)
        for r in rows:
            w.writerow(r)
        return buf.getvalue().encode("utf-8")
    else:
        arr = [dict(zip(cols, r)) for r in rows]
        return json.dumps(arr, ensure_ascii=False, indent=2).encode("utf-8")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./mcp_ai/output_files/state.sqlite")
    ap.add_argument("--sql", default="SELECT id, path, extraction_status FROM files LIMIT 20")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(args.sql)
    rows = cur.fetchall()
    for r in rows:
        print(json.dumps(dict(r), ensure_ascii=False))
    con.close()

if __name__ == "__main__":
    main()