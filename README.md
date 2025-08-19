# MCP-AI (Multi-Contextual Catalog Platform — AI) — Demo MVP

**Status:** MVP demo implementing: scanner → extractor → AI router (none/local/cloud_demo) → insights → campaigns (API) → Ask MCP (NL→SQL) → Streamlit dashboard.

> **Golden Rules**
> - **Fast first**: scanning & hashing never block on AI.
> - **Config over code**: all knobs in `config.yaml`.
> - **Private by default**: outbound network **OFF** unless enabled.
> - **AI modes**: `none | local | cloud_demo` (later `cloud_real`).
> - **Budgets enforced**: run + per-file caps with auto-downgrade.
> - **Auditable**: decisions logged and recorded to SQLite.

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install streamlit pdfminer.six PyMuPDF python-docx openpyxl beautifulsoup4 html2text pytesseract pillow python-magic
# Optional extras: spacy, easyocr, tesseract-ocr system binary

# Run a basic scan (edit roots first!)
python -m mcp_ai.main --config ./mcp_ai/config.yaml

# Launch Streamlit UI
streamlit run mcp_ai/dashboard.py
```

> **Note**: The default `paths.roots` point to `/mnt/share/...` which likely won't exist on your machine. Update these in `config.yaml` to a small sample directory first.

---

## What’s Implemented

- **Config-driven** (`config.yaml`): modes, profiles, budgets, safety, storage locations.
- **Scanner**: recursive walk, stat, mime/magic, SHA1; streams to JSONL + CSV and writes to SQLite.
- **Extractor**: type policies (`skip`, `metadata_only`, `prefer_ocr`, `try_text_then_metadata_only`), text extraction (PDF/DOCX/XLSX/HTML/TXT), OCR timeboxing (if libraries present).
- **AI Router**: `none | local | cloud_demo`:
  - **Fast pass**: heuristic category/tags/PII/confidence.
  - **Deep pass**: conditional on confidence or high-value folders; demo uses **fixtures** with simulated latency. Local deep falls back to heuristics if no LLM.
  - **Redaction**: regex masking for emails/phones/IDs before "cloud".
  - **Budgets**: simple ledger with flat demo costs.
- **Insights**: stale years, duplicates, business value score, recommendation with evidence.
- **Campaigns (MVP)**: create a campaign and select files via a SQL-like WHERE clause (admin input).
- **Ask MCP**: rule-based NL→SQL for common questions, shows SQL, export CSV/JSON in UI.
- **Streamlit Dashboard**: Pre-scan mode picker, safety/budget, run button, key metrics, Ask MCP with export.
- **Outputs**:
  - `out/metadata_catalog.jsonl` (streamed during scan)
  - `out/metadata_catalog.csv` (metadata-only view)
  - `out/state.sqlite` (full relational store)

---

## Acceptance Criteria Mapping

- ✅ Run **Basic**, **Deep–Local**, **Deep–Cloud (Demo)** end-to-end.
- ✅ Write **JSONL** as it scans, plus **CSV**, and record to **SQLite** (resumable runner; idempotence by SHA handling kept simple in MVP).
- ✅ Produce **insights** (stale/duplicates/recommendations) and allow CSV/JSON exports via Ask MCP.
- ✅ **Campaigns** primitives present (create + select files by criteria).
- ✅ **Ask MCP** answers common inventory/risk/cost questions (rule-based NL→SQL) and exports.
- ✅ **Safety**: network off by default; demo cloud uses local fixtures; redaction before any cloud path.

---

## Fixtures (Cloud Demo)

Add JSON files under `mcp_ai/fixtures/`. Name them as `<sha1>.json` to simulate deep cloud enrichment for a given file. Example content:

```json
{
  "confidence": 0.91,
  "pii": ["email","phone"],
  "glossary_terms": ["Termination Clause","PO Number"]
}
```

If a file has no matching fixture, the system falls back to a local deep heuristic when the strategy is `hybrid`.

---

## Execution Modes

- **CLI**: `python -m mcp_ai.main --config ./mcp_ai/config.yaml`
- **Streamlit**: `streamlit run mcp_ai/dashboard.py` for pre-scan picker, budgets, Ask MCP, exports.
- **Config-only**: Switch defaults in `config.yaml`.

---

## Notes & Next Steps

- Add concurrency (async / thread pools) for scanner & extractor.
- Strengthen resume logic to skip unchanged files (size + mtime) across runs.
- Pluggable OCR pipelines (easyocr/layoutparser) and corrupted-PDF healing as per your Phase 2 plan.
- NL→SQL via a local LLM when **local** mode is enabled (and Ollama available).
- Campaign scheduler with budgets/timeboxes and resume checkpoints.


---

## Local LLM (Ollama) — Deep – Local AI

1. Install Ollama and pull a model:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh    # or use your OS package
   ollama pull mistral:7b-instruct
   ollama serve  # ensure the server runs at http://localhost:11434
   ```

2. In `mcp_ai/config.yaml` set:
   ```yaml
   ai:
     mode: local
   ollama:
     endpoint: "http://localhost:11434"
     model: "mistral:7b-instruct"
   ```

3. Run a Deep – Local AI scan via Streamlit or CLI. The router will call Ollama for the **deep pass** (and optionally fast pass) and fall back to heuristics if the server is unavailable.
