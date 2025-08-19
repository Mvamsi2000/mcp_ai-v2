# mcp_ai/extractor.py
from __future__ import annotations
import io, logging, os
from typing import Any, Dict, List, Optional, Tuple

from .utils import ext_lower

log = logging.getLogger("extractor")

# ───────────────────────────── Public API ─────────────────────────────

def extract_content(path: str, profile: Dict[str, Any], type_policies: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a dict:
    {
      "extraction_status": "ok" | "skipped" | "metadata_only" | "error",
      "skipped_reason": str|None,
      "text": str,
      "meta": {...}
    }
    """
    try:
        if not os.path.exists(path):
            return _err("error", f"file not found: {path}")

        size = os.path.getsize(path)
        meta: Dict[str, Any] = {"ext": ext_lower(path), "engine": "none", "cascade": profile.get("parse_pdf_order", [])}

        # Type policies
        ext = meta["ext"]
        tp = type_policies or {}
        skip_exts = set((_norm_list(tp.get("skip"))) or [])
        if ext in skip_exts:
            return _ok("", meta, status="skipped", reason=f"type policy skip: {ext}")

        # profile size gate
        max_mb = float(profile.get("max_file_mb", 100))
        if size > max_mb * 1024 * 1024:
            return _ok("", meta, status="metadata_only", reason=f"file>{max_mb}MB")

        prefer_ocr = set((_norm_list(tp.get("prefer_ocr"))) or [])
        try_text_then_meta = set((_norm_list(tp.get("try_text_then_metadata_only"))) or [])
        metadata_only = set((_norm_list(tp.get("metadata_only"))) or [])

        text = ""
        if ext in metadata_only:
            return _ok("", meta, status="metadata_only", reason=f"policy: metadata_only for {ext}")

        if ext in (".txt", ".csv", ".tsv", ".log", ".md"):
            text = _read_text(path)
            return _ok(text, {**meta, "engine": "text"})

        if ext in (".pdf",):
            cascade = profile.get("parse_pdf_order", ["pymupdf", "pdfminer", "pdfplumber", "ocr"])
            text, engine = _extract_pdf(path, cascade, profile)
            return _ok(text, {**meta, "engine": engine})

        if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            if ext in prefer_ocr or profile.get("ocr", True):
                txt = _ocr_image(path, profile)
                return _ok(txt, {**meta, "engine": "ocr"})
            return _ok("", {**meta, "engine": "none"}, status="metadata_only", reason="image no OCR by policy")

        if ext in (".docx",):
            txt = _read_docx(path)
            return _ok(txt, {**meta, "engine": "docx"})

        if ext in (".pptx",):
            txt = _read_pptx(path, profile)
            if not txt and ext in try_text_then_meta:
                return _ok("", {**meta, "engine": "pptx"}, status="metadata_only", reason="empty after text")
            return _ok(txt, {**meta, "engine": "pptx"})

        if ext in (".xlsx", ".xlsm"):
            txt = _read_xlsx(path, profile)
            if not txt and ext in try_text_then_meta:
                return _ok("", {**meta, "engine": "xlsx"}, status="metadata_only", reason="empty after text")
            return _ok(txt, {**meta, "engine": "xlsx"})

        # default: try reading as text; else metadata_only
        try:
            txt = _read_text(path)
            if txt:
                return _ok(txt, {**meta, "engine": "text"})
        except Exception:
            pass
        return _ok("", meta, status="metadata_only", reason=f"unhandled ext {ext}")

    except Exception as e:
        log.exception("extract_content failed for %s", path)
        return _err("error", f"{e!r}")

# ───────────────────────────── Helpers ─────────────────────────────

def _norm_list(v: Any) -> List[str]:
    if not v:
        return []
    return [str(x).lower() for x in v]

def _ok(text: str, meta: Dict[str, Any], status: str = "ok", reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "extraction_status": status,
        "skipped_reason": reason,
        "text": text or "",
        "meta": meta,
    }

def _err(status: str, reason: str) -> Dict[str, Any]:
    return {
        "extraction_status": status,
        "skipped_reason": reason,
        "text": "",
        "meta": {"engine": "none"},
    }

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# ───────────────────────────── PDF extractors ─────────────────────────────

def _extract_pdf(path: str, cascade: List[str], profile: Dict[str, Any]) -> Tuple[str, str]:
    last_error = None
    for engine in cascade:
        try:
            if engine == "pymupdf":
                try:
                    import fitz  # PyMuPDF
                except Exception:
                    raise RuntimeError("pymupdf not available")
                txt = []
                with fitz.open(path) as doc:
                    for page in doc:
                        txt.append(page.get_text("text"))
                t = "\n".join(txt).strip()
                if t:
                    return t, "pymupdf"
            elif engine == "pdfminer":
                try:
                    from pdfminer.high_level import extract_text  # type: ignore
                except Exception:
                    raise RuntimeError("pdfminer not available")
                t = extract_text(path) or ""
                if t.strip():
                    return t, "pdfminer"
            elif engine == "pdfplumber":
                try:
                    import pdfplumber  # type: ignore
                except Exception:
                    raise RuntimeError("pdfplumber not available")
                txt = []
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        try:
                            txt.append(page.extract_text() or "")
                        except Exception:
                            continue
                t = "\n".join(txt).strip()
                if t:
                    return t, "pdfplumber"
            elif engine == "ocr":
                t = _ocr_pdf(path, profile)
                if t.strip():
                    return t, "ocr"
        except Exception as e:
            last_error = e
            continue
    if last_error:
        log.warning("PDF cascade failed for %s: %r", path, last_error)
    return "", cascade[-1] if cascade else "none"

# ───────────────────────────── OCR helpers ─────────────────────────────

def _ocr_pdf(path: str, profile: Dict[str, Any]) -> str:
    # Requires: pdf2image, pytesseract, PIL
    try:
        from pdf2image import convert_from_path  # type: ignore
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        log.warning("OCR PDF requested but dependencies missing (pdf2image/pytesseract/PIL).")
        return ""
    dpi = int(profile.get("ocr_dpi", 144))
    max_pages = int(profile.get("ocr_max_pages", 10))
    secs = int(profile.get("timebox_ocr_s", 30))
    pages = convert_from_path(path, dpi=dpi, fmt="png")
    out = []
    start = time.time()
    for i, im in enumerate(pages):
        if i >= max_pages:
            break
        if time.time() - start > secs:
            log.warning("OCR timeboxed at %ss for %s", secs, path)
            break
        try:
            out.append(pytesseract.image_to_string(im))
        except Exception:
            continue
    return "\n".join(out).strip()

def _ocr_image(path: str, profile: Dict[str, Any]) -> str:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        log.warning("OCR image requested but PIL/pytesseract missing.")
        return ""
    try:
        im = Image.open(path)
        return pytesseract.image_to_string(im)
    except Exception:
        return ""

# ───────────────────────────── Office formats ─────────────────────────────

def _read_docx(path: str) -> str:
    try:
        import docx2txt  # type: ignore
        return docx2txt.process(path) or ""
    except Exception:
        # very small fallback
        return ""

def _read_pptx(path: str, profile: Dict[str, Any]) -> str:
    try:
        from pptx import Presentation  # type: ignore
    except Exception:
        return ""
    txt = []
    try:
        prs = Presentation(path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    txt.append(shape.text)
    except Exception:
        pass
    return "\n".join(txt).strip()

def _read_xlsx(path: str, profile: Dict[str, Any]) -> str:
    try:
        import openpyxl  # type: ignore
    except Exception:
        return ""
    out: List[str] = []
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                vals = [str(c) if c is not None else "" for c in row]
                if any(vals):
                    out.append("\t".join(vals))
    except Exception:
        pass
    return "\n".join(out).strip()