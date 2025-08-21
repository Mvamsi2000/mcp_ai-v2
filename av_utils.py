# mcp_ai/av_utils.py
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import logging
log = logging.getLogger("av_utils")

# ──────────────────────────────────────────────────────────────────────────────
# Optional deps (both are optional; we degrade gracefully)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:  # pragma: no cover
    WhisperModel = None  # type: ignore

try:
    import whisper as openai_whisper  # type: ignore
except Exception:  # pragma: no cover
    openai_whisper = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# FFmpeg / probing
# ──────────────────────────────────────────────────────────────────────────────
def _has_tool(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def _have_ffmpeg() -> bool:
    return _has_tool("ffmpeg") and _has_tool("ffprobe")

def _ffprobe_json(path: str) -> Dict[str, Any]:
    """Run ffprobe; return parsed JSON or {} (never raises)."""
    if not _has_tool("ffprobe"):
        return {"error": "ffprobe not on PATH"}
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
        if p.returncode != 0:
            return {"error": (p.stderr or "")[:800]}
        return json.loads(p.stdout or "{}") or {}
    except Exception as e:
        return {"error": f"ffprobe exception: {e!r}"}

def _extract_wav_16k_mono(
    src: str,
    *,
    out_wav: str,
    start_at_seconds: int = 0,
    max_seconds: int = 0,
) -> Tuple[bool, Optional[str]]:
    """
    Demux/re-encode to mono 16k PCM WAV. Returns (ok, error).
    Optionally trims with -ss and -t to timebox long media.
    """
    if not _has_tool("ffmpeg"):
        return False, "ffmpeg not on PATH"
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y",
    ]
    if start_at_seconds and start_at_seconds > 0:
        cmd += ["-ss", str(int(start_at_seconds))]
    cmd += ["-i", src, "-ac", "1", "-ar", "16000", "-vn"]
    if max_seconds and max_seconds > 0:
        cmd += ["-t", str(int(max_seconds))]
    cmd += [out_wav]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=1800)
        if p.returncode != 0:
            return False, (p.stderr or p.stdout or "")[:800]
        if not os.path.exists(out_wav) or os.path.getsize(out_wav) == 0:
            return False, "ffmpeg produced empty or missing wav"
        return True, None
    except Exception as e:
        return False, f"ffmpeg exception: {e!r}"

def _remux_to_m4a(src: str, *, start_at_seconds: int = 0, max_seconds: int = 0) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to remux the first audio stream to M4A (copy). If that fails, re-encode AAC.
    Returns (m4a_path, error).
    """
    if not _has_tool("ffmpeg"):
        return None, "ffmpeg not on PATH"

    def _try(cmd: List[str]) -> Tuple[Optional[str], Optional[str]]:
        out = tempfile.NamedTemporaryFile(prefix="mcpai_", suffix=".m4a", delete=False).name
        full = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        if start_at_seconds and start_at_seconds > 0:
            full += ["-ss", str(int(start_at_seconds))]
        full += ["-i", src, "-vn", "-map", "0:a:0"] + cmd
        if max_seconds and max_seconds > 0:
            full += ["-t", str(int(max_seconds))]
        full.append(out)
        p = subprocess.run(full, capture_output=True, text=True, check=False, timeout=1800)
        if p.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0:
            return out, None
        try:
            os.unlink(out)
        except Exception:
            pass
        return None, (p.stderr or p.stdout or "")[:800]

    # 1) Copy
    out1, err1 = _try(["-c:a", "copy"])
    if out1:
        return out1, None
    # 2) Re-encode
    out2, err2 = _try(["-c:a", "aac", "-b:a", "192k"])
    if out2:
        return out2, None
    return None, f"remux/re-encode failed: {err1 or ''} {err2 or ''}".strip()

def _segment_wav(src_wav: str, chunk_seconds: int) -> Tuple[List[str], Optional[str]]:
    """Split WAV into fixed chunks with ffmpeg segmenter. Returns (chunks, err)."""
    if chunk_seconds <= 0 or not _has_tool("ffmpeg"):
        return [src_wav], None
    segdir = tempfile.mkdtemp(prefix="mcpai_seg_")
    pat = os.path.join(segdir, "chunk_%06d.wav")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-i", src_wav, "-f", "segment", "-segment_time", str(chunk_seconds),
        "-c", "copy", pat
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=3600)
        if p.returncode != 0:
            shutil.rmtree(segdir, ignore_errors=True)
            return [src_wav], (p.stderr or p.stdout or "")[:800]
        chunks = sorted(
            os.path.join(segdir, f) for f in os.listdir(segdir)
            if f.startswith("chunk_") and f.endswith(".wav")
        )
        return (chunks if chunks else [src_wav]), None
    except Exception as e:
        try:
            shutil.rmtree(segdir, ignore_errors=True)
        except Exception:
            pass
        return [src_wav], f"segment exception: {e!r}"


# ──────────────────────────────────────────────────────────────────────────────
# ASR config helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get_asr_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge ASR knobs from either:
      - ai.asr
      - ai.local.faster_whisper / ai.local.openai_whisper
    ai.asr takes precedence if present.
    """
    ai = (cfg.get("ai") or {})
    asr = (ai.get("asr") or {}).copy()

    # Back-compat: pull defaults from ai.local.faster_whisper / openai_whisper
    local = (ai.get("local") or {})
    fw = (local.get("faster_whisper") or {})
    ow = (local.get("openai_whisper") or {})

    # Fill in defaults only if not set in ai.asr
    asr.setdefault("engine_order", ["faster_whisper", "whisper", "demo"])
    asr.setdefault("model", fw.get("model", "base"))
    asr.setdefault("device", fw.get("device", "cpu"))
    asr.setdefault("compute_type", fw.get("compute_type", "int8"))
    asr.setdefault("vad_filter", fw.get("vad_filter", True))
    asr.setdefault("beam_size", fw.get("beam_size", 1))
    asr.setdefault("language", fw.get("language"))  # may be None

    # OpenAI Whisper specific (optional)
    asr.setdefault("whisper_model", ow.get("model", "base"))
    asr.setdefault("whisper_device", ow.get("device", "cpu"))

    return asr


# ──────────────────────────────────────────────────────────────────────────────
# ASR engines (faster-whisper + openai/whisper)
# ──────────────────────────────────────────────────────────────────────────────
_FW_CACHE: Dict[str, Any] = {}  # memoized model per (model,device,compute_type)

def _get_fw_model(asr: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    if WhisperModel is None:
        return None, "faster-whisper not installed"
    model_name = str(asr.get("model", "base"))
    device = str(asr.get("device", "cpu"))
    compute_type = str(asr.get("compute_type", "int8"))
    key = f"{model_name}::{device}::{compute_type}"
    if _FW_CACHE.get("key") != key or not _FW_CACHE.get("model"):
        try:
            _FW_CACHE["model"] = WhisperModel(model_name, device=device, compute_type=compute_type)
            _FW_CACHE["key"] = key
        except Exception as e:
            return None, f"WhisperModel load failed: {e!r}"
    return _FW_CACHE["model"], None

def _fw_transcribe(src_path: str, asr: Dict[str, Any], *, language_hint: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Transcribe with faster-whisper; return (text, meta)."""
    meta: Dict[str, Any] = {"engine_detail": "faster_whisper"}
    model, err = _get_fw_model(asr)
    if err or not model:
        meta["error"] = err or "no model"
        return "", meta

    vad_filter = bool(asr.get("vad_filter", True))
    beam_size = int(asr.get("beam_size", 1))
    language = language_hint or asr.get("language")
    t0 = time.time()
    try:
        segments, info = model.transcribe(
            src_path,
            vad_filter=vad_filter,
            beam_size=beam_size,
            language=(language if (isinstance(language, str) and language.strip()) else None),
        )
        text = "".join(getattr(s, "text", "") for s in segments if getattr(s, "text", None)).strip()
        meta["language"] = getattr(info, "language", None)
        # info.duration may be None on some builds
        try:
            meta["duration"] = float(getattr(info, "duration", 0.0) or 0.0)
        except Exception:
            meta["duration"] = None
        meta["elapsed_s"] = round(time.time() - t0, 3)
        return text, meta
    except Exception as e:
        meta["error"] = f"faster_whisper error: {e!r}"
        meta["elapsed_s"] = round(time.time() - t0, 3)
        return "", meta

def _openai_whisper_transcribe(src_path: str, asr: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Transcribe with openai/whisper (local pip)."""
    meta: Dict[str, Any] = {"engine_detail": "openai_whisper"}
    if openai_whisper is None:
        meta["error"] = "openai-whisper not installed"
        return "", meta
    model_name = str(asr.get("whisper_model", "base"))
    device = str(asr.get("whisper_device", "cpu"))
    t0 = time.time()
    try:
        m = openai_whisper.load_model(model_name, device=device)
        res = m.transcribe(src_path, verbose=False)
        txt = (res or {}).get("text") or ""
        meta["language"] = (res or {}).get("language")
        meta["elapsed_s"] = round(time.time() - t0, 3)
        return txt.strip(), meta
    except Exception as e:
        meta["error"] = f"openai-whisper error: {e!r}"
        meta["elapsed_s"] = round(time.time() - t0, 3)
        return "", meta


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _truncate_errors(errs: List[str], max_len: int = 400) -> List[str]:
    out: List[str] = []
    for e in errs:
        if isinstance(e, str) and len(e) > max_len:
            out.append(e[-max_len:])
        else:
            out.append(e)
    return out

def _is_video_ext(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

def _probe_duration_sec(probe: Dict[str, Any]) -> Optional[float]:
    try:
        return float((probe.get("format") or {}).get("duration") or 0.0)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Public API — transcribe_av
# ──────────────────────────────────────────────────────────────────────────────
def transcribe_av(path: str, profile: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort A/V transcription with strong fallbacks:
      1) ffprobe (diagnostic) and optional timeboxing (av_start_at_seconds / av_max_seconds)
      2) ffmpeg → mono 16k WAV
      3) ASR engine order (faster_whisper → openai/whisper → demo)
      4) Optional chunking for long media (av_chunk_seconds)
      5) Remux-to-M4A fallback if decoding is flaky

    Returns: { extraction_status, skipped_reason, text, meta }
    """
    t0 = time.time()
    asr = _get_asr_cfg(cfg)

    # Profile knobs
    order = list(profile.get("av_transcribe_order") or asr.get("engine_order") or ["faster_whisper", "whisper", "demo"])
    chunk_seconds = int(profile.get("av_chunk_seconds", 0) or 0)       # 0 = no chunking
    start_at_seconds = int(profile.get("av_start_at_seconds", 0) or 0) # optional seek
    max_seconds = int(profile.get("av_max_seconds", 0) or 0)           # 0 = full duration

    attempts: List[Dict[str, Any]] = []
    errors: List[str] = []
    best_language: Optional[str] = None

    # Probe is diagnostic only (we still try extraction regardless)
    probe = _ffprobe_json(path)
    audio_streams = [s for s in (probe.get("streams") or []) if s.get("codec_type") == "audio"]
    src_duration = _probe_duration_sec(probe)

    # Extract normalized WAV (timeboxed if configured)
    tmpdir = tempfile.mkdtemp(prefix="mcpai_av_")
    wav_path = os.path.join(tmpdir, "audio.16k.wav")
    ok_wav, wav_err = _extract_wav_16k_mono(
        path,
        out_wav=wav_path,
        start_at_seconds=start_at_seconds,
        max_seconds=max_seconds,
    )
    attempts.append({"step": "ffmpeg_wav", "ok": ok_wav})
    if not ok_wav and wav_err:
        errors.append(wav_err)

    # Build candidate inputs for each engine
    def _engine_inputs(engine: str) -> List[str]:
        # WAV is preferred for stability; fall back to original source path
        cands: List[str] = []
        if ok_wav:
            cands.append(wav_path)
        cands.append(path)
        return cands

    transcript_parts: List[str] = []
    engine_used: Optional[str] = None

    for eng in order:
        inputs = _engine_inputs(eng)
        engine_ok = False

        for inp in inputs:
            # Special fallback: if WAV failed and we're on the original path, try M4A remux
            m4a_path: Optional[str] = None
            if (not ok_wav) and inp == path and eng in ("faster_whisper", "whisper"):
                m4a_path, remux_err = _remux_to_m4a(path, start_at_seconds=start_at_seconds, max_seconds=max_seconds)
                if remux_err:
                    errors.append(remux_err)
                if m4a_path:
                    inputs = [m4a_path] + inputs  # prioritize remuxed audio

            try:
                if eng == "faster_whisper":
                    # Optional chunking (WAV only)
                    chunks, seg_err = _segment_wav(inp, chunk_seconds) if (inp == wav_path and chunk_seconds > 0) else ([inp], None)
                    if seg_err:
                        errors.append(seg_err)
                    seg_texts: List[str] = []
                    for ch in chunks:
                        txt, meta_fw = _fw_transcribe(ch, asr)
                        attempts.append({"step": "faster_whisper", "input": os.path.basename(ch), "elapsed_s": meta_fw.get("elapsed_s"), "error": meta_fw.get("error")})
                        if meta_fw.get("language") and not best_language:
                            best_language = meta_fw.get("language")
                        if meta_fw.get("error"):
                            errors.append(meta_fw["error"])
                        if txt:
                            seg_texts.append(txt)
                    text = " ".join(seg_texts).strip()
                    if text:
                        transcript_parts.append(text)
                        engine_used = "faster_whisper"
                        engine_ok = True
                        # cleanup remux if created
                        if m4a_path:
                            try: os.unlink(m4a_path)
                            except Exception: pass
                        break

                elif eng == "whisper":
                    txt, meta_w = _openai_whisper_transcribe(inp, asr)
                    attempts.append({"step": "openai_whisper", "input": os.path.basename(inp), "elapsed_s": meta_w.get("elapsed_s"), "error": meta_w.get("error")})
                    if meta_w.get("language") and not best_language:
                        best_language = meta_w.get("language")
                    if meta_w.get("error"):
                        errors.append(meta_w["error"])
                    if txt:
                        transcript_parts.append(txt)
                        engine_used = "openai_whisper"
                        engine_ok = True
                        if m4a_path:
                            try: os.unlink(m4a_path)
                            except Exception: pass
                        break

                elif eng == "demo":
                    attempts.append({"step": "demo"})
                    engine_used = "demo"
                    engine_ok = True
                    if m4a_path:
                        try: os.unlink(m4a_path)
                        except Exception: pass
                    break

                else:
                    attempts.append({"step": f"unknown_engine:{eng}"})

            finally:
                # cleanup m4a if we didn’t already
                if m4a_path:
                    try: os.unlink(m4a_path)
                    except Exception: pass

        if engine_ok and engine_used:
            break

    # Cleanup temp artifacts
    try:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        os.rmdir(tmpdir)
    except Exception:
        pass

    # Compose response meta
    meta: Dict[str, Any] = {
        "ext": os.path.splitext(path)[1].lower(),
        "engine": engine_used or "none",
        "attempts": attempts,
        "errors": _truncate_errors(errors),
        "ffmpeg_present": _have_ffmpeg(),
        "av_transcribe_order": order,
        "probe": {
            "format": probe.get("format") or {},
            "streams": [
                {
                    "index": s.get("index"),
                    "codec_name": s.get("codec_name"),
                    "channels": s.get("channels"),
                    "sample_rate": s.get("sample_rate"),
                    "duration": s.get("duration"),
                    "tags": s.get("tags"),
                } for s in audio_streams
            ],
        },
        "kind": "video" if _is_video_ext(path) else "audio",
        "elapsed_s": round(time.time() - t0, 3),
    }

    if best_language:
        meta["language"] = best_language
    if src_duration is not None and src_duration > 0:
        meta["source_duration"] = src_duration
    if start_at_seconds:
        meta["start_at_seconds"] = start_at_seconds
    if max_seconds:
        meta["max_seconds"] = max_seconds
    if chunk_seconds:
        meta["chunk_seconds"] = chunk_seconds

    text = " ".join(p for p in transcript_parts if p).strip()

    # Success
    if text:
        return {
            "extraction_status": "ok",
            "skipped_reason": None,
            "text": text,
            "meta": meta,
        }

    # No audio present → metadata_only, explicit reason
    if not audio_streams:
        return {
            "extraction_status": "metadata_only",
            "skipped_reason": "no audio stream",
            "text": "",
            "meta": meta,
        }

    # Demo explicitly returns ok + empty text (keeps your pipeline behavior)
    if engine_used == "demo":
        return {
            "extraction_status": "ok",
            "skipped_reason": None,
            "text": "",
            "meta": meta,
        }

    # Tried but got nothing
    return {
        "extraction_status": "metadata_only",
        "skipped_reason": "asr_empty_or_failed",
        "text": "",
        "meta": meta,
    }