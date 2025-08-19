# mcp_ai/av_utils.py
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

# Optional dependencies
try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore

try:
    import whisper as openai_whisper  # type: ignore
except Exception:
    openai_whisper = None  # type: ignore


# ------------------------------ ffmpeg helpers ------------------------------

def _have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        return True
    except Exception:
        return False


def _ffprobe_json(path: str) -> Dict[str, Any]:
    """Run ffprobe -show_streams -show_format; return parsed JSON (or {})."""
    if not _have_ffmpeg():
        return {}
    try:
        cp = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30
        )
        if cp.returncode != 0:
            return {}
        return json.loads(cp.stdout or "{}") or {}
    except Exception:
        return {}


def _probe_audio_streams(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Return (audio_streams, probe_json). audio_streams is a list of stream dicts with codec_type='audio'.
    """
    data = _ffprobe_json(path)
    streams = (data.get("streams") or [])
    audio_streams = [s for s in streams if (s.get("codec_type") == "audio")]
    return audio_streams, data


def _extract_wav_16k_mono(src: str, *, max_seconds: int = 0) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract 16k mono PCM wav for robust ASR. Returns (wav_path, error).
    If max_seconds > 0, trim to that duration.
    """
    if not _have_ffmpeg():
        return None, "ffmpeg not available"
    out = tempfile.NamedTemporaryFile(prefix="mcpai_", suffix=".wav", delete=False).name
    cmd = ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-vn", "-f", "wav"]
    if max_seconds and max_seconds > 0:
        cmd.extend(["-t", str(int(max_seconds))])
    cmd.append(out)
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    if cp.returncode != 0 or (not os.path.exists(out)) or (os.path.getsize(out) == 0):
        try:
            if os.path.exists(out):
                os.unlink(out)
        except Exception:
            pass
        return None, f"ffmpeg wav extract failed: {(cp.stdout or '')[-400:]}"
    return out, None


def _remux_to_m4a(src: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try .m4a copy (no re-encode). If that fails, re-encode to AAC.
    Returns (m4a_path, error).
    """
    if not _have_ffmpeg():
        return None, "ffmpeg not available"

    # 1) Remux (copy)
    out1 = tempfile.NamedTemporaryFile(prefix="mcpai_", suffix=".m4a", delete=False).name
    cmd1 = ["ffmpeg", "-y", "-i", src, "-vn", "-map", "0:a:0", "-c:a", "copy", out1]
    cp1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
    if cp1.returncode == 0 and os.path.exists(out1) and os.path.getsize(out1) > 0:
        return out1, None
    try:
        os.unlink(out1)
    except Exception:
        pass

    # 2) Re-encode to AAC
    out2 = tempfile.NamedTemporaryFile(prefix="mcpai_", suffix=".m4a", delete=False).name
    cmd2 = ["ffmpeg", "-y", "-i", src, "-vn", "-map", "0:a:0", "-c:a", "aac", "-b:a", "192k", out2]
    cp2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    if cp2.returncode == 0 and os.path.exists(out2) and os.path.getsize(out2) > 0:
        return out2, None
    try:
        os.unlink(out2)
    except Exception:
        pass
    return None, f"remux/re-encode failed: {(cp2.stdout or '')[-400:]}"


def _segment_wav(src_wav: str, chunk_seconds: int) -> Tuple[List[str], Optional[str]]:
    """
    Split WAV into fixed chunks using ffmpeg segmenter. Returns (list_of_chunk_paths, error).
    """
    if chunk_seconds <= 0:
        return [src_wav], None
    if not _have_ffmpeg():
        return [src_wav], "ffmpeg not available"
    segdir = tempfile.mkdtemp(prefix="mcpai_seg_")
    pat = os.path.join(segdir, "chunk_%04d.wav")
    cmd = ["ffmpeg", "-y", "-i", src_wav, "-f", "segment", "-segment_time", str(chunk_seconds), "-c", "copy", pat]
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1200)
    if cp.returncode != 0:
        try:
            shutil.rmtree(segdir, ignore_errors=True)
        except Exception:
            pass
        return [src_wav], f"segment failed: {(cp.stdout or '')[-400:]}"
    chunks = sorted([os.path.join(segdir, f) for f in os.listdir(segdir) if f.startswith("chunk_") and f.endswith(".wav")])
    return (chunks if chunks else [src_wav]), None


# ------------------------------ ASR engines ------------------------------

_FW_SESSION: Dict[str, Any] = {}  # simple singleton cache for faster-whisper model


def _get_fw_model(cfg: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str], Dict[str, Any]]:
    """
    Load (or reuse) a faster-whisper model using ai.local.faster_whisper config.
    Returns (model_or_None, error, fw_cfg_dict)
    """
    if WhisperModel is None:
        return None, "faster-whisper not installed", {}

    fw_cfg = (((cfg.get("ai") or {}).get("local") or {}).get("faster_whisper") or {})
    model_name = fw_cfg.get("model") or "base"
    device = fw_cfg.get("device") or "cpu"
    compute_type = fw_cfg.get("compute_type") or "int8"

    key = f"{model_name}::{device}::{compute_type}"
    if _FW_SESSION.get("key") != key or not _FW_SESSION.get("model"):
        try:
            m = WhisperModel(model_name, device=device, compute_type=compute_type)
            _FW_SESSION["key"] = key
            _FW_SESSION["model"] = m
        except Exception as e:
            return None, f"WhisperModel load failed: {e!r}", fw_cfg
    return _FW_SESSION.get("model"), None, fw_cfg


def _fw_transcribe(src: str, cfg: Dict[str, Any], *, language_hint: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Transcribe with faster-whisper; return (text, meta).
    """
    model, err, fw_cfg = _get_fw_model(cfg)
    meta = {"engine_detail": "faster_whisper", "language": None, "duration": None}
    if err or not model:
        meta["error"] = err or "no model"
        return "", meta

    vad_filter = bool(fw_cfg.get("vad_filter", True))
    beam_size = int(fw_cfg.get("beam_size", 1))
    lang = language_hint or fw_cfg.get("language")  # can be None (auto)

    t0 = time.time()
    try:
        segments, info = model.transcribe(
            src,
            vad_filter=vad_filter,
            beam_size=beam_size,
            language=lang if (lang and isinstance(lang, str) and lang.strip()) else None,
        )
        text_parts: List[str] = []
        for seg in segments:
            if seg and getattr(seg, "text", ""):
                text_parts.append(seg.text)
        text = " ".join(t.strip() for t in text_parts if t and t.strip())
        meta["language"] = getattr(info, "language", None)
        dur = getattr(info, "duration", None)
        try:
            meta["duration"] = float(dur) if dur is not None else None
        except Exception:
            meta["duration"] = None
        meta["elapsed_s"] = round(time.time() - t0, 3)
        return text, meta
    except Exception as e:
        meta["error"] = f"faster_whisper error: {e!r}"
        meta["elapsed_s"] = round(time.time() - t0, 3)
        return "", meta


def _openai_whisper_transcribe(src: str, cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Optional fallback with openai/whisper (local). Returns (text, meta).
    """
    meta = {"engine_detail": "openai_whisper"}
    if openai_whisper is None:
        meta["error"] = "openai-whisper not installed"
        return "", meta

    wcfg = (((cfg.get("ai") or {}).get("local") or {}).get("openai_whisper") or {})
    model_name = wcfg.get("model") or "base"
    device = wcfg.get("device") or "cpu"

    t0 = time.time()
    try:
        m = openai_whisper.load_model(model_name, device=device)
        # openai-whisper likes wav/mp3/m4a best; if src is a video, it will invoke ffmpeg internally.
        res = m.transcribe(src, verbose=False)
        txt = (res or {}).get("text") or ""
        meta["language"] = (res or {}).get("language")
        meta["elapsed_s"] = round(time.time() - t0, 3)
        return (txt or "").strip(), meta
    except Exception as e:
        meta["error"] = f"openai-whisper error: {e!r}"
        meta["elapsed_s"] = round(time.time() - t0, 3)
        return "", meta


# ------------------------------ Top-level API ------------------------------

def transcribe_av(path: str, profile: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort A/V transcription with layered fallbacks.
    Returns a dict with:
      - extraction_status: "ok" | "metadata_only"
      - skipped_reason: None or string (e.g., "no audio stream")
      - text: transcript or ""
      - meta: dict with probe, engine, attempts, errors, etc.
    """
    t0 = time.time()
    ext = os.path.splitext(path)[1].lower()

    # Config knobs
    order = (profile.get("av_transcribe_order") or ["faster_whisper", "whisper", "demo"])
    av_chunk_seconds = int(profile.get("av_chunk_seconds", 0) or 0)  # 0 = no chunking
    av_max_seconds = int(profile.get("av_max_seconds", 0) or 0)      # 0 = full duration
    lang_probe_seconds = int(profile.get("av_lang_probe_seconds", 60) or 60)

    meta: Dict[str, Any] = {
        "ext": ext,
        "engine": "none",
        "attempts": [],
        "errors": [],
        "ffmpeg_present": _have_ffmpeg(),
        "av_transcribe_order": list(order),
        "kind": "audio" if ext in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg") else "video"
    }

    # Preflight probe
    audio_streams, probe = _probe_audio_streams(path)
    meta["probe"] = {
        "format": (probe.get("format") or {}),
        "streams": [
            {
                "index": s.get("index"),
                "codec_name": s.get("codec_name"),
                "channels": s.get("channels"),
                "sample_rate": s.get("sample_rate"),
                "duration": s.get("duration"),
                "tags": s.get("tags"),
            } for s in (audio_streams or [])
        ],
    }

    if not audio_streams:
        # No audio present → don’t try to transcribe.
        meta["engine"] = "none"
        return {
            "extraction_status": "metadata_only",
            "skipped_reason": "no audio stream",
            "text": "",
            "meta": _finalize_meta(meta, t0),
        }

    # Compute duration (best effort)
    try:
        duration = float((probe.get("format") or {}).get("duration") or 0.0)
    except Exception:
        duration = 0.0
    meta["duration"] = duration

    # ───────────────────────────── Strategy 1: faster-whisper direct ─────────────────────────────
    if "faster_whisper" in order and WhisperModel is not None:
        meta["attempts"].append("faster_whisper:direct")
        # Optional language hint: we can probe short clip first for auto-language
        language_hint = None
        # If config explicitly forces language, _fw_transcribe will honor it.

        # Try direct transcription
        text, fw_meta = _fw_transcribe(path, cfg)
        meta.update({k: v for k, v in fw_meta.items() if k not in ("elapsed_s",)})
        if text and text.strip():
            meta["engine"] = "faster_whisper"
            meta["elapsed_fw_direct_s"] = fw_meta.get("elapsed_s")
            return {
                "extraction_status": "ok",
                "skipped_reason": None,
                "text": text,
                "meta": _finalize_meta(meta, t0),
            }
        # If direct failed: go WAV normalize
        if fw_meta.get("error"):
            meta["errors"].append(fw_meta["error"])

        # If we didn’t get text, try language probe on a short clip → then full WAV pass
        if not language_hint:
            clip_path, err = _extract_wav_16k_mono(path, max_seconds=lang_probe_seconds if av_max_seconds == 0 else min(av_max_seconds, lang_probe_seconds))
            if clip_path:
                meta["attempts"].append("faster_whisper:lang_probe")
                txt_probe, fw_meta_probe = _fw_transcribe(clip_path, cfg)
                # best-effort hint
                language_hint = fw_meta_probe.get("language")
                meta["lang_probe_language"] = language_hint
                try:
                    os.unlink(clip_path)
                except Exception:
                    pass

        # Extract full (or clipped) WAV and retry
        wav_path, err = _extract_wav_16k_mono(path, max_seconds=av_max_seconds)
        if not wav_path:
            meta["errors"].append(err or "wav extract failed")
        else:
            # Optional chunking
            if av_chunk_seconds and (duration and duration > av_chunk_seconds):
                chunks, seg_err = _segment_wav(wav_path, av_chunk_seconds)
                if seg_err:
                    meta["errors"].append(seg_err)
                meta["attempts"].append(f"segment:{len(chunks)}")
                parts: List[str] = []
                for c in chunks:
                    meta["attempts"].append("faster_whisper:chunk")
                    t_i, fw_m = _fw_transcribe(c, cfg, language_hint=language_hint)
                    if fw_m.get("error"):
                        meta["errors"].append(fw_m["error"])
                    if t_i and t_i.strip():
                        parts.append(t_i.strip())
                # cleanup
                try:
                    for c in chunks:
                        if os.path.exists(c):
                            os.unlink(c)
                    os.rmdir(os.path.dirname(chunks[0]))
                except Exception:
                    pass
                try:
                    if os.path.exists(wav_path):
                        os.unlink(wav_path)
                except Exception:
                    pass

                text2 = " ".join(parts).strip()
                if text2:
                    meta["engine"] = "faster_whisper"
                    return {
                        "extraction_status": "ok",
                        "skipped_reason": None,
                        "text": text2,
                        "meta": _finalize_meta(meta, t0),
                    }
            else:
                meta["attempts"].append("faster_whisper:wav")
                text2, fw_meta2 = _fw_transcribe(wav_path, cfg, language_hint=language_hint)
                if fw_meta2.get("error"):
                    meta["errors"].append(fw_meta2["error"])
                try:
                    if os.path.exists(wav_path):
                        os.unlink(wav_path)
                except Exception:
                    pass
                if text2 and text2.strip():
                    meta["engine"] = "faster_whisper"
                    return {
                        "extraction_status": "ok",
                        "skipped_reason": None,
                        "text": text2,
                        "meta": _finalize_meta(meta, t0),
                    }

        # Remux to m4a and try again (last resort for container weirdness)
        m4a_path, remux_err = _remux_to_m4a(path)
        if m4a_path:
            meta["attempts"].append("faster_whisper:m4a")
            text3, fw_meta3 = _fw_transcribe(m4a_path, cfg)
            if fw_meta3.get("error"):
                meta["errors"].append(fw_meta3["error"])
            try:
                os.unlink(m4a_path)
            except Exception:
                pass
            if text3 and text3.strip():
                meta["engine"] = "faster_whisper"
                return {
                    "extraction_status": "ok",
                    "skipped_reason": None,
                    "text": text3.strip(),
                    "meta": _finalize_meta(meta, t0),
                }
        else:
            meta["errors"].append(remux_err or "remux failed")

    # ───────────────────────────── Strategy 2: openai/whisper (optional) ─────────────────────────────
    if "whisper" in order and openai_whisper is not None:
        meta["attempts"].append("openai_whisper:direct")
        txt_w, w_meta = _openai_whisper_transcribe(path, cfg)
        meta.update({f"whisper_{k}": v for k, v in w_meta.items() if k != "engine_detail"})
        if txt_w and txt_w.strip():
            meta["engine"] = "openai_whisper"
            return {
                "extraction_status": "ok",
                "skipped_reason": None,
                "text": txt_w.strip(),
                "meta": _finalize_meta(meta, t0),
            }

    # ───────────────────────────── Strategy 3: demo (no transcript) ─────────────────────────────
    if "demo" in order:
        # Keep behavior: ok status + engine demo + text_len 0 (so pipeline doesn't crash).
        meta["engine"] = "demo"
        return {
            "extraction_status": "ok",
            "skipped_reason": None,
            "text": "",
            "meta": _finalize_meta(meta, t0),
        }

    # If we got here, we only have metadata
    return {
        "extraction_status": "metadata_only",
        "skipped_reason": "transcription failed or disabled",
        "text": "",
        "meta": _finalize_meta(meta, t0),
    }


def _finalize_meta(meta: Dict[str, Any], t0: float) -> Dict[str, Any]:
    meta = dict(meta or {})
    meta["elapsed_s"] = round(time.time() - t0, 3)
    # Truncate very long error strings defensively
    if "errors" in meta and isinstance(meta["errors"], list):
        trunc = []
        for e in meta["errors"]:
            if isinstance(e, str) and len(e) > 400:
                trunc.append(e[-400:])
            else:
                trunc.append(e)
        meta["errors"] = trunc
    return meta