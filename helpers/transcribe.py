"""Transcribe a video with Whisper (local faster-whisper, OpenAI cloud, or
the reference openai-whisper package). Optional speaker diarization via
pyannote.audio.

Output is a single JSON file at <edit_dir>/transcripts/<video_stem>.json
matching the schema the rest of the video-use pipeline expects:

    {
      "words": [
        {"type": "word",    "text": "Hello", "start": 0.10, "end": 0.45,
         "speaker_id": "speaker_0"},
        {"type": "spacing", "text": " ",     "start": 0.45, "end": 0.52},
        {"type": "word",    "text": "world", "start": 0.52, "end": 0.91,
         "speaker_id": "speaker_0"}
      ],
      "language": "en",
      "language_probability": 0.99,
      "backend": "faster-whisper",
      "model": "large-v3"
    }

Entries of type "spacing" carry the inter-word silence gaps that
`pack_transcripts.py` and `timeline_view.py` use to find cut candidates.
Audio events (laughter, applause) are NOT detected by Whisper — that was
a Scribe-only feature and is the documented tradeoff for going local.

Cached: if the output JSON already exists, it is returned immediately.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --num-speakers 2
    python helpers/transcribe.py <video_path> --backend faster-whisper --model large-v3
    python helpers/transcribe.py <video_path> --backend openai
    python helpers/transcribe.py <video_path> --backend whisper --model medium
    python helpers/transcribe.py <video_path> --no-diarize
    python helpers/transcribe.py <video_path> --device cuda --compute-type float16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Force UTF-8 stdio so unicode prints don't crash the script on Windows
# where the default locale is cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Config / env loading
# ---------------------------------------------------------------------------

DEFAULT_BACKEND = "faster-whisper"
DEFAULT_MODEL = {
    "faster-whisper": "large-v3",
    "openai": "whisper-1",
    "whisper": "large-v3",
}
DEFAULT_DEVICE = "auto"
DEFAULT_COMPUTE_TYPE = "auto"

# Environment keys the loader recognizes (read from .env or process env).
ENV_KEYS = (
    "OPENAI_API_KEY",
    "HUGGINGFACE_TOKEN",
    "HF_TOKEN",
    "WHISPER_BACKEND",
    "WHISPER_MODEL",
    "WHISPER_DEVICE",
    "WHISPER_COMPUTE_TYPE",
)


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_env() -> dict[str, str]:
    """Merge env vars from the repo .env (preferred), the cwd .env, and the
    actual process environment. Process env wins for already-exported vars.
    """
    repo_env = Path(__file__).resolve().parent.parent / ".env"
    cwd_env = Path(".env").resolve()

    merged: dict[str, str] = {}
    merged.update(_parse_env_file(repo_env))
    if cwd_env != repo_env:
        merged.update(_parse_env_file(cwd_env))
    for k in ENV_KEYS:
        v = os.environ.get(k)
        if v:
            merged[k] = v
    # Normalize HF_TOKEN <-> HUGGINGFACE_TOKEN
    if "HUGGINGFACE_TOKEN" not in merged and "HF_TOKEN" in merged:
        merged["HUGGINGFACE_TOKEN"] = merged["HF_TOKEN"]
    return merged


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------


def extract_audio(video_path: Path, dest: Path) -> None:
    """Extract mono 16 kHz PCM WAV — the canonical input for Whisper models."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Word -> Scribe-compatible schema conversion
# ---------------------------------------------------------------------------

# Minimum gap (seconds) between consecutive words to materialize a "spacing"
# entry. Anything below this is just normal speech rate, not a cut candidate.
SPACING_MIN_GAP = 0.02


def _build_word_entries(
    words: list[dict],
    diarization: list[tuple[float, float, str]] | None = None,
    default_speaker: str = "speaker_0",
) -> list[dict]:
    """Take a flat list of {text, start, end} word dicts, attach speaker
    labels, and interleave 'spacing' entries between consecutive words so
    the downstream tools see the same shape they got from Scribe.
    """
    entries: list[dict] = []
    prev_end: float | None = None

    for w in words:
        text = (w.get("text") or "").strip()
        start = w.get("start")
        end = w.get("end")
        if not text or start is None or end is None:
            continue
        # Whisper sometimes returns end <= start on very short tokens; clamp.
        if end < start:
            end = start
        speaker = _assign_speaker(diarization, start, end, default_speaker)

        if prev_end is not None:
            gap = max(0.0, start - prev_end)
            if gap >= SPACING_MIN_GAP:
                entries.append({
                    "type": "spacing",
                    "text": " ",
                    "start": prev_end,
                    "end": start,
                })
        entries.append({
            "type": "word",
            "text": text,
            "start": start,
            "end": end,
            "speaker_id": speaker,
        })
        prev_end = end

    return entries


def _assign_speaker(
    diarization: list[tuple[float, float, str]] | None,
    start: float,
    end: float,
    default_speaker: str,
) -> str:
    """Find the diarization segment with the largest temporal overlap with
    [start, end]. Falls back to default_speaker if no overlap.
    """
    if not diarization:
        return default_speaker
    best_overlap = 0.0
    best_speaker = default_speaker
    for seg_start, seg_end, label in diarization:
        overlap = max(0.0, min(end, seg_end) - max(start, seg_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = label
    return best_speaker


# ---------------------------------------------------------------------------
# Backend: faster-whisper (default; local, fast, GPU-capable)
# ---------------------------------------------------------------------------


def transcribe_faster_whisper(
    audio_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    verbose: bool,
) -> tuple[list[dict], dict]:
    """Returns (flat word list, meta dict with language/probability)."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "faster-whisper is not installed. Install with:\n"
            "    pip install faster-whisper\n"
            "or pick a different --backend (openai or whisper)."
        ) from e

    if device == "auto":
        device = _autodetect_device()
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    if verbose:
        print(f"  loading faster-whisper model={model_name} device={device} "
              f"compute_type={compute_type}", flush=True)

    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    if verbose:
        print("  decoding (word-level timestamps, vad filter on)…", flush=True)

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        beam_size=5,
        condition_on_previous_text=False,
    )

    words: list[dict] = []
    for seg in segments_iter:
        for w in (seg.words or []):
            text = (w.word or "").strip()
            if not text:
                continue
            words.append({
                "text": text,
                "start": float(w.start) if w.start is not None else None,
                "end": float(w.end) if w.end is not None else None,
            })

    meta = {
        "language": info.language,
        "language_probability": float(info.language_probability or 0.0),
        "duration": float(info.duration or 0.0),
    }
    return words, meta


def _autodetect_device() -> str:
    """Best-effort CUDA detection for faster-whisper. CPU otherwise."""
    try:
        import ctranslate2  # type: ignore
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Backend: openai (cloud Whisper API; whisper-1 with word timestamps)
# ---------------------------------------------------------------------------


def transcribe_openai_api(
    audio_path: Path,
    model_name: str,
    api_key: str,
    language: str | None,
    verbose: bool,
) -> tuple[list[dict], dict]:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "The openai package is not installed. Install with:\n"
            "    pip install openai\n"
            "or pick a different --backend."
        ) from e

    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is required for --backend openai. "
            "Add it to .env or export it."
        )

    if verbose:
        print(f"  uploading to OpenAI Whisper ({model_name}) …", flush=True)

    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model_name,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language=language,
        )

    # The OpenAI SDK returns a model object; convert to dict.
    payload = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

    words: list[dict] = []
    for w in payload.get("words", []) or []:
        text = (w.get("word") or "").strip()
        if not text:
            continue
        words.append({
            "text": text,
            "start": float(w["start"]) if w.get("start") is not None else None,
            "end": float(w["end"]) if w.get("end") is not None else None,
        })

    meta = {
        "language": payload.get("language"),
        "language_probability": None,
        "duration": float(payload.get("duration") or 0.0),
    }
    return words, meta


# ---------------------------------------------------------------------------
# Backend: openai-whisper (the original reference package; CPU-friendly fallback)
# ---------------------------------------------------------------------------


def transcribe_openai_whisper(
    audio_path: Path,
    model_name: str,
    device: str,
    language: str | None,
    verbose: bool,
) -> tuple[list[dict], dict]:
    try:
        import whisper  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "openai-whisper is not installed. Install with:\n"
            "    pip install openai-whisper\n"
            "or pick a different --backend."
        ) from e

    if device == "auto":
        device = _autodetect_device()

    if verbose:
        print(f"  loading whisper model={model_name} device={device}", flush=True)

    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        verbose=False,
        condition_on_previous_text=False,
    )

    words: list[dict] = []
    for seg in result.get("segments", []) or []:
        for w in seg.get("words", []) or []:
            text = (w.get("word") or "").strip()
            if not text:
                continue
            words.append({
                "text": text,
                "start": float(w["start"]) if w.get("start") is not None else None,
                "end": float(w["end"]) if w.get("end") is not None else None,
            })

    meta = {
        "language": result.get("language"),
        "language_probability": None,
        "duration": None,
    }
    return words, meta


# ---------------------------------------------------------------------------
# Optional speaker diarization (pyannote.audio)
# ---------------------------------------------------------------------------


def diarize_pyannote(
    audio_path: Path,
    hf_token: str,
    num_speakers: int | None,
    device: str,
    verbose: bool,
) -> list[tuple[float, float, str]] | None:
    """Run pyannote/speaker-diarization-3.1 on the audio and return a list of
    (start, end, "speaker_N") tuples normalized to Scribe's label format.

    Returns None (and logs a one-line warning) if pyannote isn't installed,
    no HF token, model isn't accessible, or any runtime error. Diarization
    is ALWAYS optional — the pipeline works with a single default speaker.
    """
    if not hf_token:
        if verbose:
            print("  diarization skipped: no HUGGINGFACE_TOKEN", flush=True)
        return None
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError:
        if verbose:
            print("  diarization skipped: pyannote.audio not installed "
                  "(pip install pyannote.audio)", flush=True)
        return None

    try:
        if verbose:
            print("  loading pyannote/speaker-diarization-3.1 …", flush=True)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except Exception as e:
        print(f"  diarization skipped: failed to load pyannote pipeline ({e})",
              flush=True)
        return None

    # Move to GPU if available.
    if device == "auto":
        device = _autodetect_device()
    if device == "cuda":
        try:
            import torch  # type: ignore
            pipeline.to(torch.device("cuda"))
        except Exception:
            pass

    kwargs: dict = {}
    if num_speakers and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers

    try:
        if verbose:
            print("  diarizing …", flush=True)
        diarization = pipeline(str(audio_path), **kwargs)
    except Exception as e:
        print(f"  diarization skipped: pipeline error ({e})", flush=True)
        return None

    # pyannote returns labels like "SPEAKER_00"; normalize to Scribe's
    # "speaker_0" form so the rest of the pipeline (and pack_transcripts'
    # "speaker_" prefix strip) keeps working unchanged.
    label_map: dict[str, str] = {}
    out: list[tuple[float, float, str]] = []
    next_idx = 0
    for turn, _, raw_label in diarization.itertracks(yield_label=True):
        if raw_label not in label_map:
            label_map[raw_label] = f"speaker_{next_idx}"
            next_idx += 1
        out.append((float(turn.start), float(turn.end), label_map[raw_label]))
    out.sort(key=lambda t: t[0])
    if verbose:
        print(f"  diarization: {len(out)} segments, {next_idx} speaker(s)",
              flush=True)
    return out


# ---------------------------------------------------------------------------
# Top-level transcribe_one (cached, backend-agnostic)
# ---------------------------------------------------------------------------


def transcribe_one(
    video: Path,
    edit_dir: Path,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: str | None = None,
    num_speakers: int | None = None,
    diarize: bool = True,
    env: dict[str, str] | None = None,
    verbose: bool = True,
) -> Path:
    """Transcribe one video. Returns path to the resulting JSON.

    Cached: returns immediately if <edit_dir>/transcripts/<stem>.json exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    env = env or load_env()
    backend = backend or env.get("WHISPER_BACKEND") or DEFAULT_BACKEND
    if backend not in DEFAULT_MODEL:
        raise SystemExit(
            f"unknown backend '{backend}'. "
            f"choose from: {', '.join(DEFAULT_MODEL)}"
        )
    if model is None:
        model = env.get("WHISPER_MODEL") or DEFAULT_MODEL[backend]
    if device == DEFAULT_DEVICE:
        device = env.get("WHISPER_DEVICE") or DEFAULT_DEVICE
    if compute_type == DEFAULT_COMPUTE_TYPE:
        compute_type = env.get("WHISPER_COMPUTE_TYPE") or DEFAULT_COMPUTE_TYPE

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio)
        size_mb = audio.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"  audio: {audio.name} ({size_mb:.1f} MB)", flush=True)

        if backend == "faster-whisper":
            words, meta = transcribe_faster_whisper(
                audio, model, device, compute_type, language, verbose,
            )
        elif backend == "openai":
            words, meta = transcribe_openai_api(
                audio, model, env.get("OPENAI_API_KEY", ""), language, verbose,
            )
        elif backend == "whisper":
            words, meta = transcribe_openai_whisper(
                audio, model, device, language, verbose,
            )
        else:
            raise SystemExit(f"unknown backend '{backend}'")

        diarization: list[tuple[float, float, str]] | None = None
        if diarize:
            diarization = diarize_pyannote(
                audio,
                hf_token=env.get("HUGGINGFACE_TOKEN", ""),
                num_speakers=num_speakers,
                device=device,
                verbose=verbose,
            )

    entries = _build_word_entries(words, diarization)
    payload = {
        "words": entries,
        "language": meta.get("language"),
        "language_probability": meta.get("language_probability"),
        "duration": meta.get("duration"),
        "backend": backend,
        "model": model,
        "diarized": bool(diarization),
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        word_count = sum(1 for e in entries if e["type"] == "word")
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        print(f"    backend={backend} model={model}  words={word_count}  "
              f"diarized={'yes' if diarization else 'no'}")

    return out_path


# ---------------------------------------------------------------------------
# Back-compat shim — transcribe_batch.py used to import load_api_key. The
# new signature is load_env() returning a dict; we expose a thin stub that
# returns the merged env so old call sites keep working without raising.
# ---------------------------------------------------------------------------


def load_api_key() -> dict[str, str]:
    """Deprecated: kept for back-compat with older transcribe_batch.py.
    Returns the full env dict instead of a single string.
    """
    return load_env()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Transcribe a video with Whisper "
                    "(faster-whisper / openai / whisper) + optional pyannote diarization",
    )
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--backend",
        choices=list(DEFAULT_MODEL.keys()),
        default=None,
        help=f"Transcription backend (default: {DEFAULT_BACKEND}). "
             "faster-whisper = local CTranslate2, fastest. "
             "openai = hosted Whisper API. "
             "whisper = reference openai-whisper package.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. faster-whisper/whisper: tiny|base|small|medium|"
             "large-v2|large-v3 (default large-v3). openai: whisper-1.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="auto|cpu|cuda (auto-detects CUDA, falls back to CPU)",
    )
    ap.add_argument(
        "--compute-type",
        type=str,
        default=DEFAULT_COMPUTE_TYPE,
        help="faster-whisper compute type: auto|int8|int8_float16|float16|float32",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers, improves pyannote diarization.",
    )
    diar = ap.add_mutually_exclusive_group()
    diar.add_argument(
        "--diarize",
        dest="diarize",
        action="store_true",
        help="Force speaker diarization (default: on if pyannote + HF token available)",
    )
    diar.add_argument(
        "--no-diarize",
        dest="diarize",
        action="store_false",
        help="Skip diarization entirely; every word gets speaker_0",
    )
    ap.set_defaults(diarize=True)
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        backend=args.backend or DEFAULT_BACKEND,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        num_speakers=args.num_speakers,
        diarize=args.diarize,
    )


if __name__ == "__main__":
    main()
