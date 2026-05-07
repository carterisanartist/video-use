"""Batch-transcribe every video in a directory with parallel workers.

Walks <videos_dir> for common video extensions, runs Whisper on each,
writes transcripts to <videos_dir>/edit/transcripts/<name>.json.

Cached per-file: any source that already has a transcript is skipped.

Usage:
    python helpers/transcribe_batch.py <videos_dir>
    python helpers/transcribe_batch.py <videos_dir> --workers 4
    python helpers/transcribe_batch.py <videos_dir> --num-speakers 2
    python helpers/transcribe_batch.py <videos_dir> --edit-dir /custom/edit
    python helpers/transcribe_batch.py <videos_dir> --backend openai
    python helpers/transcribe_batch.py <videos_dir> --backend faster-whisper --model medium
    python helpers/transcribe_batch.py <videos_dir> --no-diarize

Worker concurrency notes:
- Local backends (faster-whisper / whisper) load the model into RAM/VRAM
  per worker. Default is 1 to avoid OOM on a single GPU. Bump to 2-4
  only if you have headroom (CPU-only or a big GPU).
- The hosted "openai" backend defaults to 4 workers since the work is
  network-bound, just like the original Scribe pipeline.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Force UTF-8 stdio so unicode prints don't crash the script on Windows
# where the default locale is cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from transcribe import (
    DEFAULT_BACKEND,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    load_env,
    transcribe_one,
)


VIDEO_EXTS = {
    ".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV",
    ".avi", ".AVI", ".m4v", ".M4V", ".webm", ".WEBM",
}


def find_videos(videos_dir: Path) -> list[Path]:
    return sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTS
    )


def _default_workers(backend: str) -> int:
    return 4 if backend == "openai" else 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parallel batch transcription of a videos directory (Whisper)",
    )
    ap.add_argument("videos_dir", type=Path, help="Directory containing source videos")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <videos_dir>/edit)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers. Default: 1 for local backends, 4 for openai.",
    )
    ap.add_argument(
        "--backend",
        choices=list(DEFAULT_MODEL.keys()),
        default=None,
        help=f"Transcription backend (default: {DEFAULT_BACKEND}).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default depends on backend).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="auto|cpu|cuda for local backends",
    )
    ap.add_argument(
        "--compute-type",
        type=str,
        default=DEFAULT_COMPUTE_TYPE,
        help="faster-whisper compute type",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code. Omit to auto-detect per file.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers, improves diarization accuracy.",
    )
    diar = ap.add_mutually_exclusive_group()
    diar.add_argument("--diarize", dest="diarize", action="store_true")
    diar.add_argument("--no-diarize", dest="diarize", action="store_false")
    ap.set_defaults(diarize=True)

    args = ap.parse_args()

    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        sys.exit(f"not a directory: {videos_dir}")

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
    (edit_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    videos = find_videos(videos_dir)
    if not videos:
        sys.exit(f"no videos found in {videos_dir}")

    already_cached = [v for v in videos if (edit_dir / "transcripts" / f"{v.stem}.json").exists()]
    pending = [v for v in videos if v not in already_cached]

    print(f"found {len(videos)} videos ({len(already_cached)} cached, "
          f"{len(pending)} to transcribe)")
    if not pending:
        print("nothing to do")
        return

    env = load_env()
    backend = args.backend or env.get("WHISPER_BACKEND") or DEFAULT_BACKEND
    workers = args.workers if args.workers is not None else _default_workers(backend)

    print(f"transcribing {len(pending)} files with backend={backend}, workers={workers}")
    if backend != "openai" and workers > 1:
        print("  note: local backends load the model into memory per worker; "
              "watch RAM/VRAM usage")

    t0 = time.time()
    errors: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                transcribe_one,
                video=v,
                edit_dir=edit_dir,
                backend=backend,
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                language=args.language,
                num_speakers=args.num_speakers,
                diarize=args.diarize,
                env=env,
                verbose=False,
            ): v
            for v in pending
        }
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                out = fut.result()
                print(f"  + {v.stem}  ->  {out.name}")
            except Exception as e:
                errors.append((v, str(e)))
                print(f"  x {v.stem}  FAILED: {e}")

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s")
    if errors:
        print(f"{len(errors)} failures:")
        for v, msg in errors:
            print(f"  {v.name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
