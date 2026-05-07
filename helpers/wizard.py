"""Interactive wizard for the Whisper fork of video-use.

Runs the same toddler-mode conversation that the SKILL flow uses, but as
a plain Python script for users who don't have the Cursor agent in front
of them. Asks one short, jargon-free question at a time, then prints the
exact one-liners to run (transcribe -> hand-edit EDL -> render).

Usage:
    python helpers/wizard.py
    python helpers/wizard.py --raw /path/to/raw   # skip the "where" question

The wizard never edits the EDL itself — picking the actual cuts is a
human job (or an LLM job downstream). It exists to gather the structural
choices (platform, length, mood, captions) and emit the right CLI calls.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

# UTF-8 stdio so we can print arrows / em-dashes on Windows without crashing.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Toddler-mode prompting primitives
# ---------------------------------------------------------------------------


def ask_choice(question: str, options: list[tuple[str, str]],
               default_key: str | None = None) -> str:
    """Ask a single A/B(/C) question. Returns the chosen key.

    options: list of (key, label). Keys are short ('a', 'b', 'tiktok', ...).
    default_key: highlighted as [default] and accepted when user just hits Enter.
    """
    print()
    print(question)
    keys = [k for k, _ in options]
    for k, label in options:
        marker = " [default]" if k == default_key else ""
        print(f"  {k}) {label}{marker}")
    while True:
        raw = input("> ").strip().lower()
        if not raw and default_key:
            return default_key
        if raw in keys:
            return raw
        print(f"  please type one of: {', '.join(keys)}")


def ask_text(question: str, default: str = "") -> str:
    """Free-form text input with optional default."""
    print()
    print(question)
    if default:
        print(f"  (press Enter for: {default})")
    raw = input("> ").strip()
    return raw or default


def ask_int(question: str, default: int, lo: int = 1, hi: int = 10_000) -> int:
    """Integer with a default + bounds."""
    print()
    print(question)
    print(f"  (press Enter for: {default})")
    while True:
        raw = input("> ").strip()
        if not raw:
            return default
        try:
            v = int(raw)
        except ValueError:
            print(f"  please enter a number between {lo} and {hi}")
            continue
        if lo <= v <= hi:
            return v
        print(f"  please enter a number between {lo} and {hi}")


def banner(text: str) -> None:
    bar = "-" * max(8, len(text))
    print(f"\n{bar}\n{text}\n{bar}")


# ---------------------------------------------------------------------------
# Wizard flow
# ---------------------------------------------------------------------------


# Map the toddler-mode "where will it live" question onto the platform
# preset that --platform expects in render.py.
PLATFORM_BY_KEY: dict[str, str] = {
    "tiktok":   "tiktok",
    "reels":    "reels",
    "shorts":   "shorts",
    "youtube":  "youtube",
    "x":        "x",
    "ig-feed":  "instagram-feed",
    "li":       "linkedin",
}


def gather_choices() -> dict:
    """Run the conversation and return a dict of structural choices."""
    banner("video-use wizard (Whisper fork)")
    print("I'll ask a few short questions, then print the commands to run.")
    print("Hit Enter on any question to take the [default].")

    where = ask_choice(
        "Where will this video live?",
        [
            ("tiktok",  "TikTok          (vertical, 9:16, ~3 min)"),
            ("reels",   "Instagram Reels (vertical, 9:16, ~90 s)"),
            ("shorts",  "YouTube Shorts  (vertical, 9:16, <60 s)"),
            ("youtube", "YouTube         (horizontal, 16:9, longer)"),
            ("x",       "X / Twitter     (horizontal, 16:9, ~2 min)"),
            ("ig-feed", "Instagram Feed  (square, 1:1)"),
            ("li",      "LinkedIn        (square, 1:1, professional)"),
        ],
        default_key="tiktok",
    )
    platform = PLATFORM_BY_KEY[where]

    length_default = {
        "tiktok":  180,
        "reels":   90,
        "shorts":  55,
        "youtube": 600,
        "x":       120,
        "ig-feed": 60,
        "li":      90,
    }[where]
    length_s = ask_int(
        "How long should the final video be (in seconds)?",
        default=length_default, lo=10, hi=3600,
    )

    mood = ask_choice(
        "What's the vibe?",
        [
            ("a", "snappy and punchy   (fast cuts, energetic music)"),
            ("b", "warm and friendly   (medium cuts, soft music)"),
            ("c", "calm and thoughtful (slower cuts, sparse music)"),
            ("d", "no music           (just talking, clean cuts)"),
        ],
        default_key="a",
    )
    mood_label = {"a": "snappy", "b": "warm", "c": "calm", "d": "no-music"}[mood]

    captions = ask_choice(
        "Do you want big readable captions on top of the video?",
        [
            ("y", "yes, bold captions  (TikTok / Reels look)"),
            ("n", "no captions, just the audio"),
        ],
        default_key="y",
    )
    want_captions = captions == "y"

    if where in ("tiktok", "reels", "shorts", "ig-feed", "li"):
        crop = ask_choice(
            "How should the crop work?",
            [
                ("auto",    "smart    (follow the speaker's face if I can find one) [recommended]"),
                ("track",   "tracking (per-frame, for moving subjects — heavier)"),
                ("center",  "center   (no face detection, just middle of frame)"),
            ],
            default_key="auto",
        )
    else:
        crop = "center"

    fit = "crop"
    if where == "youtube" and ask_choice(
        "Some clips are vertical. How do you want to handle that?",
        [
            ("crop", "crop them to 16:9 (cuts the top/bottom off)"),
            ("blur", "blurred background bars (TikTok-import style)"),
            ("pad",  "plain black bars on the sides"),
        ],
        default_key="blur",
    ) != "crop":
        fit = ask_choice(
            "Confirm the bar style:",
            [("blur", "blurred"), ("pad", "black")],
            default_key="blur",
        )

    backend = ask_choice(
        "Which Whisper backend?",
        [
            ("auto",   "auto-pick (recommended — uses MLX on Apple Silicon, faster-whisper elsewhere)"),
            ("fw",     "faster-whisper (local, CPU/CUDA)"),
            ("mlx",    "mlx-whisper   (Apple Silicon only, blazing fast)"),
            ("openai", "OpenAI hosted (cloud, requires OPENAI_API_KEY, paid)"),
        ],
        default_key="auto",
    )
    backend_name = {
        "auto":   "auto",
        "fw":     "faster-whisper",
        "mlx":    "mlx",
        "openai": "openai",
    }[backend]

    model = ask_choice(
        "How accurate should transcription be?",
        [
            ("base",    "fast      (good enough for quick cuts)"),
            ("small",   "balanced  (recommended for most projects)"),
            ("medium",  "careful   (slower; reliable for noisy audio)"),
            ("large-v3","best      (slowest; for finals where every word matters)"),
        ],
        default_key="small",
    )

    return {
        "where":         where,
        "platform":      platform,
        "length_s":      length_s,
        "mood":          mood_label,
        "want_captions": want_captions,
        "crop":          crop,
        "fit":           fit,
        "backend":       backend_name,
        "model":         model,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def emit_plan(choices: dict, raw_dir: Path, edit_dir: Path) -> None:
    banner("here's what I'll do")
    mins = choices["length_s"] / 60
    print(f"  destination : {choices['where']} (preset: {choices['platform']})")
    print(f"  target length: {choices['length_s']}s  (~{mins:.1f} min)")
    print(f"  vibe        : {choices['mood']}")
    print(f"  captions    : {'big bold overlay' if choices['want_captions'] else 'none'}")
    print(f"  crop mode   : {choices['crop']}")
    print(f"  fit         : {choices['fit']}")
    print(f"  whisper     : backend={choices['backend']}, model={choices['model']}")
    print()
    print(f"  raw videos  : {raw_dir}")
    print(f"  edit folder : {edit_dir}")


def emit_commands(choices: dict, raw_dir: Path, edit_dir: Path) -> None:
    banner("step 1 — transcribe (one-time per source video)")
    transcribe_cmd = [
        sys.executable, "helpers/transcribe_batch.py",
        "--raw", str(raw_dir),
        "--edit", str(edit_dir),
        "--backend", choices["backend"],
        "--model", choices["model"],
    ]
    print("  " + " ".join(shlex.quote(p) for p in transcribe_cmd))
    print()
    print("  then pack the transcripts into one searchable file:")
    pack_cmd = [
        sys.executable, "helpers/pack_transcripts.py",
        "--edit", str(edit_dir),
    ]
    print("  " + " ".join(shlex.quote(p) for p in pack_cmd))

    banner("step 2 — pick your cuts")
    edl_path = edit_dir / "edl.json"
    print(f"  open  {edit_dir / 'all_transcripts.md'}")
    print(f"  read through it, mark the moments you want to keep, and write")
    print(f"  the cut list to:")
    print(f"    {edl_path}")
    print()
    print(f"  EDL skeleton (copy into {edl_path.name}):")
    skeleton = {
        "sources": {"clip1": "raw/clip1.mp4"},
        "ranges": [
            {"source": "clip1", "start": 0.00, "end": 5.00,
             "note": "intro hook"},
        ],
        "grade": "auto",
        "output": {
            "aspect": choices["platform"],
            "fit":    choices["fit"],
        },
    }
    if choices["want_captions"]:
        skeleton["subtitles"] = "master.srt"
    print(json.dumps(skeleton, indent=2))

    banner("step 3 — render")
    out_path = edit_dir / "final.mp4"
    render_cmd = [
        sys.executable, "helpers/render.py",
        str(edl_path),
        "-o", str(out_path),
        "--platform", choices["platform"],
        "--crop-mode", choices["crop"],
    ]
    if choices["fit"] != "crop":
        # Platform preset already set the fit, but if the user overrode it
        # in the bar-style question we surface that here too.
        render_cmd += ["--fit", choices["fit"]]
    if choices["want_captions"]:
        render_cmd.append("--build-subtitles")
    print("  " + " ".join(shlex.quote(p) for p in render_cmd))
    print()
    print("  draft pass first (quick, low-res, for cut-point QC):")
    draft_cmd = render_cmd + ["--draft"]
    print("  " + " ".join(shlex.quote(p) for p in draft_cmd))

    banner("done")
    print(f"final lands at: {out_path}")
    print(f"if anything looks off, edit  {edl_path}  and re-run step 3.")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Interactive wizard for video-use (Whisper fork)."
    )
    ap.add_argument("--raw", type=Path, default=None,
                    help="Folder of raw source videos (default: ./raw)")
    ap.add_argument("--edit", type=Path, default=None,
                    help="Edit working folder (default: ./edit)")
    ap.add_argument("--no-prompt", action="store_true",
                    help="Use all defaults; just print the commands.")
    args = ap.parse_args()

    raw_dir = (args.raw or Path("raw")).resolve()
    edit_dir = (args.edit or Path("edit")).resolve()

    if not raw_dir.exists():
        print(f"warning: raw folder doesn't exist yet: {raw_dir}")
        print("  (the wizard will print commands; create the folder before "
              "running them)")

    if args.no_prompt:
        choices = {
            "where": "tiktok", "platform": "tiktok", "length_s": 180,
            "mood": "snappy", "want_captions": True,
            "crop": "auto", "fit": "crop",
            "backend": "auto", "model": "small",
        }
    else:
        try:
            choices = gather_choices()
        except (KeyboardInterrupt, EOFError):
            print("\n(aborted)")
            sys.exit(1)

    emit_plan(choices, raw_dir, edit_dir)
    emit_commands(choices, raw_dir, edit_dir)


if __name__ == "__main__":
    main()
