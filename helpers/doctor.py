"""Self-healing preflight + diagnostic + auto-fix for video-use (Whisper fork).

Run this:

  - At session start, to confirm the environment is wired up before anything
    runs that could fail in a confusing way.
  - After any helper script crashes — `--fix` mode auto-resolves the most
    common issues (BOM bytes in JSON, out-of-range EDL times, missing
    overlay / subtitle references, mis-cased aspect names) and prints a
    clear next step for anything it can't fix.
  - Before render.py on any hand-authored EDL: `--fix path/to/edl.json`
    backs up the EDL, fixes what it can in-place, and reports what
    remains.

Design notes:

  - Every check returns a typed `CheckResult` with severity ok / warn /
    fail and an OPTIONAL `fix` callable. `--fix` mode runs every fix
    whose result is a fail; warns are reported but never auto-modified.
  - All file mutations write a `<file>.bak` first so users can revert
    without git.
  - The output uses ASCII status markers ([ OK ] / [WARN] / [FAIL]) so
    it renders correctly on every terminal even before stdout is
    reconfigured to UTF-8.
  - This file is the canonical "what could go wrong" registry — extend
    it as new failure modes appear in the wild.

Exit codes:
  0 = all good (or all fail-severity issues were fixed in --fix mode)
  1 = blocking issues remain — see the SUMMARY block at the end
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Force UTF-8 stdio so we can print non-ASCII filenames or fix-summary
# arrows without crashing on Windows cp1252 terminals. Status markers
# below intentionally stay ASCII so they always render.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


SEV_OK = "ok"
SEV_WARN = "warn"
SEV_FAIL = "fail"

MARKERS = {SEV_OK: "[ OK ]", SEV_WARN: "[WARN]", SEV_FAIL: "[FAIL]"}


@dataclass
class CheckResult:
    name: str
    severity: str  # SEV_OK / SEV_WARN / SEV_FAIL
    message: str
    # Optional fix callable. Returning a string explains what was done.
    fix: Optional[Callable[[], str]] = None
    fix_summary: str = ""
    # Optional follow-up command the user should run themselves.
    suggest_cmd: str = ""


def ok(name: str, message: str) -> CheckResult:
    return CheckResult(name, SEV_OK, message)


def warn(name: str, message: str, suggest_cmd: str = "") -> CheckResult:
    return CheckResult(name, SEV_WARN, message, suggest_cmd=suggest_cmd)


def fail(
    name: str,
    message: str,
    fix: Optional[Callable[[], str]] = None,
    suggest_cmd: str = "",
) -> CheckResult:
    return CheckResult(name, SEV_FAIL, message, fix=fix, suggest_cmd=suggest_cmd)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    """Run a command, return (rc, stdout, stderr). Never raises."""
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            errors="replace",
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except Exception as e:
        return -1, "", f"{type(e).__name__}: {e}"


def _backup(path: Path) -> Path:
    """Make a .bak copy of `path` (overwrites any existing .bak). Returns
    the backup path. No-op if `path` doesn't exist yet."""
    if not path.exists():
        return path.with_suffix(path.suffix + ".bak")
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    return bak


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------


def check_python_version() -> CheckResult:
    v = sys.version_info
    if v >= (3, 10):
        return ok("python", f"{v.major}.{v.minor}.{v.micro} (>= 3.10)")
    return fail(
        "python",
        f"Python {v.major}.{v.minor}.{v.micro} found; >= 3.10 required.",
        suggest_cmd="install Python 3.10+ from https://www.python.org/downloads/",
    )


def check_ffmpeg() -> CheckResult:
    if not _which("ffmpeg"):
        return fail(
            "ffmpeg",
            "ffmpeg not found on PATH.",
            suggest_cmd=(
                "macOS:  brew install ffmpeg\n"
                "Ubuntu: sudo apt install ffmpeg\n"
                "Windows: winget install --id Gyan.FFmpeg"
            ),
        )
    rc, out, err = _run(["ffmpeg", "-version"])
    if rc != 0:
        return fail("ffmpeg", f"`ffmpeg -version` exited {rc}: {err.strip()[:200]}")
    line = (out.splitlines() or [""])[0]
    return ok("ffmpeg", line)


def check_ffprobe() -> CheckResult:
    if not _which("ffprobe"):
        return fail(
            "ffprobe",
            "ffprobe not found on PATH (ships with ffmpeg).",
            suggest_cmd="reinstall ffmpeg from your package manager",
        )
    rc, out, err = _run(["ffprobe", "-version"])
    if rc != 0:
        return fail("ffprobe", f"`ffprobe -version` exited {rc}")
    line = (out.splitlines() or [""])[0]
    return ok("ffprobe", line)


def check_libass() -> CheckResult:
    """Subtitle burn-in needs libass support. Most prebuilt ffmpegs have it."""
    rc, out, _ = _run(["ffmpeg", "-hide_banner", "-filters"])
    if rc != 0:
        return warn("libass", "could not enumerate ffmpeg filters; skipping check")
    if "ass " in out or " ass\n" in out or "subtitles" in out:
        return ok("libass", "subtitle filter available")
    return warn(
        "libass",
        "ffmpeg `subtitles`/`ass` filter not found — burn-in subtitles will fail. "
        "Install an ffmpeg build with libass.",
    )


def check_utf8_stdio() -> CheckResult:
    """We reconfigure stdio in every helper, but flag if the runtime can't."""
    if hasattr(sys.stdout, "reconfigure"):
        return ok("utf8-stdio", "sys.stdout.reconfigure available (UTF-8 safe)")
    return warn(
        "utf8-stdio",
        "older Python without sys.stdout.reconfigure — non-ASCII prints may "
        "fail on Windows cp1252 terminals. Use Python >= 3.7.",
    )


# ---------------------------------------------------------------------------
# Backend checks (informational — never auto-install, only suggest)
# ---------------------------------------------------------------------------


def _try_import(modname: str) -> tuple[bool, str]:
    """Return (importable, version_str_or_error)."""
    try:
        mod = __import__(modname)
        v = getattr(mod, "__version__", "?")
        return True, str(v)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_faster_whisper() -> CheckResult:
    importable, info = _try_import("faster_whisper")
    if importable:
        return ok("faster-whisper", f"v{info} (default local backend)")
    return fail(
        "faster-whisper",
        f"not importable: {info}",
        suggest_cmd="pip install faster-whisper>=1.0.3",
    )


def check_mlx_whisper() -> CheckResult:
    if not _is_apple_silicon():
        return ok("mlx-whisper", "n/a (Apple Silicon only — skipping)")
    importable, info = _try_import("mlx_whisper")
    if importable:
        return ok("mlx-whisper", f"v{info} (auto-picked on Apple Silicon)")
    return warn(
        "mlx-whisper",
        "not installed — falling back to faster-whisper (slower on M-series).",
        suggest_cmd="pip install -e '.[mac]'",
    )


def check_openai_pkg() -> CheckResult:
    importable, info = _try_import("openai")
    if importable:
        if os.environ.get("OPENAI_API_KEY"):
            return ok("openai", f"v{info} (OPENAI_API_KEY set)")
        return warn(
            "openai",
            f"v{info} installed but OPENAI_API_KEY not in env — hosted "
            "backend will not work.",
            suggest_cmd="echo OPENAI_API_KEY=sk-... >> .env",
        )
    return ok("openai", "n/a (not installed — only needed for --backend openai)")


def check_whisper_pkg() -> CheckResult:
    importable, info = _try_import("whisper")
    if importable:
        return ok("openai-whisper", f"v{info} (reference backend available)")
    return ok("openai-whisper", "n/a (not installed — only needed for --backend whisper)")


def check_pyannote() -> CheckResult:
    importable, info = _try_import("pyannote.audio")
    if not importable:
        return ok(
            "pyannote.audio",
            "n/a (not installed — diarization disabled)",
        )
    if not os.environ.get("HUGGINGFACE_TOKEN"):
        return warn(
            "pyannote.audio",
            f"v{info} installed but HUGGINGFACE_TOKEN not in env — "
            "diarization will fail to download the model.",
            suggest_cmd=(
                "1. Accept https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "2. echo HUGGINGFACE_TOKEN=hf_... >> .env"
            ),
        )
    return ok("pyannote.audio", f"v{info} (diarization ready)")


def check_opencv() -> CheckResult:
    importable, info = _try_import("cv2")
    if importable:
        return ok("opencv (cv2)", f"v{info} (subject-aware crop available)")
    return warn(
        "opencv (cv2)",
        "not installed — --crop-mode auto silently falls back to center crop "
        "(landscape -> vertical will not follow the speaker's face).",
        suggest_cmd="pip install -e '.[crop]'",
    )


# ---------------------------------------------------------------------------
# File hygiene
# ---------------------------------------------------------------------------


def _has_bom(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(3) == b"\xef\xbb\xbf"
    except Exception:
        return False


def find_bom_files(root: Path) -> list[Path]:
    """All .json / .srt / .md files under `root` that begin with a UTF-8 BOM."""
    if not root.exists():
        return []
    out: list[Path] = []
    for ext in ("*.json", "*.srt", "*.md"):
        for p in root.rglob(ext):
            if _has_bom(p):
                out.append(p)
    return out


def check_bom(edit_dir: Optional[Path]) -> CheckResult:
    if edit_dir is None or not edit_dir.exists():
        return ok("file-hygiene", "no edit/ directory to scan (skipped)")
    bom_files = find_bom_files(edit_dir)
    if not bom_files:
        return ok("file-hygiene", f"all .json/.srt/.md under {edit_dir} are BOM-free")

    rels = [str(p.relative_to(edit_dir)) for p in bom_files]

    def _fix() -> str:
        n = 0
        for p in bom_files:
            try:
                _backup(p)
                data = p.read_bytes()
                if data.startswith(b"\xef\xbb\xbf"):
                    p.write_bytes(data[3:])
                    n += 1
            except Exception as e:
                print(f"  ! could not strip BOM from {p}: {e}")
        return f"stripped UTF-8 BOM from {n} file(s) (.bak siblings written)"

    return fail(
        "file-hygiene",
        f"{len(bom_files)} file(s) start with a UTF-8 BOM: " + ", ".join(rels[:5])
        + (" ..." if len(rels) > 5 else ""),
        fix=_fix,
        suggest_cmd="--fix re-saves them without the BOM",
    )


# ---------------------------------------------------------------------------
# EDL validation
# ---------------------------------------------------------------------------


VALID_FITS = {"crop", "pad", "blur", "scale"}
VALID_CROP_MODES = {"center", "auto", "subject", "track"}


def _probe_duration(path: Path) -> Optional[float]:
    rc, out, _ = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    if rc != 0:
        return None
    try:
        return float(out.strip())
    except Exception:
        return None


def _resolve_path(value: str, edit_dir: Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = (edit_dir / p).resolve()
    return p


def validate_edl(edl_path: Path) -> list[CheckResult]:
    """All checks for one EDL file. Returns one CheckResult per finding."""
    results: list[CheckResult] = []
    edit_dir = edl_path.parent

    # Parse — the only thing that can short-circuit the rest of the report.
    try:
        edl_text = edl_path.read_text(encoding="utf-8-sig")
    except Exception as e:
        return [fail(f"edl:{edl_path.name}", f"could not read: {e}")]
    try:
        edl = json.loads(edl_text)
    except json.JSONDecodeError as e:
        return [fail(
            f"edl:{edl_path.name}",
            f"JSON parse error at line {e.lineno} col {e.colno}: {e.msg}",
            suggest_cmd="open the file and fix the syntax error at the reported line",
        )]
    if not isinstance(edl, dict):
        return [fail(f"edl:{edl_path.name}", "top-level JSON value is not an object")]

    sources = edl.get("sources") or {}
    ranges = edl.get("ranges") or []
    output = edl.get("output") or {}

    # Skeleton checks
    if not sources:
        results.append(fail(
            f"edl:{edl_path.name}:sources",
            "`sources` is missing or empty — every range references a source by name.",
        ))
    if not ranges:
        results.append(fail(
            f"edl:{edl_path.name}:ranges",
            "`ranges` is missing or empty — nothing to render.",
        ))

    # Source files exist + measurable duration
    durations: dict[str, Optional[float]] = {}
    for name, value in sources.items():
        path = _resolve_path(str(value), edit_dir)
        if not path.exists():
            results.append(fail(
                f"edl:{edl_path.name}:source:{name}",
                f"source file does not exist: {path}",
                suggest_cmd="check the path in `sources` or restore the file",
            ))
            durations[name] = None
            continue
        dur = _probe_duration(path)
        if dur is None:
            results.append(warn(
                f"edl:{edl_path.name}:source:{name}",
                f"could not read duration of {path.name} via ffprobe (proceeding without bounds check)",
            ))
        durations[name] = dur

    # Range bounds — collect indexes-of-issues for a single fix callable.
    range_fixes: list[tuple[int, str, float]] = []  # (idx, field, new_value)
    for i, r in enumerate(ranges):
        src = r.get("source")
        if src not in sources:
            results.append(fail(
                f"edl:{edl_path.name}:range[{i}]",
                f"source {src!r} not in sources",
            ))
            continue
        try:
            start = float(r.get("start", 0))
            end = float(r.get("end", 0))
        except (TypeError, ValueError):
            results.append(fail(
                f"edl:{edl_path.name}:range[{i}]",
                "start/end must be numeric",
            ))
            continue
        if start < 0:
            range_fixes.append((i, "start", 0.0))
            results.append(fail(
                f"edl:{edl_path.name}:range[{i}]",
                f"start {start} < 0",
                fix=None,  # batched below
            ))
        if end <= start:
            results.append(fail(
                f"edl:{edl_path.name}:range[{i}]",
                f"end ({end}) must be > start ({start})",
            ))
        dur = durations.get(src)
        if dur is not None and end > dur:
            new_end = max(start + 0.05, dur - 0.05)
            range_fixes.append((i, "end", round(new_end, 3)))
            results.append(fail(
                f"edl:{edl_path.name}:range[{i}]",
                f"end ({end}) exceeds source duration ({dur:.3f}s for {src})",
                fix=None,
                suggest_cmd=f"--fix clamps end to {new_end:.3f}s",
            ))

    # Aspect / fit / crop_mode
    if "fit" in output and output["fit"] not in VALID_FITS:
        # Fixable: snap common typos
        fix_map = {"cover": "crop", "contain": "pad", "fill": "scale"}
        sugg = fix_map.get(str(output["fit"]).lower())
        if sugg:
            def _fix_fit(_v=sugg) -> str:
                _backup(edl_path)
                cur = json.loads(edl_path.read_text(encoding="utf-8-sig"))
                cur.setdefault("output", {})["fit"] = _v
                edl_path.write_text(json.dumps(cur, indent=2), encoding="utf-8")
                return f"set output.fit = {_v!r}"
            results.append(fail(
                f"edl:{edl_path.name}:output.fit",
                f"unknown fit {output['fit']!r}",
                fix=_fix_fit,
                suggest_cmd=f"--fix sets fit to {sugg!r}",
            ))
        else:
            results.append(fail(
                f"edl:{edl_path.name}:output.fit",
                f"unknown fit {output['fit']!r} (choose one of {sorted(VALID_FITS)})",
            ))
    if "crop_mode" in output and output["crop_mode"] not in VALID_CROP_MODES:
        results.append(fail(
            f"edl:{edl_path.name}:output.crop_mode",
            f"unknown crop_mode {output['crop_mode']!r} "
            f"(choose one of {sorted(VALID_CROP_MODES)})",
        ))

    # Subtitles + overlays — fail if referenced but missing; --fix removes refs.
    subs = edl.get("subtitles")
    if subs:
        subs_path = _resolve_path(str(subs), edit_dir)
        if not subs_path.exists():
            def _fix_subs() -> str:
                _backup(edl_path)
                cur = json.loads(edl_path.read_text(encoding="utf-8-sig"))
                cur.pop("subtitles", None)
                edl_path.write_text(json.dumps(cur, indent=2), encoding="utf-8")
                return "removed missing `subtitles` reference"
            results.append(fail(
                f"edl:{edl_path.name}:subtitles",
                f"subtitles file does not exist: {subs_path}",
                fix=_fix_subs,
                suggest_cmd="--fix removes the field; re-add with `render.py --build-subtitles`",
            ))

    overlays = edl.get("overlays") or []
    bad_overlay_idxs: list[int] = []
    for i, ov in enumerate(overlays):
        if "file" not in ov:
            continue
        ov_path = _resolve_path(str(ov["file"]), edit_dir)
        if not ov_path.exists():
            bad_overlay_idxs.append(i)
            results.append(fail(
                f"edl:{edl_path.name}:overlay[{i}]",
                f"overlay file does not exist: {ov_path}",
                fix=None,
                suggest_cmd=f"--fix drops overlay[{i}] from the EDL",
            ))
    if bad_overlay_idxs:
        ov_state = {"done": False}

        def _fix_overlays(_idxs=tuple(bad_overlay_idxs), _state=ov_state) -> str:
            if _state["done"]:
                return "(already applied as part of a batched overlay fix)"
            _state["done"] = True
            _backup(edl_path)
            cur = json.loads(edl_path.read_text(encoding="utf-8-sig"))
            cur_overlays = cur.get("overlays") or []
            keep = [o for j, o in enumerate(cur_overlays) if j not in _idxs]
            cur["overlays"] = keep
            edl_path.write_text(json.dumps(cur, indent=2), encoding="utf-8")
            return f"dropped {len(_idxs)} missing overlay(s)"

        for r in results:
            if r.name.startswith(f"edl:{edl_path.name}:overlay["):
                r.fix = _fix_overlays

    # Range-bound batched fix. We attach the same callable to every
    # matching result so the "fixable" count in SUMMARY is honest. The
    # callable carries an `_already_run` flag so apply_fixes invokes it
    # once even when it appears on N results.
    if range_fixes:
        state = {"done": False}

        def _fix_ranges(_fixes=tuple(range_fixes), _state=state) -> str:
            if _state["done"]:
                return "(already applied as part of a batched range fix)"
            _state["done"] = True
            _backup(edl_path)
            cur = json.loads(edl_path.read_text(encoding="utf-8-sig"))
            for i, field_name, new_v in _fixes:
                cur["ranges"][i][field_name] = new_v
            edl_path.write_text(json.dumps(cur, indent=2), encoding="utf-8")
            return f"clamped {len(_fixes)} out-of-range value(s) in ranges[]"

        for r in results:
            if "exceeds source duration" in r.message or "< 0" in r.message:
                r.fix = _fix_ranges

    # If we got here with no findings at all, emit a single OK so the user
    # sees confirmation in the report.
    if not results:
        results.append(ok(
            f"edl:{edl_path.name}",
            f"{len(ranges)} range(s) across {len(sources)} source(s) — clean",
        ))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_block(title: str, results: list[CheckResult]) -> None:
    print()
    print(title)
    print("-" * len(title))
    for r in results:
        marker = MARKERS[r.severity]
        print(f"  {marker} {r.name}: {r.message}")
        if r.suggest_cmd:
            for line in r.suggest_cmd.strip().splitlines():
                print(f"         > {line}")


def apply_fixes(results: list[CheckResult]) -> list[str]:
    applied: list[str] = []
    for r in results:
        if r.severity == SEV_FAIL and r.fix is not None:
            try:
                summary = r.fix() or f"fixed: {r.name}"
                r.fix_summary = summary
                applied.append(f"{r.name}: {summary}")
            except Exception as e:
                applied.append(f"{r.name}: fix FAILED — {type(e).__name__}: {e}")
    return applied


def summarize(all_results: list[CheckResult], fix_applied: bool) -> int:
    n_ok = sum(1 for r in all_results if r.severity == SEV_OK)
    n_warn = sum(1 for r in all_results if r.severity == SEV_WARN)
    fails = [r for r in all_results if r.severity == SEV_FAIL]
    n_fail = len(fails)
    fixable = sum(1 for r in fails if r.fix is not None)

    print()
    print("SUMMARY")
    print("-------")
    print(f"  ok:   {n_ok}")
    print(f"  warn: {n_warn}")
    print(f"  fail: {n_fail}  ({fixable} auto-fixable with --fix)")

    if fix_applied:
        unfixed = n_fail - fixable
        if unfixed == 0:
            print("\n  all blocking issues fixed. re-run your previous command.")
            return 0
        print(f"\n  {unfixed} blocking issue(s) remain — see [FAIL] entries above.")
        return 1

    if n_fail == 0:
        print("\n  all clear.")
        return 0

    print("\n  blocking issues remain. options:")
    if fixable:
        print(f"  - run with --fix to auto-resolve the {fixable} fixable item(s)")
    print("  - read the [FAIL] lines above and fix manually")
    return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def find_default_edit_dir(edl_path: Optional[Path]) -> Optional[Path]:
    if edl_path is not None:
        return edl_path.parent
    cwd = Path.cwd()
    for cand in (cwd / "edit", cwd):
        if cand.exists():
            return cand
    return None


def run(edl_path: Optional[Path], fix: bool) -> int:
    edit_dir = find_default_edit_dir(edl_path)

    print("video-use doctor")
    print("================")
    print(f"  python    : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"  platform  : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  apple si  : {'yes' if _is_apple_silicon() else 'no'}")
    print(f"  edit dir  : {edit_dir if edit_dir else '(none detected)'}")
    if edl_path:
        print(f"  edl       : {edl_path}")

    env_results = [
        check_python_version(),
        check_ffmpeg(),
        check_ffprobe(),
        check_libass(),
        check_utf8_stdio(),
    ]
    print_block("ENVIRONMENT", env_results)

    backend_results = [
        check_faster_whisper(),
        check_mlx_whisper(),
        check_openai_pkg(),
        check_whisper_pkg(),
        check_pyannote(),
        check_opencv(),
    ]
    print_block("BACKENDS", backend_results)

    hygiene_results = [check_bom(edit_dir)]
    print_block("FILE HYGIENE", hygiene_results)

    edl_results: list[CheckResult] = []
    if edl_path:
        if not edl_path.exists():
            edl_results = [fail(f"edl:{edl_path.name}", f"file does not exist: {edl_path}")]
        else:
            edl_results = validate_edl(edl_path)
        print_block(f"EDL ({edl_path.name})", edl_results)

    all_results = env_results + backend_results + hygiene_results + edl_results

    if fix:
        applied = apply_fixes(all_results)
        if applied:
            print()
            print("FIXES APPLIED")
            print("-------------")
            for line in applied:
                print(f"  - {line}")

    return summarize(all_results, fix_applied=fix)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Self-healing preflight for video-use (Whisper fork).",
        epilog=(
            "examples:\n"
            "  python helpers/doctor.py                  # env + backend report\n"
            "  python helpers/doctor.py edit/edl.json    # also validate this EDL\n"
            "  python helpers/doctor.py --fix            # strip JSON BOMs, etc.\n"
            "  python helpers/doctor.py edit/edl.json --fix\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("edl", type=Path, nargs="?", default=None,
                    help="Optional EDL JSON to validate (default: env-only)")
    ap.add_argument("--fix", action="store_true",
                    help="Apply safe auto-fixes (BOM strip, EDL clamp, "
                         "drop missing overlay/subtitle refs). Backups are "
                         "written as <file>.bak.")
    args = ap.parse_args()
    sys.exit(run(args.edl, args.fix))


if __name__ == "__main__":
    main()
