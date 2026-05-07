"""Render a video from an EDL.

Implements the HEURISTICS render pipeline in the correct order:

  1. Per-segment extract with color grade + 30ms audio fades baked in
  2. Lossless -c copy concat into base.mp4
  3. If overlays or subtitles: single filter graph that overlays animations
     (with PTS shift so frame 0 lands at the overlay window start)
     and applies `subtitles` filter LAST → final.mp4

Optionally builds a master SRT from the per-source transcripts + EDL
output-timeline offsets, applies the proven force_style (2-word
UPPERCASE chunks, Helvetica 18 Bold, MarginV=35).

Usage:
    python helpers/render.py <edl.json> -o final.mp4
    python helpers/render.py <edl.json> -o preview.mp4 --preview
    python helpers/render.py <edl.json> -o final.mp4 --build-subtitles
    python helpers/render.py <edl.json> -o final.mp4 --no-subtitles
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Force UTF-8 stdio so unicode prints don't crash the script on Windows
# where the default locale is cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

try:
    from grade import get_preset, auto_grade_for_clip  # same directory
except Exception:
    def get_preset(name: str) -> str:
        return ""

    def auto_grade_for_clip(video, start=0.0, duration=None, verbose=False):  # type: ignore
        return "eq=contrast=1.03:saturation=0.98", {}

# Subject-aware cropping is fully optional. If opencv isn't installed,
# auto_crop.has_opencv() returns False and crop_mode "auto" silently falls
# back to "center", so importing the module never blocks the rest of render.py.
try:
    import auto_crop  # same directory
except Exception:
    auto_crop = None  # type: ignore[assignment]


# -------- Subtitle style (bold-overlay, proven at 1920×1080 and 1080×1920) --
#
# MarginV is NOT taste — it is a platform safe-zone rule.
# TikTok / IG Reels / Shorts UI (caption, username, music, right-rail actions)
# covers roughly the bottom ~25–30% of a 1080×1920 frame. Captions placed near
# the bottom edge get clipped or obscured by the UI. libass auto-scales the
# render canvas relative to PlayResY=288, so MarginV=90 lands the caption
# baseline roughly 30% up from the bottom on any aspect — clear of the UI on
# every major vertical-video platform. Do not drop this below ~75 without a
# specific reason.
SUB_FORCE_STYLE = (
    "FontName=Helvetica,FontSize=18,Bold=1,"
    "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,"
    "BorderStyle=1,Outline=2,Shadow=0,"
    "Alignment=2,MarginV=90"
)

# -------- Helpers ------------------------------------------------------------


def run(cmd: list[str], quiet: bool = False) -> None:
    if not quiet:
        print(f"  $ {' '.join(str(c) for c in cmd[:6])}{' …' if len(cmd) > 6 else ''}")
    subprocess.run(cmd, check=True)


def resolve_grade_filter(grade_field: str | None) -> str:
    """The EDL's 'grade' field can be a preset name, a raw ffmpeg filter, or 'auto'.

    Returns the filter string to embed into the per-segment -vf chain.
    For 'auto', returns the sentinel "__AUTO__" which is resolved per-segment.
    """
    if not grade_field:
        return ""
    if grade_field == "auto":
        return "__AUTO__"
    # Preset names are short identifiers, filter strings contain '=' or ','.
    if re.fullmatch(r"[a-zA-Z0-9_\-]+", grade_field):
        try:
            return get_preset(grade_field)
        except KeyError:
            print(f"warning: unknown preset '{grade_field}', using as raw filter")
            return grade_field
    return grade_field


def resolve_path(maybe_path: str, base: Path) -> Path:
    """Resolve a path that may be absolute or relative to `base`."""
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


# -------- HDR → SDR tone mapping (HLG / PQ sources) --------------------------
#
# iPhone defaults to HLG HDR in Rec.2020 (and many mirrorless cameras ship PQ).
# If the source is HDR and we only downconvert bit depth (yuv420p10le → yuv420p)
# without tone-mapping, the output is 8-bit but still carries HLG/PQ transfer
# metadata. Players that honor the metadata (screen recorders, most social
# upload re-encodes) interpret 8-bit values in an HDR container and the result
# looks oversaturated / blown out. QuickTime on macOS can hide this locally —
# screen recording and uploaded renders cannot.
#
# Fix: detect HDR via color_transfer and prepend a zscale+tonemap chain to the
# vf graph so the output is clean Rec.709 SDR.

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}  # PQ (HDR10) and HLG

TONEMAP_CHAIN = (
    "zscale=t=linear:npl=100,"
    "format=gbrpf32le,"
    "zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv,"
    "format=yuv420p"
)


def is_hdr_source(video: Path) -> bool:
    """Return True if the source uses a PQ or HLG transfer function."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_transfer",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() in HDR_TRANSFERS
    except subprocess.CalledProcessError:
        return False


# -------- Aspect / output-size presets --------------------------------------
#
# Output dimensions and fit behavior are first-class. The renderer reads them
# from the EDL `output` field (preferred) or from --aspect / --fit on the CLI.
# Subtitles work on every aspect because libass MarginV is in PlayResY units
# and scales by ratio.

ASPECT_PRESETS: dict[str, tuple[int, int]] = {
    # Vertical / short-form social. All three platforms use 1080x1920 9:16.
    "vertical":  (1080, 1920),
    "tiktok":    (1080, 1920),
    "reels":     (1080, 1920),
    "shorts":    (1080, 1920),
    # Horizontal / widescreen. The legacy default.
    "horizontal": (1920, 1080),
    "youtube":    (1920, 1080),
    "tv":         (1920, 1080),
    "1080p":      (1920, 1080),
    # Square (Instagram feed, X video).
    "square":     (1080, 1080),
    "instagram":  (1080, 1080),
    # 4K cinema.
    "4k":         (3840, 2160),
    "uhd":        (3840, 2160),
}

DEFAULT_ASPECT = "horizontal"
DEFAULT_FIT = "crop"  # for portrait<->landscape conversions
DEFAULT_BLUR_SIGMA = 24  # ffmpeg gblur sigma for fit=blur background

# All four crop modes:
#   - center  : classic centered window (no detection needed)
#   - auto    : if opencv is installed and a face is found, behave as
#               "subject"; otherwise fall back silently to "center"
#   - subject : one fixed crop per segment, centered on the average face
#               position within that segment
#   - track   : true dynamic per-frame trajectory via piecewise-linear
#               expressions in the ffmpeg crop filter's x= / y= args
CROP_MODES = ("center", "auto", "subject", "track")
DEFAULT_CROP_MODE = "auto"

# Platform shortcuts. --platform NAME sets aspect + fit + crop_mode in
# one go; explicit CLI flags still win. Built around the dominant
# delivery formats; users can compose their own with --aspect / --fit /
# --crop-mode if they need something else.
PLATFORM_PRESETS: dict[str, dict[str, str]] = {
    "tiktok":          {"aspect": "tiktok",     "fit": "crop", "crop_mode": "auto"},
    "reels":           {"aspect": "reels",      "fit": "crop", "crop_mode": "auto"},
    "shorts":          {"aspect": "shorts",     "fit": "crop", "crop_mode": "auto"},
    "youtube-shorts":  {"aspect": "shorts",     "fit": "crop", "crop_mode": "auto"},
    "instagram":       {"aspect": "square",     "fit": "crop", "crop_mode": "auto"},
    "instagram-feed":  {"aspect": "square",     "fit": "crop", "crop_mode": "auto"},
    "instagram-reels": {"aspect": "reels",      "fit": "crop", "crop_mode": "auto"},
    "youtube":         {"aspect": "youtube",    "fit": "pad",  "crop_mode": "center"},
    "linkedin":        {"aspect": "square",     "fit": "crop", "crop_mode": "auto"},
    "x":               {"aspect": "horizontal", "fit": "pad",  "crop_mode": "center"},
    "twitter":         {"aspect": "horizontal", "fit": "pad",  "crop_mode": "center"},
}


def parse_aspect(value: str | None) -> tuple[int, int]:
    """Resolve an --aspect string to (width, height).

    Accepts a preset name or an explicit "WxH" / "W,H" / "W:H" form.
    """
    if not value:
        value = DEFAULT_ASPECT
    v = value.strip().lower()
    if v in ASPECT_PRESETS:
        return ASPECT_PRESETS[v]
    for sep in ("x", ":", ","):
        if sep in v:
            try:
                w_str, h_str = v.split(sep, 1)
                w, h = int(w_str), int(h_str)
                if w <= 0 or h <= 0:
                    raise ValueError
                return w, h
            except ValueError:
                break
    raise ValueError(
        f"invalid --aspect '{value}'. Use a preset "
        f"({', '.join(sorted(set(ASPECT_PRESETS)))}) or WxH like 1080x1920."
    )


def build_size_filter(
    target_w: int,
    target_h: int,
    fit: str,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
) -> str:
    """Build a ffmpeg -vf chain that resizes input to exactly (target_w, target_h)
    using the chosen fit mode.

    Fit modes:
      - crop:  center-crop input to target aspect, then scale exactly. No black
               bars, no distortion. Default for landscape <-> portrait. The
               static center crop here is overridden by extract_segment when a
               dynamic crop_mode is in use.
      - pad:   scale to fit inside the target box, fill the remainder with black.
               Loses no pixels but adds bars when source aspect != target aspect.
      - blur:  scale to fit inside, fill the remainder with a Gaussian-blurred
               copy of the source. The TikTok / Reels filler-bg style.
               `blur_sigma` controls how blurry the background is (default 24).
      - scale: just stretch. Distorts; rarely wanted but available.
    """
    target_ar = target_w / target_h

    if fit == "scale":
        return f"scale={target_w}:{target_h}"

    if fit == "crop":
        # iw / ih are input width/height. Pick the larger inner box that fits
        # the target aspect, center-crop to it, then scale.
        return (
            f"crop='if(gt(iw/ih,{target_ar:.6f}),ih*{target_ar:.6f},iw)'"
            f":'if(gt(iw/ih,{target_ar:.6f}),ih,iw/{target_ar:.6f})',"
            f"scale={target_w}:{target_h}:flags=lanczos,setsar=1"
        )

    if fit == "pad":
        return (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
        )

    if fit == "blur":
        # split=2 inside -vf works in modern ffmpeg. Background gets scaled to
        # cover (force_original_aspect_ratio=increase + crop), gaussian blur,
        # then the foreground (fit-inside) is overlaid on top. `blur_sigma`
        # is the gblur strength — higher = softer / more dreamy.
        sigma = max(0.1, float(blur_sigma))
        return (
            f"split=2[bg][fg];"
            f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={target_w}:{target_h},gblur=sigma={sigma:.2f}[bgblur];"
            f"[fg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:flags=lanczos[fgs];"
            f"[bgblur][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1"
        )

    raise ValueError(f"unknown fit mode: {fit!r} (choose crop|pad|blur|scale)")


def resolve_output_size(
    edl: dict,
    cli_aspect: str | None,
    cli_fit: str | None,
    cli_crop_mode: str | None = None,
    cli_blur_sigma: float | None = None,
) -> tuple[int, int, str, str, float]:
    """Resolve the final (width, height, fit, crop_mode, blur_sigma) by
    precedence: CLI > EDL > defaults.

    EDL `output` block looks like:
        "output": {"width": 1080, "height": 1920, "fit": "crop"}
        "output": {"aspect": "tiktok", "fit": "blur", "blur_sigma": 32}
        "output": {"aspect": "tiktok", "fit": "crop", "crop_mode": "track"}
    """
    edl_out = (edl.get("output") or {}) if isinstance(edl, dict) else {}

    if cli_aspect:
        w, h = parse_aspect(cli_aspect)
    elif "width" in edl_out and "height" in edl_out:
        w, h = int(edl_out["width"]), int(edl_out["height"])
    elif "aspect" in edl_out:
        w, h = parse_aspect(str(edl_out["aspect"]))
    else:
        w, h = parse_aspect(DEFAULT_ASPECT)

    fit = cli_fit or edl_out.get("fit") or DEFAULT_FIT
    if fit not in {"crop", "pad", "blur", "scale"}:
        raise ValueError(f"invalid fit '{fit}' (choose crop|pad|blur|scale)")

    crop_mode = cli_crop_mode or edl_out.get("crop_mode") or DEFAULT_CROP_MODE
    if crop_mode not in CROP_MODES:
        raise ValueError(f"invalid crop_mode '{crop_mode}' "
                         f"(choose {'|'.join(CROP_MODES)})")

    if cli_blur_sigma is not None:
        blur_sigma = float(cli_blur_sigma)
    else:
        blur_sigma = float(edl_out.get("blur_sigma", DEFAULT_BLUR_SIGMA))

    return w, h, fit, crop_mode, blur_sigma


# -------- Per-segment extraction (Rule 2 + Rule 3) --------------------------


def extract_segment(
    source: Path,
    seg_start: float,
    duration: float,
    grade_filter: str,
    out_path: Path,
    preview: bool = False,
    draft: bool = False,
    target_width: int = 1920,
    target_height: int = 1080,
    fit: str = DEFAULT_FIT,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
    size_filter_override: str | None = None,
) -> None:
    """Extract a cut range as its own MP4 with grade + 30ms audio fades baked in.

    `-ss` before `-i` for fast accurate seeking. Scales to (target_width,
    target_height) using the given fit mode, applies HDR tone-mapping when the
    source carries HLG/PQ transfer.

    `size_filter_override`: when subject-aware crop is in use, the caller
    pre-builds the crop+scale vf for this exact segment (with face-tracked
    coordinates) and passes it here. We use it instead of the static
    build_size_filter() output. None = use the default static filter.

    Quality ladder:
      - final (default): full target res, libx264 fast CRF 20
      - preview:         full target res, libx264 medium CRF 22
      - draft:           half-target res, libx264 ultrafast CRF 28 (cut-point check)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Draft mode halves both dimensions while preserving aspect (snap to even).
    if draft:
        target_width = max(2, (target_width // 2) - (target_width // 2) % 2)
        target_height = max(2, (target_height // 2) - (target_height // 2) % 2)

    if size_filter_override:
        size_filter = size_filter_override
    else:
        size_filter = build_size_filter(target_width, target_height, fit, blur_sigma)

    vf_parts: list[str] = []
    if is_hdr_source(source):
        vf_parts.append(TONEMAP_CHAIN)
    vf_parts.append(size_filter)
    if grade_filter:
        vf_parts.append(grade_filter)
    vf = ",".join(vf_parts)

    # 30ms audio fades at both edges (Rule 3) — prevent pops
    fade_out_start = max(0.0, duration - 0.03)
    af = f"afade=t=in:st=0:d=0.03,afade=t=out:st={fade_out_start:.3f}:d=0.03"

    if draft:
        preset, crf = "ultrafast", "28"
    elif preview:
        preset, crf = "medium", "22"
    else:
        preset, crf = "fast", "20"

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{seg_start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-af", af,
        "-c:v", "libx264", "-preset", preset, "-crf", crf,
        "-pix_fmt", "yuv420p", "-r", "24",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _resolve_crop_mode(requested: str, fit: str) -> str:
    """Translate 'auto' (and validate other values) into a concrete mode.

    Subject-aware crop only makes sense when fit=crop — pad/blur/scale either
    don't crop or paste the full frame onto a background, so subject tracking
    is moot. For those, we collapse to "center" and let build_size_filter
    handle layout. When fit=crop and OpenCV is unavailable, "auto" silently
    becomes "center" so the pipeline still renders.
    """
    if requested not in CROP_MODES:
        raise ValueError(f"unknown --crop-mode {requested!r} "
                         f"(choose {'|'.join(CROP_MODES)})")
    if fit != "crop":
        return "center"
    if requested == "auto":
        if auto_crop is not None and auto_crop.has_opencv():
            return "subject"
        return "center"
    if requested in ("subject", "track"):
        if auto_crop is None or not auto_crop.has_opencv():
            print(f"  warning: --crop-mode {requested} requested but opencv "
                  f"is not installed; falling back to center crop. "
                  f"Install with: pip install opencv-python-headless")
            return "center"
    return requested


def extract_all_segments(
    edl: dict,
    edit_dir: Path,
    preview: bool,
    draft: bool = False,
    target_width: int = 1920,
    target_height: int = 1080,
    fit: str = DEFAULT_FIT,
    crop_mode: str = DEFAULT_CROP_MODE,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
) -> list[Path]:
    """Extract every EDL range into edit_dir/clips_graded/seg_NN.mp4.
    Returns the ordered list of segment paths.

    Each segment is extracted at the target output dimensions so the
    downstream concat is uniform and -c copy works without re-encoding.

    When crop_mode is "subject" or "track" (or "auto" + OpenCV available),
    we run face detection once per source (cached) and replace the static
    center-crop with a per-segment subject-aware crop expression.
    """
    resolved = resolve_grade_filter(edl.get("grade"))
    is_auto = resolved == "__AUTO__"
    clips_dir = edit_dir / (
        "clips_draft" if draft else ("clips_preview" if preview else "clips_graded")
    )
    clips_dir.mkdir(parents=True, exist_ok=True)

    ranges = edl["ranges"]
    sources = edl["sources"]

    effective_crop_mode = _resolve_crop_mode(crop_mode, fit)

    # Pre-compute (and cache) per-source face tracks if we're going dynamic.
    # Uses one detection pass per unique source, no matter how many segments
    # reference it.
    track_cache: dict[str, dict] = {}
    if effective_crop_mode in ("subject", "track") and auto_crop is not None:
        unique_srcs = {r["source"] for r in ranges}
        for src_name in unique_srcs:
            src_path = resolve_path(sources[src_name], edit_dir)
            try:
                track_cache[src_name] = auto_crop.get_or_compute_track(
                    src_path, edit_dir, verbose=True,
                )
            except SystemExit as e:
                print(f"  ! face detection failed for {src_name}: {e}")
                effective_crop_mode = "center"
                break

    seg_paths: list[Path] = []
    crop_label = effective_crop_mode if fit == "crop" else "n/a"
    print(f"extracting {len(ranges)} segment(s) -> {clips_dir.name}/  "
          f"({target_width}x{target_height}, fit={fit}, crop={crop_label})")
    if is_auto:
        print("  (auto-grade per segment: analyzing each range)")
    for i, r in enumerate(ranges):
        src_name = r["source"]
        src_path = resolve_path(sources[src_name], edit_dir)
        start = float(r["start"])
        end = float(r["end"])
        duration = end - start
        out_path = clips_dir / f"seg_{i:02d}_{src_name}.mp4"

        if is_auto:
            seg_filter, _stats = auto_grade_for_clip(src_path, start=start, duration=duration, verbose=False)
        else:
            seg_filter = resolved

        # Build a dynamic size filter for this segment if we have a track.
        size_filter_override: str | None = None
        if effective_crop_mode in ("subject", "track") and src_name in track_cache:
            track = track_cache[src_name]
            if effective_crop_mode == "subject":
                size_filter_override = auto_crop.build_subject_crop_filter(
                    track, start, end, target_width, target_height,
                )
            else:  # "track"
                size_filter_override = auto_crop.build_dynamic_crop_filter(
                    track, start, end, target_width, target_height,
                )
            if size_filter_override is None:
                # No face in this segment — silent fallback to center crop.
                pass

        note = r.get("beat") or r.get("note") or ""
        crop_marker = ""
        if size_filter_override:
            crop_marker = f"  [{effective_crop_mode}]"
        print(f"  [{i:02d}] {src_name}  {start:7.2f}-{end:7.2f}  "
              f"({duration:5.2f}s)  {note}{crop_marker}")
        if is_auto:
            print(f"        grade: {seg_filter or '(none)'}")
        extract_segment(
            src_path, start, duration, seg_filter, out_path,
            preview=preview, draft=draft,
            target_width=target_width, target_height=target_height, fit=fit,
            blur_sigma=blur_sigma,
            size_filter_override=size_filter_override,
        )
        seg_paths.append(out_path)

    return seg_paths


# -------- Lossless concat ----------------------------------------------------


def concat_segments(segment_paths: list[Path], out_path: Path, edit_dir: Path) -> None:
    """Lossless concat via the concat demuxer. No re-encode."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list = edit_dir / "_concat.txt"
    concat_list.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in segment_paths),
        encoding="utf-8",
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"concat → {out_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    concat_list.unlink(missing_ok=True)


# -------- Master SRT (Rule 5) ------------------------------------------------


PUNCT_BREAK = set(".,!?;:")


def _srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _words_in_range(transcript: dict, t_start: float, t_end: float) -> list[dict]:
    out: list[dict] = []
    for w in transcript.get("words", []):
        if w.get("type") != "word":
            continue
        ws = w.get("start")
        we = w.get("end")
        if ws is None or we is None:
            continue
        if we <= t_start or ws >= t_end:
            continue
        out.append(w)
    return out


def build_master_srt(edl: dict, edit_dir: Path, out_path: Path) -> None:
    """Build an output-timeline SRT from per-source transcripts.

    - 2-word chunks (break on any punctuation in between)
    - UPPERCASE text
    - Output times computed as word.start - segment_start + segment_offset
    """
    transcripts_dir = edit_dir / "transcripts"
    sources = edl["sources"]

    entries: list[tuple[float, float, str]] = []
    seg_offset = 0.0

    for r in edl["ranges"]:
        src_name = r["source"]
        seg_start = float(r["start"])
        seg_end = float(r["end"])
        seg_duration = seg_end - seg_start

        tr_path = transcripts_dir / f"{src_name}.json"
        if not tr_path.exists():
            print(f"  no transcript for {src_name}, skipping captions for this segment")
            seg_offset += seg_duration
            continue

        transcript = json.loads(tr_path.read_text(encoding="utf-8-sig"))
        words_in_seg = _words_in_range(transcript, seg_start, seg_end)

        # Group into 2-word chunks, break on punctuation
        chunks: list[list[dict]] = []
        current: list[dict] = []
        for w in words_in_seg:
            text = (w.get("text") or "").strip()
            if not text:
                continue
            current.append(w)
            # Break if the current text ends in punctuation or we hit 2 words
            ends_in_punct = bool(text) and text[-1] in PUNCT_BREAK
            if len(current) >= 2 or ends_in_punct:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        for chunk in chunks:
            local_start = max(seg_start, chunk[0].get("start", seg_start))
            local_end = min(seg_end, chunk[-1].get("end", seg_end))
            out_start = max(0.0, local_start - seg_start) + seg_offset
            out_end = max(0.0, local_end - seg_start) + seg_offset
            if out_end <= out_start:
                out_end = out_start + 0.4
            text = " ".join((w.get("text") or "").strip() for w in chunk)
            text = re.sub(r"\s+", " ", text).strip()
            # Strip trailing punctuation for cleaner uppercase look
            text = text.rstrip(",;:")
            text = text.upper()
            entries.append((out_start, out_end, text))

        seg_offset += seg_duration

    # Sort and write as SRT
    entries.sort(key=lambda e: e[0])
    lines: list[str] = []
    for i, (a, b, t) in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(a)} --> {_srt_timestamp(b)}")
        lines.append(t)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"master SRT → {out_path.name} ({len(entries)} cues)")


# -------- Loudness normalization (social-ready audio) -----------------------


# Social-media standard: -14 LUFS integrated, -1 dBTP peak, LRA 11 LU.
# Matches YouTube / Instagram / TikTok / X / LinkedIn normalization targets.
LOUDNORM_I = -14.0
LOUDNORM_TP = -1.0
LOUDNORM_LRA = 11.0


def measure_loudness(video_path: Path) -> dict[str, str] | None:
    """Run ffmpeg loudnorm first pass and parse the JSON measurement.

    Returns a dict with measured_i, measured_tp, measured_lra, measured_thresh,
    target_offset, or None if measurement failed.
    """
    filter_str = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}:print_format=json"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(video_path),
        "-af", filter_str,
        "-vn", "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # loudnorm prints the JSON to stderr at the end of the run
    stderr = proc.stderr

    # Find the JSON block — loudnorm output contains a `{ ... }` block
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(stderr[start : end + 1])
    except json.JSONDecodeError:
        return None
    needed = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}
    if not needed.issubset(data.keys()):
        return None
    return data


def apply_loudnorm_two_pass(
    input_path: Path,
    output_path: Path,
    preview: bool = False,
) -> bool:
    """Run two-pass loudnorm on input_path, write normalized copy to output_path.

    Returns True on success, False if measurement failed (caller should fall
    back to copying the input unchanged).

    In preview mode, skips the measurement pass and uses a one-pass approximation
    for speed. Final mode always does the proper two-pass.
    """
    if preview:
        # One-pass approximation — faster, slightly less accurate.
        filter_str = f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(input_path),
            "-c:v", "copy",
            "-af", filter_str,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            str(output_path),
        ]
        print(f"  loudnorm (1-pass preview) → {output_path.name}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True

    # Full two-pass
    print(f"  loudnorm pass 1: measuring {input_path.name}")
    measurement = measure_loudness(input_path)
    if measurement is None:
        print("  loudnorm measurement failed — falling back to 1-pass")
        return apply_loudnorm_two_pass(input_path, output_path, preview=True)

    print(f"    measured: I={measurement['input_i']} LUFS  "
          f"TP={measurement['input_tp']}  LRA={measurement['input_lra']}")

    filter_str = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        f":measured_I={measurement['input_i']}"
        f":measured_TP={measurement['input_tp']}"
        f":measured_LRA={measurement['input_lra']}"
        f":measured_thresh={measurement['input_thresh']}"
        f":offset={measurement['target_offset']}"
        f":linear=true"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(input_path),
        "-c:v", "copy",
        "-af", filter_str,
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(output_path),
    ]
    print(f"  loudnorm pass 2: normalizing → {output_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return True


# -------- Final compositing (Rule 1 + Rule 4) -------------------------------


def build_final_composite(
    base_path: Path,
    overlays: list[dict],
    subtitles_path: Path | None,
    out_path: Path,
    edit_dir: Path,
) -> None:
    """Final pass: base → overlays (PTS-shifted) → subtitles LAST → out.

    If there are no overlays and no subtitles, just copy base to out.
    """
    has_overlays = bool(overlays)
    has_subs = subtitles_path is not None and subtitles_path.exists()

    if not has_overlays and not has_subs:
        # Nothing to do — just rename/copy base to final name
        run(["ffmpeg", "-y", "-i", str(base_path), "-c", "copy", str(out_path)], quiet=True)
        return

    inputs: list[str] = ["-i", str(base_path)]
    for ov in overlays:
        ov_path = resolve_path(ov["file"], edit_dir)
        inputs += ["-i", str(ov_path)]

    filter_parts: list[str] = []
    # PTS-shift every overlay so its frame 0 lands at start_in_output
    for idx, ov in enumerate(overlays, start=1):
        t = float(ov["start_in_output"])
        filter_parts.append(f"[{idx}:v]setpts=PTS-STARTPTS+{t}/TB[a{idx}]")

    # Chain overlays on top of base
    current = "[0:v]"
    for idx, ov in enumerate(overlays, start=1):
        t = float(ov["start_in_output"])
        dur = float(ov["duration"])
        end = t + dur
        next_label = f"[v{idx}]"
        filter_parts.append(
            f"{current}[a{idx}]overlay=enable='between(t,{t:.3f},{end:.3f})'{next_label}"
        )
        current = next_label

    # Subtitles LAST — Rule 1
    if has_subs:
        subs_abs = str(subtitles_path.resolve()).replace(":", r"\:").replace("'", r"\'")
        filter_parts.append(
            f"{current}subtitles='{subs_abs}':force_style='{SUB_FORCE_STYLE}'[outv]"
        )
        out_label = "[outv]"
    else:
        # Rename the last overlay output to [outv] for consistency
        if has_overlays:
            filter_parts.append(f"{current}null[outv]")
            out_label = "[outv]"
        else:
            out_label = "[0:v]"

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", out_label,
        "-map", "0:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"compositing → {out_path.name}")
    print(f"  overlays: {len(overlays)}, subtitles: {'yes' if has_subs else 'no'}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


# -------- Main ---------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a video from an EDL")
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output video path")
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Preview mode: 1080p, medium, CRF 22 — evaluable for QC, faster than final.",
    )
    ap.add_argument(
        "--draft",
        action="store_true",
        help="Draft mode: 720p, ultrafast, CRF 28 — cut-point verification only.",
    )
    ap.add_argument(
        "--build-subtitles",
        action="store_true",
        help="Build master.srt from transcripts + EDL offsets before compositing",
    )
    ap.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Skip subtitles even if the EDL references one",
    )
    ap.add_argument(
        "--no-loudnorm",
        action="store_true",
        help="Skip audio loudness normalization. Default is on (-14 LUFS, -1 dBTP, LRA 11).",
    )
    ap.add_argument(
        "--aspect",
        type=str,
        default=None,
        help="Output dimensions. Preset name "
             "(vertical|tiktok|reels|shorts | horizontal|youtube|tv|1080p | "
             "square|instagram | 4k|uhd) or explicit WxH like 1080x1920. "
             "Overrides the EDL `output` block. Default: horizontal "
             "(1920x1080) unless the EDL says otherwise.",
    )
    ap.add_argument(
        "--fit",
        type=str,
        choices=["crop", "pad", "blur", "scale"],
        default=None,
        help="How to fit source frames into the target aspect when they "
             "differ. crop = center-crop (no bars, default), pad = black "
             "bars, blur = blurred copy as background (TikTok-style), "
             "scale = stretch. Overrides the EDL `output.fit` field.",
    )
    ap.add_argument(
        "--crop-mode",
        type=str,
        choices=list(CROP_MODES),
        default=None,
        help="How to position the crop window when fit=crop. "
             "center = static center-crop (no detection). "
             "auto = follow the subject if opencv is installed and a face "
             "is found (default), else center. "
             "subject = one fixed crop per segment, centered on the average "
             "face position in that segment. "
             "track = full per-frame dynamic tracking (use for shots where "
             "the subject actually moves). Ignored when fit != crop.",
    )
    ap.add_argument(
        "--blur-sigma",
        type=float,
        default=DEFAULT_BLUR_SIGMA,
        help=f"Gaussian blur sigma for fit=blur background. Higher = softer. "
             f"Default {DEFAULT_BLUR_SIGMA}.",
    )
    ap.add_argument(
        "--platform",
        type=str,
        choices=sorted(PLATFORM_PRESETS.keys()),
        default=None,
        help="One-shot platform preset that bundles --aspect, --fit and "
             "--crop-mode. Explicit individual flags still win. "
             "Choices: " + ", ".join(sorted(PLATFORM_PRESETS.keys())) + ".",
    )
    args = ap.parse_args()

    # Apply platform preset before resolving anything else, so individual
    # flags layered on top can override the preset's choices.
    if args.platform:
        preset = PLATFORM_PRESETS[args.platform]
        if args.aspect is None:
            args.aspect = preset["aspect"]
        if args.fit is None:
            args.fit = preset["fit"]
        if args.crop_mode is None:
            args.crop_mode = preset["crop_mode"]
        print(f"platform preset '{args.platform}': aspect={preset['aspect']}, "
              f"fit={preset['fit']}, crop-mode={preset['crop_mode']}")

    # crop_mode and blur_sigma resolution: CLI > EDL > default. We pass
    # None when the user didn't set the CLI flag so the EDL value can win.
    cli_crop_mode = args.crop_mode  # may be None
    cli_blur_sigma = args.blur_sigma if args.blur_sigma != DEFAULT_BLUR_SIGMA else None

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")

    # utf-8-sig tolerates both BOM-ed (Notepad, some VSCode setups) and
    # plain UTF-8 EDL files. Hand-authored EDLs often arrive with a BOM.
    edl = json.loads(edl_path.read_text(encoding="utf-8-sig"))
    edit_dir = edl_path.parent
    out_path = args.output.resolve()

    target_w, target_h, fit, crop_mode, blur_sigma = resolve_output_size(
        edl, args.aspect, args.fit, cli_crop_mode, cli_blur_sigma,
    )
    extras = []
    if fit == "crop":
        extras.append(f"crop-mode={crop_mode}")
    if fit == "blur":
        extras.append(f"blur-sigma={blur_sigma:g}")
    extras_str = (", " + ", ".join(extras)) if extras else ""
    print(f"output: {target_w}x{target_h} (fit={fit}{extras_str})")

    # 1. Extract per-segment (auto-grade per range if EDL grade is "auto")
    segment_paths = extract_all_segments(
        edl, edit_dir,
        preview=args.preview, draft=args.draft,
        target_width=target_w, target_height=target_h, fit=fit,
        crop_mode=crop_mode, blur_sigma=blur_sigma,
    )

    # 2. Concat → base
    if args.draft:
        base_name = "base_draft.mp4"
    elif args.preview:
        base_name = "base_preview.mp4"
    else:
        base_name = "base.mp4"
    base_path = edit_dir / base_name
    concat_segments(segment_paths, base_path, edit_dir)

    # 3. Subtitles: build if requested, resolve final path
    subs_path: Path | None = None
    if not args.no_subtitles:
        if args.build_subtitles:
            subs_path = edit_dir / "master.srt"
            build_master_srt(edl, edit_dir, subs_path)
        elif edl.get("subtitles"):
            subs_path = resolve_path(edl["subtitles"], edit_dir)
            if not subs_path.exists():
                print(f"warning: subtitles path in EDL does not exist: {subs_path}")
                subs_path = None

    # 4. Composite (overlays + subtitles LAST) → intermediate (pre-loudnorm) path
    overlays = edl.get("overlays") or []
    if args.no_loudnorm:
        # Composite directly to final output
        build_final_composite(base_path, overlays, subs_path, out_path, edit_dir)
    else:
        # Composite to a temp file, then run loudnorm → final output
        tmp_composite = out_path.with_suffix(".prenorm.mp4")
        build_final_composite(base_path, overlays, subs_path, tmp_composite, edit_dir)
        print("loudness normalization → social-ready (-14 LUFS / -1 dBTP / LRA 11)")
        apply_loudnorm_two_pass(tmp_composite, out_path, preview=args.draft)
        tmp_composite.unlink(missing_ok=True)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\ndone: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
