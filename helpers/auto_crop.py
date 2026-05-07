"""Subject-aware dynamic cropping for aspect conversion.

Given a source video and a target output aspect, detect the subject (face)
trajectory, smooth it, and emit an ffmpeg crop filter that keeps the
subject centered in the target frame as the shot progresses.

Two modes plus a passthrough:

  - "subject"  : one fixed crop window per segment, centered on the
                 segment's average subject position. Static within each
                 segment, so no motion artifacts.
  - "track"    : true dynamic per-frame tracking via piecewise-linear
                 expressions in the crop x/y= arguments. Best for shots
                 where the subject actually moves.
  - "center"   : ignore subject, classic centered crop. Used as the
                 fallback whenever no faces are detected.

Face detection uses OpenCV's bundled Haar cascade — no model download
required. If `opencv-python` (or the headless build) isn't installed,
this module raises a clear SystemExit telling the user how to enable
subject-aware crop, and `render.py` falls back to "center" automatically.

Tracks are cached per source at <edit>/face_tracks/<source_stem>.json
so repeated renders don't re-detect.
"""

from __future__ import annotations

import json
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


# Tuning constants. These are deliberately conservative so the default
# behavior is "calm", not "jumpy".
SAMPLE_FPS = 4               # face detection samples per second
SMOOTH_ALPHA = 0.20          # exponential smoothing weight (lower = calmer)
MAX_KEYFRAMES_PER_SEGMENT = 24  # cap nested-if depth in ffmpeg expressions
# Detection min-face size scales with frame width — Haar needs a few dozen
# pixels of actual face features to fire reliably, so we anchor it to the
# detection-scale frame width (3% works well for talking-head footage).
MIN_FACE_FRAC = 0.03
MIN_FACE_PIXELS_FLOOR = 24
# Source frames wider than this are downscaled before detection (Haar
# doesn't need 4K resolution to find a face, and tiny inputs gain nothing
# from further downscaling). Below this we feed the original frame.
DETECT_DOWNSCALE_TARGET = 960


# ---------------------------------------------------------------------------
# Availability check — gives the rest of the codebase a cheap way to ask.
# ---------------------------------------------------------------------------


def has_opencv() -> bool:
    """Return True if cv2 is importable. Used by render.py to choose
    between auto-modes without raising."""
    try:
        import cv2  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_track(
    video_path: Path,
    sample_fps: int = SAMPLE_FPS,
    verbose: bool = False,
) -> dict:
    """Sample the video at ~sample_fps Hz, run a Haar face detector, return
    the trajectory of the largest detected face per sampled frame.

    Returns a dict shaped like:

        {
          "source_width":  1920,
          "source_height": 1080,
          "duration":      32.5,
          "sample_fps":    4,
          "samples": [
            {"t": 0.25, "cx": 0.51, "cy": 0.42, "w": 0.18, "h": 0.32},
            ...
          ]
        }

    cx/cy/w/h are normalized to [0, 1] of the source dimensions. Frames
    where no face is found are simply skipped (not present in samples).
    """
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "opencv is not installed; subject-aware crop unavailable.\n"
            "Install with:\n"
            "    pip install opencv-python-headless\n"
            "or pass --crop-mode center to disable subject-aware crop."
        ) from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (n_frames / src_fps) if src_fps > 0 else 0.0

    # Two frontal cascades + a profile cascade. `_alt2` catches more 3/4
    # angle faces that the default misses; profile catches sideways speakers.
    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    haar_alt = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    profile = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml")

    sample_dt = 1.0 / max(1, sample_fps)
    samples: list[dict] = []
    t = 0.0
    sampled = 0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        sampled += 1

        # Downsample for detection speed; preserve aspect. Skip downscaling
        # when the source is already at or below DETECT_DOWNSCALE_TARGET,
        # otherwise small faces become too few pixels for Haar to fire.
        h_full, w_full = frame.shape[:2]
        if w_full > DETECT_DOWNSCALE_TARGET:
            scale = w_full / DETECT_DOWNSCALE_TARGET
            small = cv2.resize(
                frame,
                (int(round(w_full / scale)), int(round(h_full / scale))),
            )
        else:
            small = frame
        sw, sh = small.shape[1], small.shape[0]
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)

        min_px = max(MIN_FACE_PIXELS_FLOOR, int(sw * MIN_FACE_FRAC))
        faces = haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3,
            minSize=(min_px, min_px),
        )
        if len(faces) == 0:
            faces = haar_alt.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3,
                minSize=(min_px, min_px),
            )
        if len(faces) == 0:
            # Try profile faces (sideways speakers, walking shots)
            faces = profile.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3,
                minSize=(min_px, min_px),
            )
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            samples.append({
                "t": round(t, 3),
                "cx": round((fx + fw / 2) / sw, 4),
                "cy": round((fy + fh / 2) / sh, 4),
                "w": round(fw / sw, 4),
                "h": round(fh / sh, 4),
            })
        t += sample_dt

    cap.release()

    if verbose:
        hit_rate = (len(samples) / sampled * 100.0) if sampled else 0.0
        print(f"  detected face in {len(samples)}/{sampled} sampled frames "
              f"({hit_rate:.0f}%)", flush=True)

    return {
        "source_width":  src_w,
        "source_height": src_h,
        "duration":      duration,
        "sample_fps":    sample_fps,
        "samples":       samples,
    }


def get_or_compute_track(
    video_path: Path,
    edit_dir: Path,
    sample_fps: int = SAMPLE_FPS,
    verbose: bool = True,
) -> dict:
    """Cached wrapper around detect_track. Returns the same dict shape."""
    cache_dir = edit_dir / "face_tracks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{video_path.stem}.json"

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8-sig"))
            # Light validation
            if "samples" in cached and "source_width" in cached:
                if verbose:
                    print(f"  cached face track: {cache_path.name} "
                          f"({len(cached['samples'])} keyframes)")
                return cached
        except Exception:
            pass  # fall through and recompute

    if verbose:
        print(f"  detecting subject track for {video_path.name}...", flush=True)
    track = detect_track(video_path, sample_fps=sample_fps, verbose=verbose)
    cache_path.write_text(json.dumps(track, indent=2), encoding="utf-8")
    return track


# ---------------------------------------------------------------------------
# Smoothing & decimation
# ---------------------------------------------------------------------------


def _ema(values: list[float], alpha: float) -> list[float]:
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def smooth_samples(samples: list[dict], alpha: float = SMOOTH_ALPHA) -> list[dict]:
    """Zero-phase exponential smoothing: forward EMA then backward EMA.

    A plain forward EMA introduces lag proportional to (1-alpha)/alpha
    samples, which makes the crop window chase the subject instead of
    following it. Since we have the full trajectory in memory before we
    render, we can run the EMA forward, reverse the result, run it
    backward, and reverse again — that cancels the phase shift while
    keeping the same noise-rejection. Net result: a calm, jitter-free
    track that stays centered on the subject.
    """
    if not samples:
        return []
    cx = [s["cx"] for s in samples]
    cy = [s["cy"] for s in samples]

    cx = _ema(cx, alpha)
    cx = list(reversed(_ema(list(reversed(cx)), alpha)))
    cy = _ema(cy, alpha)
    cy = list(reversed(_ema(list(reversed(cy)), alpha)))

    return [
        {
            "t":  samples[i]["t"],
            "cx": cx[i],
            "cy": cy[i],
            "w":  samples[i].get("w", 0.0),
            "h":  samples[i].get("h", 0.0),
        }
        for i in range(len(samples))
    ]


def downsample(samples: list[dict], max_n: int) -> list[dict]:
    """If we have more than max_n samples, evenly decimate. Preserves the
    first and last samples so the trajectory still hits the segment edges."""
    if len(samples) <= max_n:
        return samples
    step = (len(samples) - 1) / (max_n - 1)
    return [samples[round(i * step)] for i in range(max_n)]


# ---------------------------------------------------------------------------
# Crop window math
# ---------------------------------------------------------------------------


def _crop_window_dims(src_w: int, src_h: int, target_ar: float) -> tuple[int, int]:
    """Largest crop window of `target_ar` aspect that fits inside (src_w, src_h)."""
    src_ar = src_w / src_h
    if src_ar > target_ar:
        crop_h = src_h
        crop_w = int(round(src_h * target_ar))
    else:
        crop_w = src_w
        crop_h = int(round(src_w / target_ar))
    crop_w -= crop_w % 2  # libx264 needs even dims
    crop_h -= crop_h % 2
    return crop_w, crop_h


def _clamp_offset(center_norm: float, dim_src: int, dim_crop: int) -> int:
    """Convert a normalized center [0,1] to a top-left pixel offset that
    centers the crop on it, clamped so the window stays inside the frame."""
    px = center_norm * dim_src - dim_crop / 2
    return int(round(max(0, min(dim_src - dim_crop, px))))


# ---------------------------------------------------------------------------
# Filter builders
# ---------------------------------------------------------------------------


def build_subject_crop_filter(
    track: dict,
    seg_start: float,
    seg_end: float,
    target_w: int,
    target_h: int,
) -> str | None:
    """Single fixed crop centered on the segment's average face position.
    Returns None if no samples in range — caller falls back to center crop.
    """
    src_w = track["source_width"]
    src_h = track["source_height"]
    target_ar = target_w / target_h

    if abs(src_w / src_h - target_ar) < 0.01:
        return f"scale={target_w}:{target_h}:flags=lanczos,setsar=1"

    seg_samples = [s for s in track["samples"]
                   if seg_start - 0.1 <= s["t"] <= seg_end + 0.1]
    if not seg_samples:
        return None

    avg_cx = sum(s["cx"] for s in seg_samples) / len(seg_samples)
    avg_cy = sum(s["cy"] for s in seg_samples) / len(seg_samples)

    crop_w, crop_h = _crop_window_dims(src_w, src_h, target_ar)
    x_off = _clamp_offset(avg_cx, src_w, crop_w)
    y_off = _clamp_offset(avg_cy, src_h, crop_h)

    return (
        f"crop={crop_w}:{crop_h}:{x_off}:{y_off},"
        f"scale={target_w}:{target_h}:flags=lanczos,setsar=1"
    )


def build_dynamic_crop_filter(
    track: dict,
    seg_start: float,
    seg_end: float,
    target_w: int,
    target_h: int,
) -> str | None:
    """Per-frame piecewise-linear crop trajectory. Returns None if no
    samples in range. Times in the expression are segment-local (seg start
    becomes t=0) since render.py extracts each segment with -ss before -i.
    """
    src_w = track["source_width"]
    src_h = track["source_height"]
    target_ar = target_w / target_h

    if abs(src_w / src_h - target_ar) < 0.01:
        return f"scale={target_w}:{target_h}:flags=lanczos,setsar=1"

    seg_samples = [s for s in track["samples"]
                   if seg_start - 0.1 <= s["t"] <= seg_end + 0.1]
    if not seg_samples:
        return None

    seg_samples = smooth_samples(seg_samples)
    seg_samples = downsample(seg_samples, MAX_KEYFRAMES_PER_SEGMENT)

    crop_w, crop_h = _crop_window_dims(src_w, src_h, target_ar)

    keyframes_x: list[tuple[float, int]] = []
    keyframes_y: list[tuple[float, int]] = []
    for s in seg_samples:
        local_t = max(0.0, s["t"] - seg_start)
        keyframes_x.append((local_t, _clamp_offset(s["cx"], src_w, crop_w)))
        keyframes_y.append((local_t, _clamp_offset(s["cy"], src_h, crop_h)))

    # Single keyframe collapses to a static crop — same as subject mode.
    if len(keyframes_x) == 1:
        return (
            f"crop={crop_w}:{crop_h}:{keyframes_x[0][1]}:{keyframes_y[0][1]},"
            f"scale={target_w}:{target_h}:flags=lanczos,setsar=1"
        )

    x_expr = piecewise_linear_expr(keyframes_x)
    y_expr = piecewise_linear_expr(keyframes_y)

    return (
        f"crop={crop_w}:{crop_h}:'{x_expr}':'{y_expr}',"
        f"scale={target_w}:{target_h}:flags=lanczos,setsar=1"
    )


def piecewise_linear_expr(keyframes: list[tuple[float, int]]) -> str:
    """Build an ffmpeg expression that linearly interpolates between
    timestamped keyframes [(t0, v0), (t1, v1), ...] using the crop filter's
    `t` variable. Result is a nested if() tree:

        if(lt(t,t1), v0+(t-t0)*(v1-v0)/(t1-t0),
           if(lt(t,t2), v1+(t-t1)*(v2-v1)/(t2-t1),
              ...,
              vN))         <- last value held after the final keyframe

    Before the first keyframe the result holds v0 (the outermost
    if(lt(t,t1),...) handles t < t1 with the first interp segment).
    """
    if not keyframes:
        return "0"
    if len(keyframes) == 1:
        return f"{keyframes[0][1]}"

    expr = f"{keyframes[-1][1]}"  # held after final keyframe
    for i in range(len(keyframes) - 1, 0, -1):
        t0, v0 = keyframes[i - 1]
        t1, v1 = keyframes[i]
        if t1 <= t0:
            continue
        slope = (v1 - v0) / (t1 - t0)
        interp = f"{v0:.0f}+(t-{t0:.3f})*{slope:.4f}"
        expr = f"if(lt(t,{t1:.3f}),{interp},{expr})"
    return expr


# ---------------------------------------------------------------------------
# CLI (probe-only — render.py owns the actual integration)
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Probe a video's subject (face) track and dump it as JSON. "
                    "render.py uses this for --crop-mode subject|track.",
    )
    ap.add_argument("video", type=Path, help="Source video")
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Edit directory for the cache (default: <video_parent>/edit)")
    ap.add_argument("--sample-fps", type=int, default=SAMPLE_FPS,
                    help=f"Detection samples per second (default {SAMPLE_FPS})")
    ap.add_argument("--no-cache", action="store_true",
                    help="Bypass and overwrite the cached face track")
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    if args.no_cache:
        cache_path = edit_dir / "face_tracks" / f"{video.stem}.json"
        cache_path.unlink(missing_ok=True)

    track = get_or_compute_track(video, edit_dir, sample_fps=args.sample_fps)
    print(f"source: {track['source_width']}x{track['source_height']}  "
          f"duration={track['duration']:.2f}s")
    print(f"samples: {len(track['samples'])} face detections "
          f"@ {track['sample_fps']} fps")
    if track["samples"]:
        first = track["samples"][0]
        last = track["samples"][-1]
        print(f"first: t={first['t']:.2f}s  center=({first['cx']:.2f}, {first['cy']:.2f})")
        print(f"last:  t={last['t']:.2f}s  center=({last['cx']:.2f}, {last['cy']:.2f})")
    else:
        print("no faces detected — render.py will fall back to center crop")


if __name__ == "__main__":
    main()
