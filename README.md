<p align="center">
  <img src="static/video-use-banner.png" alt="video-use-whisper" width="100%">
</p>

# video-use-whisper

A fork of [browser-use/video-use](https://github.com/browser-use/video-use) that swaps ElevenLabs Scribe for **local Whisper** transcription. Same conversation-driven editing pipeline, no API key, runs on your laptop.

Drop raw footage in a folder, chat with your coding agent, get `final.mp4` back. Works for any content — talking heads, montages, tutorials, travel, interviews — without presets or menus.

## What changed from upstream

- **No ElevenLabs.** No `ELEVENLABS_API_KEY`. No per-minute billing.
- **Cross-platform local transcription.** Default backend is [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) — a CTranslate2 port of Whisper with word-level timestamps. Auto-detects CUDA, falls back to CPU.
- **First-class macOS / Apple Silicon path** via [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Apple's MLX framework runs Whisper natively on the M-series GPU + Neural Engine, ~3–5× faster than `faster-whisper-CPU` on the same Mac. The smart default automatically picks `mlx` on Apple Silicon when it's installed.
- **Optional speaker diarization** via [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) — gated on a HuggingFace token. Picks up CUDA on Linux/Windows or MPS on Apple Silicon automatically. If you don't set up a token, every word gets `speaker_0` and the rest of the pipeline keeps working.
- **Optional cloud backend** (`--backend openai`) — uses OpenAI's hosted `whisper-1` with word timestamps. Useful when you have no GPU.
- **Optional MPS backend** (`--backend whisper --device mps`) — the reference openai-whisper package via PyTorch MPS. Slower than mlx but plays nicely with pyannote inside the same process.
- **Same Scribe-compatible JSON schema** — `pack_transcripts.py`, `timeline_view.py`, and `render.py` are bit-identical to upstream. Drop-in replacement regardless of which backend you pick.
- **Tradeoff:** Whisper does not detect Scribe-style audio events like `(laughter)` or `(applause)`. They were never load-bearing for cuts.

## What it does

- **Talks to you like you're five.** One plain-English question at a time — "tall or wide?", "1 minute or 3?", "punchy or chill?" — never a checklist, never jargon. Say "just go" and it stops asking and starts cutting.
- **Outputs vertical, horizontal, or square** with one flag. Built-in presets for `tiktok` / `reels` / `shorts` (1080×1920), `youtube` / `tv` (1920×1080), `square` / `instagram` (1080×1080), `4k`, plus arbitrary `WxH`. Fit modes: `crop` (no bars), `pad` (black bars), `blur` (TikTok-style blurred-copy background, `--blur-sigma N` for strength). One-shot platform bundles via `--platform tiktok|reels|shorts|youtube|youtube-shorts|instagram|instagram-feed|instagram-reels|x|twitter|linkedin`.
- **Dynamic crop that follows the speaker.** Landscape → vertical doesn't chop the speaker out of frame. With `opencv-python-headless` installed, `--crop-mode auto` (the default) detects the face per source, smooths the trajectory zero-phase, and emits a per-segment ffmpeg crop expression that keeps the subject framed. `--crop-mode track` for full per-frame tracking on moving shots; `--crop-mode subject` for one-fixed-window-per-segment; `--crop-mode center` to disable. Detection is cached at `<edit>/face_tracks/<source>.json`.
- **Cuts out filler words** (`umm`, `uh`, false starts) and dead space between takes.
- **Auto color grades** every segment (warm cinematic, neutral punch, or any custom ffmpeg chain).
- **30ms audio fades** at every cut so you never hear a pop.
- **Burns subtitles** in your style — 2-word UPPERCASE chunks by default, MarginV that clears the TikTok / Reels / Shorts safe zones automatically, fully customizable.
- **Loudness-normalizes** to social-media standard (-14 LUFS / -1 dBTP / LRA 11) so the volume sounds right on every platform.
- **Generates animation overlays** via [HyperFrames](https://github.com/heygen-com/hyperframes), [Remotion](https://www.remotion.dev/), [Manim](https://www.manim.community/), or PIL — spawned in parallel sub-agents, one per animation.
- **Self-evaluates the rendered output** at every cut boundary before showing you anything.
- **Self-heals on failure.** Every helper that touches ffmpeg captures real stderr, auto-clamps out-of-range cut times before they crash, retries past bad grade filters, and falls back to re-encode when `-c copy` concat refuses heterogeneous segments. On any unrecoverable error, `render.py` prints the exact `python helpers/doctor.py --fix` command to run — doctor strips JSON BOMs, clamps EDL bounds, drops missing overlay/subtitle refs, snaps `fit` typos, and surveys every backend, with `.bak` backups before any rewrite. Designed so an agent picking this up cold can always fix itself once and re-try.
- **Persists session memory** in `project.md` so next week's session picks up where you left off.

### Make a vertical 3-minute TikTok in one sentence

```
edit these into a tall 3-minute TikTok, cut the umms, big captions
```

The agent confirms in one sentence ("OK — making a 3-minute tall video for TikTok, cutting umms, big captions, no music. Yes?"), waits for "go", then produces `edit/final.mp4` at 1080×1920.

### Or one-shot it from the CLI once you have an EDL

```bash
# vertical with subject-aware dynamic crop (default --crop-mode auto)
python helpers/render.py edit/edl.json -o edit/final.mp4 --platform tiktok --build-subtitles

# moving subjects, force per-frame tracking
python helpers/render.py edit/edl.json -o edit/final.mp4 --platform tiktok --crop-mode track

# vertical-into-horizontal with a soft blurred background instead of bars
python helpers/render.py edit/edl.json -o edit/final.mp4 --platform youtube --fit blur --blur-sigma 32
```

### Don't have an agent? Use the wizard

```bash
python helpers/wizard.py
```

Asks the same toddler-mode questions in your terminal (one A/B at a time), then prints the exact `transcribe_batch` → hand-edit-EDL → `render.py` commands to run. `--no-prompt` accepts all defaults for shell scripts.

## Setup prompt

Paste into Claude Code, Codex, Hermes, Openclaw, or any agent with shell access:

```text
Set up https://github.com/carterisanartist/video-use for me.

Read install.md first to install this repo, wire up ffmpeg, register the skill with whichever agent you're running under, and confirm the default Whisper backend imports cleanly. If I'm on Apple Silicon (M1/M2/M3/M4), also install the [mac] extra so mlx-whisper becomes the auto-picked fast path. No API keys are required for the default flow. If I tell you I want speaker diarization, walk me through the HuggingFace token setup. Then read SKILL.md for daily usage, and always read helpers/ because that's where the editing scripts live. After install, don't transcribe anything on your own — just tell me it's ready and wait for me to drop footage into a folder.
```

The agent handles the clone, dependencies, skill registration, and (only if you ask) the optional HuggingFace token for diarization.

Then point your agent at a folder of raw takes:

```bash
cd /path/to/your/videos
claude    # or codex, hermes, etc.
```

And in the session:

> edit these into a launch video

It inventories the sources, proposes a strategy, waits for your OK, then produces `edit/final.mp4` next to your sources. All outputs live in `<videos_dir>/edit/` — the skill directory stays clean.

## Manual install

```bash
# 1. Clone and symlink into your agent's skills directory
git clone https://github.com/carterisanartist/video-use ~/Developer/video-use-whisper
ln -sfn ~/Developer/video-use-whisper ~/.claude/skills/video-use        # Claude Code
# ln -sfn ~/Developer/video-use-whisper ~/.codex/skills/video-use       # Codex

# 2. Install deps
cd ~/Developer/video-use-whisper
uv sync                         # or: pip install -e .
brew install ffmpeg             # required (apt/pacman/winget on other OSes)
brew install yt-dlp             # optional, for downloading online sources

# 3. (Apple Silicon only) Install the native MLX backend for ~3-5x speedup
pip install -e '.[mac]'
# After this, transcribe.py auto-picks mlx-whisper without any extra flag.

# 4. (Optional) Speaker diarization (works with MPS on Apple Silicon)
pip install -e '.[diarize]'
# Then accept https://huggingface.co/pyannote/speaker-diarization-3.1 and add:
echo "HUGGINGFACE_TOKEN=hf_..." >> .env

# 5. (Optional) Hosted backend
pip install -e '.[openai]'
echo "OPENAI_API_KEY=sk-..." >> .env

# 6. (Recommended for vertical) Subject-aware dynamic crop
pip install -e '.[crop]'
# Once installed, --crop-mode auto (the default) starts following the
# speaker's face whenever you convert landscape sources to vertical.
# Without this, crop falls back to a static center crop.

# 7. Verify everything is wired up
python helpers/doctor.py
# Surveys Python / ffmpeg / libass, every Whisper backend, opencv,
# pyannote + HUGGINGFACE_TOKEN. Run with --fix to auto-resolve common
# issues (BOM bytes in JSON, out-of-range EDL cuts, missing overlay
# refs, fit-mode typos like "cover" -> "crop").
```

The first time `transcribe.py` runs, the chosen backend downloads model weights to `~/.cache/huggingface/hub/`. `large-v3` is ~3 GB; `tiny` is ~150 MB. MLX models live under `mlx-community/whisper-*-mlx`; everything else is the standard `openai/whisper-*` repos.

## Backend cheat sheet

| Backend | Install | Best on | Speed | Diarization | Cost |
|---|---|---|---|---|---|
| `faster-whisper` (default) | included | Linux/Windows + NVIDIA, Intel Mac | ~10–20× realtime (CUDA), ~1× (CPU) | optional via pyannote | free |
| `mlx` (auto on Apple Silicon) | `pip install -e '.[mac]'` | Apple Silicon (M1/M2/M3/M4) | ~3–5× realtime (`large-v3` on M2 Pro) | optional via pyannote (MPS) | free |
| `openai` (cloud) | `pip install -e '.[openai]'` | no local GPU | network-bound | optional via pyannote | $0.006/min |
| `whisper` (reference) | `pip install -e '.[whisper]'` | parity / MPS fallback | ~0.3× realtime CPU, ~1–2× MPS, ~3–5× CUDA | optional via pyannote | free |

Pick the model size with `--model tiny|base|small|medium|large-v2|large-v3`. For mlx use the matching `mlx-community/whisper-*-mlx` repo id. `large-v3` is what you ship; `medium` is the reasonable speed/quality tradeoff on CPU; `tiny` is only for install smoke tests.

The smart default (`--backend auto`, which is what runs when you don't pass `--backend`):

- Apple Silicon + `mlx-whisper` installed → `mlx`
- everything else → `faster-whisper`

## How it works

The LLM never watches the video. It **reads** it — through two layers that together give it everything it needs to cut with word-boundary precision.

<p align="center">
  <img src="static/timeline-view.svg" alt="timeline_view composite — filmstrip + speaker track + waveform + word labels + silence-gap cut candidates" width="100%">
</p>

**Layer 1 — Audio transcript (always loaded).** One Whisper call per source gives word-level timestamps. With pyannote enabled, you also get speaker diarization. All takes pack into a single ~12KB `takes_packed.md` — the LLM's primary reading view.

```
## C0103  (duration: 43.0s, 8 phrases)
  [002.52-005.36] S0 Ninety percent of what a web agent does is completely wasted.
  [006.08-006.74] S0 We fixed this.
```

**Layer 2 — Visual composite (on demand).** `timeline_view` produces a filmstrip + waveform + word labels PNG for any time range. Called only at decision points — ambiguous pauses, retake comparisons, cut-point sanity checks.

> Naive approach: 30,000 frames × 1,500 tokens = **45M tokens of noise**.
> Video Use: **12KB text + a handful of PNGs**.

Same idea as browser-use giving an LLM a structured DOM instead of a screenshot — but for video.

## Pipeline

```
Transcribe (Whisper) ──> Pack ──> LLM Reasons ──> EDL ──> Render ──> Self-Eval
                                                                       │
                                                                       └─ issue? fix + re-render (max 3)
```

The self-eval loop runs `timeline_view` on the _rendered output_ at every cut boundary — catches visual jumps, audio pops, hidden subtitles. You see the preview only after it passes.

## Design principles

1. **Text + on-demand visuals.** No frame-dumping. The transcript is the surface.
2. **Audio is primary, visuals follow.** Cuts come from speech boundaries and silence gaps.
3. **Ask → confirm → execute → self-eval → persist.** Never touch the cut without strategy approval.
4. **Zero assumptions about content type.** Look, ask, then edit.
5. **12 hard rules, artistic freedom elsewhere.** Production-correctness is non-negotiable. Taste isn't.
6. **Local first.** Default backend has no API key, no network round-trip, no per-minute cost.

See [`SKILL.md`](./SKILL.md) for the full production rules and editing craft.

## Credits

Original `video-use` by [browser-use](https://github.com/browser-use/video-use). This fork swaps the transcription backend; the rest of the architecture, helpers, and SKILL.md craft are theirs. Whisper is from [OpenAI](https://github.com/openai/whisper); the fast local runtime is [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper); diarization is [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio).
