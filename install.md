---
name: video-use-install
description: Install video-use-whisper into the current agent (Claude Code, Codex, Hermes, Openclaw, etc.) and wire up ffmpeg + the Whisper backend so the user can start editing immediately. No ElevenLabs key required.
---

# video-use-whisper install

Use this file only for first-time install or reconnect. For daily editing, read `SKILL.md`. Always read `helpers/` — that's where the scripts live.

## What you're doing

You're setting up a conversation-driven video editor for the user. After install, the user drops raw footage into any folder, runs their agent (`claude`, `codex`, etc.) there, and says "edit these into a launch video." You do the rest by reading `SKILL.md`.

Three things must exist on this machine:

1. The `video-use-whisper` repo cloned somewhere stable.
2. `ffmpeg` on `$PATH` (plus optional `yt-dlp` for online sources).
3. A working Whisper backend. Default is `faster-whisper` — local, free, no API key. See step 5 if you want diarization or a hosted backend.

And one thing must be true about the current agent:

4. It can discover `SKILL.md` — either via a global skills directory (`~/.claude/skills/`, `~/.codex/skills/`) or via a `CLAUDE.md` / system-prompt import.

## Install prompt contract

- Do everything yourself. Only ask the user for things you cannot generate — e.g. confirmation before `brew install`, or a HuggingFace token if they want diarization.
- Prefer a stable clone path like `~/Developer/video-use-whisper` (not `/tmp`, not `~/Downloads`).
- The skill references helpers by bare name (`transcribe.py`, `render.py`). That works because SKILL.md and `helpers/` ship together — keep them as siblings when you register the skill.
- After install, verify by running one real command against one real file. Don't declare success on file-existence checks alone.

## Steps

### 1. Clone to a stable path

```bash
test -d ~/Developer/video-use-whisper || git clone https://github.com/carterisanartist/video-use ~/Developer/video-use-whisper
cd ~/Developer/video-use-whisper
```

If the repo is already there, `git pull --ff-only` and continue.

### 2. Install Python deps

```bash
# Prefer uv if available; fall back to pip.
command -v uv >/dev/null && uv sync || pip install -e .
```

`pyproject.toml` lists `requests`, `librosa`, `matplotlib`, `pillow`, `numpy`, and `faster-whisper`. The first transcription will download the Whisper model weights (~3 GB for `large-v3`, ~150 MB for `tiny`) into the HuggingFace cache. No console scripts — helpers are invoked directly as `python helpers/<name>.py`.

Optional extras:

```bash
# Speaker diarization (pyannote.audio + torch) — heavy, only if needed.
pip install -e '.[diarize]'

# Hosted OpenAI Whisper backend.
pip install -e '.[openai]'

# Reference openai-whisper package (slower than faster-whisper).
pip install -e '.[whisper]'

# All optional extras at once.
pip install -e '.[all]'
```

### 3. Install ffmpeg (+ optional yt-dlp)

`ffmpeg` and `ffprobe` are hard requirements. `yt-dlp` is only needed if the user wants to pull sources from URLs. Animation engines such as HyperFrames, Remotion, and Manim are installed lazily the first time a project actually needs them.

```bash
# macOS
command -v ffmpeg >/dev/null || brew install ffmpeg
command -v yt-dlp >/dev/null || brew install yt-dlp     # optional

# Debian / Ubuntu
# sudo apt-get update && sudo apt-get install -y ffmpeg
# pip install yt-dlp

# Arch
# sudo pacman -S ffmpeg yt-dlp

# Windows
# winget install ffmpeg
# winget install yt-dlp.yt-dlp     # optional
```

If `brew` / `apt` / `pacman` requires a sudo prompt, tell the user the exact command and wait. Do not invent a password.

### 4. Register the skill with the current agent

Figure out which agent you are running under, and register once. A symlink of the whole repo directory is the right shape — `helpers/` needs to sit next to `SKILL.md`.

- **Claude Code** (`~/.claude/` present):

    ```bash
    mkdir -p ~/.claude/skills
    ln -sfn ~/Developer/video-use-whisper ~/.claude/skills/video-use
    ```

- **Codex** (`$CODEX_HOME` set, or `~/.codex/` present):

    ```bash
    mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
    ln -sfn ~/Developer/video-use-whisper "${CODEX_HOME:-$HOME/.codex}/skills/video-use"
    ```

- **Hermes / Openclaw / another agent with a skills directory**: symlink `~/Developer/video-use-whisper` into that agent's skills directory under the name `video-use`. If the agent has no skills directory, add a line to its system prompt / config pointing at `~/Developer/video-use-whisper/SKILL.md` (e.g. an `@~/Developer/video-use-whisper/SKILL.md` import in a `CLAUDE.md`-equivalent).

If you can't tell which agent you're in, ask the user once: "which agent am I running under — Claude Code, Codex, or something else?" Then pick the right target.

### 5. Pick a Whisper backend (optional configuration)

The default backend (`faster-whisper`) needs **no setup at all** — it runs locally and has no API key. You can stop here and the pipeline works.

Three optional knobs you can offer the user:

#### a) Speaker diarization (recommended for multi-speaker takes)

Single-speaker takes work fine without this — every word just gets `speaker_0`. For interviews, panel discussions, or anything with two or more voices you need pyannote.

1. Install the extra (already covered in step 2):
    ```bash
    pip install -e '.[diarize]'
    ```
2. The user signs up at <https://huggingface.co>, accepts the license at <https://huggingface.co/pyannote/speaker-diarization-3.1>, and creates a token at <https://huggingface.co/settings/tokens>.
3. Write the token to `.env`:
    ```bash
    cp .env.example .env 2>/dev/null || true
    # Append (don't overwrite the file if other keys already live there).
    grep -q '^HUGGINGFACE_TOKEN=' .env || printf 'HUGGINGFACE_TOKEN=%s\n' "$TOKEN" >> .env
    chmod 600 .env
    ```
   Never echo the token back. Never commit `.env`.

If `HUGGINGFACE_TOKEN` is missing, `transcribe.py` silently skips diarization and tags every word with `speaker_0` — the rest of the pipeline keeps working.

#### b) Hosted Whisper backend (OpenAI)

Useful if the user has no local GPU and doesn't want to wait for CPU transcription. Word timestamps work the same; diarization still requires pyannote.

1. Install the extra: `pip install -e '.[openai]'`
2. Add `OPENAI_API_KEY=…` to `.env` (key from <https://platform.openai.com/api-keys>).
3. Run with `--backend openai`.

#### c) GPU acceleration for `faster-whisper`

If the machine has CUDA, `faster-whisper` auto-detects it. Force it explicitly with `--device cuda --compute-type float16`. On Apple Silicon, CPU with `int8` is the right default — there is no Metal backend in CTranslate2 yet.

### 6. Verify end-to-end

Run one cheap thing to prove the pipeline is wired up:

```bash
python ~/Developer/video-use-whisper/helpers/timeline_view.py --help >/dev/null && echo "helpers OK"
python -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"
ffprobe -version | head -1
```

A full transcription test is optional at install time — for `large-v3` it triggers a multi-GB model download. Better to wait until the user hands you their first clip; if they do, run with `--model tiny` for the very first run so the download is quick (~150 MB) and only swap up to `large-v3` once they care about quality.

### 7. Hand off

Tell the user, in one short message:

- Where the skill is installed (`~/Developer/video-use-whisper`).
- That they should `cd` into their footage folder and start their agent there (e.g. `claude`).
- That a good first message is: *"edit these into a launch video"* or *"inventory these takes and propose a strategy."*
- That all outputs land in `<videos_dir>/edit/` — the repo stays clean.
- Whether diarization is enabled (depends on whether you set up a HuggingFace token).

## Keeping the skill current

- `cd ~/Developer/video-use-whisper && git pull --ff-only` pulls the latest code. The symlink auto-picks it up on the next run.
- If `pyproject.toml` changed deps, re-run `uv sync` / `pip install -e .` after pulling.

## Cold-start reminders

- Symlink the **whole directory**, not just `SKILL.md`. The helpers need to sit next to it.
- The default Whisper backend has **no API key**. You should not block install on credentials unless the user asked for diarization or the hosted OpenAI backend.
- The first Whisper invocation downloads model weights into `~/.cache/huggingface/hub/`. This is one-time.
- `ffmpeg` from static builds works fine. Any modern (>= 4.x) build is enough.
- `yt-dlp` is optional. Don't block install on it; install lazily the first time a user asks to pull from a URL.
- Node.js/npm are only needed for HyperFrames or Remotion slots. HyperFrames currently requires Node.js 22+.
- HyperFrames, Remotion, and Manim are optional animation engines. Don't install or prefer one globally during setup; pick the engine per animation slot in `SKILL.md`.
- Never run a full-quality transcription as part of install verification unless the user explicitly asks — model downloads take time. Use `--model tiny` for any sanity-check run.
- If the user is on Linux without a package manager Claude recognizes, print the manual `ffmpeg` install URL and wait rather than guessing.
