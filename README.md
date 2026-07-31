# CarlessWhisper — Charles Leclerc Detection Project

A camera-based system that plays "Careless Whisper" whenever Charles Leclerc is visible on
screen. See `/Users/tanishmisra/.claude/plans/here-s-a-handoff-for-keen-ocean.md` for the full
handoff and current-stage plan.

v1 scope is face-only, built entirely from Tanish's own footage (no web crawling).

## Setup

```
uv sync
```

## Video parser

Extracts frames from a source video and crops detected faces (blur/size filtered, deduped):

```
uv run scripts/video_parser.py data/raw_video/<clip>.mp4 --fps 1
```

Crops are written to `data/crops/face/<clip_name>/`.

## Enroll reference faces

Builds a reference embedding set from the video crops, for live matching:

```
uv run scripts/enroll_faces.py
```

Writes `data/leclerc_embeddings.npy`. Takes about a minute for ~1,000 crops.

## Live trigger

Watches the camera and plays `careless-whisper.m4a` from the top whenever a matching face is on
screen, stopping a few frames after it leaves:

```
uv run scripts/carless_whisper.py [--camera 0] [--threshold 0.4] [--max-misses 3]
```

Requires `data/leclerc_embeddings.npy` (run enrollment first) and `careless-whisper.m4a` in the
repo root. Press `q` or Esc to quit. The overlay shows a live similarity score per detected
face — use it to tune `--threshold` against your own footage.
