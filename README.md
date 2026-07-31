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
