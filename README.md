# CarlessWhisper — Charles Leclerc Detection Project

Data-acquisition tooling for a camera-based system that plays "Careless Whisper" whenever
Charles Leclerc is visible on screen. See
`/Users/tanishmisra/.claude/plans/here-s-a-handoff-for-keen-ocean.md` for the full handoff
and current-stage plan. This repo currently covers only data collection — no detection,
training, or playback logic yet.

v1 scope is face-only (no helmet/car/background categories yet).

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

## Web image crawler

Pulls face reference images via duckduckgo_search (`ddgs`). (icrawler/Google/Bing was dropped —
mostly irrelevant results, e.g. paintbrushes and historical figures also named Leclerc.
Wikimedia Commons was dropped too — very low yield for these queries.)

```
uv run scripts/crawl_all.py --per-source 100
```

Or run the source directly: `uv run scripts/crawl_ddg.py --category face --max-results 100`.

Output goes to `data/web/face/`, deduped.
