# CarlessWhisper — Charles Leclerc Detection Project

Data-acquisition tooling for a camera-based system that plays "Careless Whisper" whenever
Charles Leclerc (face, helmet, or car #16) is visible on screen. See
`/Users/tanishmisra/.claude/plans/here-s-a-handoff-for-keen-ocean.md` for the full handoff
and current-stage plan. This repo currently covers only data collection — no detection,
training, or playback logic yet.

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

## Web image crawlers

Pull reference images per category (`face`, `helmet`, `car_positive`, `car_negative`,
`background_negative`) from icrawler (Google/Bing), duckduckgo_search (`ddgs`), and
Wikimedia Commons:

```
uv run scripts/crawl_all.py --category face --per-source 100
uv run scripts/crawl_all.py                          # all categories
```

Or run a single source directly: `crawl_icrawler.py`, `crawl_ddg.py`, `crawl_wikimedia.py`
(each takes `--category` and a per-source count flag).

Output goes to `data/web/<category>/`, deduped across sources.

Note: Wikimedia Commons' API/CDN rate-limits aggressively from shared/datacenter IPs; the
wikimedia crawler retries with backoff but may need to be run from a normal residential
connection to get full throughput.
