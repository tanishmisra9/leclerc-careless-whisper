# CarlessWhisper — Charles Leclerc Detection Project

A camera-based system that plays "Careless Whisper" whenever Charles Leclerc is visible on
screen. See `/Users/tanishmisra/.claude/plans/here-s-a-handoff-for-keen-ocean.md` for the full
handoff and current-stage plan.

v1 scope is face-only, primarily Tanish's own footage plus a small set of web images pulled for
lighting/angle diversity that one video can't supply.

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

## Diversity image crawl (optional)

Pulls extra reference images targeting lighting and angle diversity the video alone lacks (night
races, rain, side profiles, etc.):

```
uv run scripts/crawl_ddg.py --max-results 40
uv run scripts/crawl_ddg.py --profile-only --max-results 60   # just the profile-angle queries
```

Writes to `data/web/face/`, deduped against everything already there (safe to re-run).

## Measure pose diversity

Checks how much of a face-image folder actually sits at off-axis head angles (yaw), rather than
trusting that a "side profile" search query returned side profiles:

```
uv run scripts/measure_pose_diversity.py data/web/face
```

## Enroll reference faces

Builds a reference embedding set from the video crops (and web images, if crawled):

```
uv run scripts/enroll_faces.py
```

Writes `data/leclerc_embeddings.npy`. Takes roughly a minute per ~1,000 images.

## Live trigger

Watches the camera and plays `careless-whisper.m4a` from the top whenever a matching face is on
screen, stopping a few frames after it leaves:

```
uv run scripts/carless_whisper.py [--camera 0] [--threshold 0.4] [--max-misses 3]
```

Requires `data/leclerc_embeddings.npy` (run enrollment first) and `careless-whisper.m4a` in the
repo root. Press `q` or Esc to quit. The overlay shows a live similarity score per detected
face — use it to tune `--threshold` against your own footage.
