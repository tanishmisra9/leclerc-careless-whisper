# CarlessWhisper

*Romanticizing Charles Leclerc*

A camera-based system that plays "Careless Whisper" the moment Charles Leclerc's face appears on
screen, and stops the instant he's gone. Point a camera at a screen playing his footage, or at him
directly, and it does the rest.

## How it works

1. **Video parser**: extracts and crops faces from source footage.
2. **Web crawler** (optional): pulls extra reference images for lighting, angle, and helmet-on
   diversity a single video can't supply.
3. **Enrollment**: builds a face embedding reference set from everything above.
4. **Live trigger**: watches a camera feed, matches faces against the reference set, and plays the
   audio the moment a match is found.

## Setup

```
uv sync
```

## Usage

### 1. Parse a video into face crops

```
uv run scripts/video_parser.py <path-to-video.mp4> --fps 1
```

Crops are written to `data/crops/face/<clip_name>/`.

### 2. Crawl extra reference images (optional)

```
uv run scripts/crawl_ddg.py --max-results 40
uv run scripts/crawl_ddg.py --profile-only --max-results 60
uv run scripts/crawl_ddg.py --helmet --max-results 50
```

Writes to `data/web/` (not tracked in this repo, see `.gitignore`).

### 3. Build the reference embeddings

```
uv run scripts/enroll_faces.py
```

Writes `data/leclerc_embeddings.npy` from everything under `data/crops/` and `data/web/`.

### 4. Run the live trigger

```
uv run whisper
```

Opens the camera, shows a live preview with match score and state, and plays
`careless-whisper.m4a` the instant a match is found. That audio file is copyrighted and not
included, add your own copy to the repo root before running this.

Options: `--camera`, `--threshold`, `--max-misses`, `--capture-width`, `--capture-height`.

## Future improvements

- **Fully occluded face detection.** Detection currently requires a face to be at least partially
  visible. A closed visor or a profile past roughly 90 degrees leaves nothing to detect. A
  CLIP-based whole-image approach was tried for this and abandoned, it could not reliably
  distinguish Leclerc from a teammate in visually similar gear.
- **Car recognition.** Identifying the car itself, not just the driver, was part of the original
  design and hasn't been started.
- **Portrait camera support.** Continuity Camera only delivers landscape video regardless of phone
  orientation, and there's currently no workaround for that.
- **Broader reference data.** More lighting conditions and camera angles would likely improve match
  reliability further.
