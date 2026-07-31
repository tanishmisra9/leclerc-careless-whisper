"""Download Leclerc reference images via duckduckgo_search (no API key required), targeting
lighting and profile-angle diversity that a single interview video can't supply.

Usage:
    uv run scripts/crawl_ddg.py --max-results 40
    uv run scripts/crawl_ddg.py --profile-only --max-results 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ddgs import DDGS

sys.path.insert(0, str(Path(__file__).parent))
from common.download import fetch_image_array
from common.image_filters import DedupTracker, dedup_directory, passes_filters

# Measured via head-pose analysis (InsightFace face.pose[1] = yaw): the 1,066 video crops
# max out at 52 degrees yaw with zero true profile shots (|yaw| >= 55). These queries target
# that specific gap. "side profile"/"profile photo" already proved effective (16/296 web images
# landed at |yaw| >= 55, max 80.4) -- the rest below are more phrasings of the same intent.
PROFILE_QUERIES = [
    "Charles Leclerc side profile",
    "Charles Leclerc profile photo",
    "Charles Leclerc profile portrait",
    "Charles Leclerc side view",
    "Charles Leclerc turned head",
    "Charles Leclerc side face",
    "Charles Leclerc profile closeup",
]

OTHER_QUERIES = [
    "Charles Leclerc night race",
    "Charles Leclerc Singapore Grand Prix",
    "Charles Leclerc rain race",
    "Charles Leclerc garage",
    "Charles Leclerc walking paddock",
    "Charles Leclerc sunset",
    "Charles Leclerc overcast",
    "Charles Leclerc indoor press conference",
]

QUERIES = OTHER_QUERIES + PROFILE_QUERIES


def next_start_idx(out_dir: Path) -> int:
    """Highest existing ddg_NNNNN.jpg index + 1, so re-running doesn't overwrite prior downloads."""
    existing = out_dir.glob("ddg_*.jpg")
    indices = [int(p.stem.split("_")[1]) for p in existing]
    return max(indices, default=0) + 1


def run(
    queries: list[str],
    max_results: int = 40,
    out_dir: Path = Path("data/web/face"),
    min_size: int = 60,
    blur_thresh: float = 100.0,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup = DedupTracker()
    kept = 0
    idx = next_start_idx(out_dir) - 1

    for query in queries:
        try:
            with DDGS() as ddgs:
                results = ddgs.images(query=query, max_results=max_results)
        except Exception as e:
            print(f"  [ddg] query failed: {query!r}: {e}")
            continue

        for r in results:
            url = r.get("image")
            if not url:
                continue
            img = fetch_image_array(url)
            if img is None:
                continue
            if not passes_filters(img, dedup, min_size, blur_thresh):
                continue
            idx += 1
            out_path = out_dir / f"ddg_{idx:05d}.jpg"
            cv2.imwrite(str(out_path), img)
            kept += 1

    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-results", type=int, default=40, help="max image results per query")
    parser.add_argument("--out", type=Path, default=Path("data/web/face"))
    parser.add_argument(
        "--profile-only", action="store_true", help="only run the profile-angle-targeted queries"
    )
    args = parser.parse_args()

    queries = PROFILE_QUERIES if args.profile_only else QUERIES
    kept = run(queries, max_results=args.max_results, out_dir=args.out)
    removed = dedup_directory(args.out)
    print(f"[ddg] kept {kept} new images, removed {removed} cross-run duplicates -> {args.out}")


if __name__ == "__main__":
    main()
