"""Download Leclerc reference images via duckduckgo_search (no API key required), targeting
lighting, profile-angle, and helmet-on diversity that a single interview video can't supply.

Usage:
    uv run scripts/crawl_ddg.py --max-results 40
    uv run scripts/crawl_ddg.py --profile-only --max-results 60
    uv run scripts/crawl_ddg.py --helmet --max-results 60
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

# Out-of-car helmet-on shots specifically (grid walk, paddock, portraits) -- explicitly avoiding
# onboard/cockpit framing, which is a different (excluded) context. These get fed into the same
# face-embedding pipeline as everything else: enroll_faces.py only keeps images where InsightFace
# still finds a face (visor up / partially visible), which a verified sample showed is possible
# even with the helmet on (ddg_00165.jpg: det_score 0.75, scored 0.324 against the pre-helmet
# reference set -- just under the 0.4 live threshold, hence this crawl).
HELMET_QUERIES = [
    "Charles Leclerc helmet grid walk",
    "Charles Leclerc helmet pit lane",
    "Charles Leclerc putting on helmet",
    "Charles Leclerc helmet walking",
    "Charles Leclerc helmet portrait",
    "Charles Leclerc helmet visor up",
    "Charles Leclerc helmet profile",
    "Charles Leclerc pre race helmet",
    "Charles Leclerc helmet paddock",
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
    parser.add_argument("--out", type=Path, default=None, help="default: data/web/face, or data/web/helmet with --helmet")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--profile-only", action="store_true", help="only run the profile-angle-targeted queries"
    )
    mode.add_argument("--helmet", action="store_true", help="only run the helmet-on queries")
    args = parser.parse_args()

    if args.helmet:
        queries, default_out = HELMET_QUERIES, Path("data/web/helmet")
    elif args.profile_only:
        queries, default_out = PROFILE_QUERIES, Path("data/web/face")
    else:
        queries, default_out = QUERIES, Path("data/web/face")
    out_dir = args.out if args.out is not None else default_out

    kept = run(queries, max_results=args.max_results, out_dir=out_dir)
    removed = dedup_directory(out_dir)
    print(f"[ddg] kept {kept} new images, removed {removed} cross-run duplicates -> {out_dir}")


if __name__ == "__main__":
    main()
