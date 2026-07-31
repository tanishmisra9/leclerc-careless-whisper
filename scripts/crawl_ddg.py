"""Download Leclerc reference images via duckduckgo_search (no API key required), targeting
lighting and profile-angle diversity that a single interview video can't supply.

Usage:
    uv run scripts/crawl_ddg.py --max-results 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ddgs import DDGS

sys.path.insert(0, str(Path(__file__).parent))
from common.download import fetch_image_array
from common.image_filters import DedupTracker, passes_filters

# Existing reference set is 1,041 embeddings from one interview video: one lighting setup,
# mostly frontal/near-frontal. These queries target the two gaps that surfaced in live testing:
# lighting (night/rain/indoor vs. the interview's studio lighting) and profile angle.
QUERIES = [
    "Charles Leclerc night race",
    "Charles Leclerc Singapore Grand Prix",
    "Charles Leclerc rain race",
    "Charles Leclerc garage",
    "Charles Leclerc side profile",
    "Charles Leclerc profile photo",
    "Charles Leclerc walking paddock",
    "Charles Leclerc sunset",
    "Charles Leclerc overcast",
    "Charles Leclerc indoor press conference",
]


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
    idx = 0

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
    args = parser.parse_args()

    kept = run(QUERIES, max_results=args.max_results, out_dir=args.out)
    print(f"[ddg] kept {kept} images -> {args.out}")


if __name__ == "__main__":
    main()
