"""Download category reference images via duckduckgo_search (no API key required).

Usage:
    uv run scripts/crawl_ddg.py --category face --max-results 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ddgs import DDGS

sys.path.insert(0, str(Path(__file__).parent))
from common.categories import CATEGORY_QUERIES
from common.download import fetch_image_array
from common.image_filters import DedupTracker, passes_filters


def run(
    category: str,
    max_results: int = 100,
    out_root: Path = Path("data/web"),
    min_size: int = 60,
    blur_thresh: float = 100.0,
) -> int:
    out_dir = out_root / category
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup = DedupTracker()
    kept = 0
    idx = 0

    for query in CATEGORY_QUERIES[category]:
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
    parser.add_argument("--category", required=True, choices=sorted(CATEGORY_QUERIES))
    parser.add_argument("--max-results", type=int, default=100, help="max image results per query")
    args = parser.parse_args()

    kept = run(args.category, max_results=args.max_results)
    print(f"[ddg] {args.category}: kept {kept} images")


if __name__ == "__main__":
    main()
