"""Download category reference images via icrawler (Google + Bing image search).

Usage:
    uv run scripts/crawl_icrawler.py --category face --max-num 50
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

sys.path.insert(0, str(Path(__file__).parent))
from common.categories import CATEGORY_QUERIES
from common.image_filters import DedupTracker, passes_filters

ENGINES = {"google": GoogleImageCrawler, "bing": BingImageCrawler}


def _crawl_one(engine_cls, query: str, max_num: int, tmp_dir: Path) -> None:
    crawler = engine_cls(storage={"root_dir": str(tmp_dir)}, log_level=40)
    crawler.crawl(keyword=query, max_num=max_num)


def run(
    category: str,
    max_num_per_query: int = 50,
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
        for engine_name, engine_cls in ENGINES.items():
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                try:
                    _crawl_one(engine_cls, query, max_num_per_query, tmp_dir)
                except Exception as e:
                    print(f"  [icrawler:{engine_name}] query failed: {query!r}: {e}")
                    continue
                for f in sorted(tmp_dir.iterdir()):
                    img = cv2.imread(str(f))
                    if img is None:
                        continue
                    if not passes_filters(img, dedup, min_size, blur_thresh):
                        continue
                    idx += 1
                    out_path = out_dir / f"icrawler_{engine_name}_{idx:05d}.jpg"
                    cv2.imwrite(str(out_path), img)
                    kept += 1

    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, choices=sorted(CATEGORY_QUERIES))
    parser.add_argument("--max-num", type=int, default=50, help="max images per (engine, query)")
    args = parser.parse_args()

    kept = run(args.category, max_num_per_query=args.max_num)
    print(f"[icrawler] {args.category}: kept {kept} images")


if __name__ == "__main__":
    main()
