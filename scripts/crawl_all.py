"""Run all three crawlers (icrawler, duckduckgo_search, Wikimedia Commons) for each category,
merging results into data/web/<category> and deduping across the combined pool.

Usage:
    uv run scripts/crawl_all.py                       # all categories, ~300 images/category
    uv run scripts/crawl_all.py --category face        # single category
    uv run scripts/crawl_all.py --per-source 10        # smoke test with a small pull
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import crawl_ddg
import crawl_icrawler
import crawl_wikimedia
from common.categories import CATEGORY_QUERIES
from common.image_filters import dedup_directory


def run_category(category: str, per_source: int, out_root: Path) -> None:
    print(f"=== {category} ===")
    icrawler_kept = crawl_icrawler.run(category, max_num_per_query=per_source, out_root=out_root)
    print(f"  icrawler: kept {icrawler_kept}")

    ddg_kept = crawl_ddg.run(category, max_results=per_source, out_root=out_root)
    print(f"  ddg: kept {ddg_kept}")

    wiki_kept = crawl_wikimedia.run(category, max_results=per_source, out_root=out_root)
    print(f"  wikimedia: kept {wiki_kept}")

    combined = icrawler_kept + ddg_kept + wiki_kept
    removed = dedup_directory(out_root / category)
    final = combined - removed
    print(f"  combined: {combined} -> {final} after cross-source dedup ({removed} removed)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=sorted(CATEGORY_QUERIES), help="single category, default: all")
    parser.add_argument(
        "--per-source",
        type=int,
        default=100,
        help="target images per source per category (~3 sources -> ~3x this per category)",
    )
    parser.add_argument("--out", type=Path, default=Path("data/web"))
    args = parser.parse_args()

    categories = [args.category] if args.category else sorted(CATEGORY_QUERIES)
    for category in categories:
        run_category(category, args.per_source, args.out)


if __name__ == "__main__":
    main()
