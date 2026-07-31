"""Download category reference images via the Wikimedia Commons API (no key required).

Usage:
    uv run scripts/crawl_wikimedia.py --category face --max-results 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import requests

sys.path.insert(0, str(Path(__file__).parent))
from common.categories import CATEGORY_QUERIES, WIKIMEDIA_CATEGORY_QUERIES
from common.download import USER_AGENT, fetch_image_array
from common.image_filters import DedupTracker, passes_filters

API_URL = "https://commons.wikimedia.org/w/api.php"


def _get(params: dict, max_retries: int = 3) -> requests.Response:
    """GET with Retry-After-aware backoff, since Wikimedia's edge rate-limits aggressively."""
    for attempt in range(max_retries):
        resp = requests.get(API_URL, params=params, timeout=10, headers={"User-Agent": USER_AGENT})
        if resp.status_code == 429 and attempt < max_retries - 1:
            wait = min(int(resp.headers.get("Retry-After", 5)), 30)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def search_file_titles(query: str, limit: int) -> list[str]:
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srnamespace": 6,  # File: namespace
        "srsearch": query,
        "srlimit": limit,
    }
    resp = _get(params)
    data = resp.json()
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def resolve_image_url(title: str) -> str | None:
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url",
    }
    resp = _get(params)
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        infos = page.get("imageinfo")
        if infos:
            return infos[0].get("url")
    return None


def run(
    category: str,
    max_results: int = 50,
    out_root: Path = Path("data/web"),
    min_size: int = 60,
    blur_thresh: float = 100.0,
) -> int:
    out_dir = out_root / category
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup = DedupTracker()
    kept = 0
    idx = 0

    for query in WIKIMEDIA_CATEGORY_QUERIES[category]:
        try:
            titles = search_file_titles(query, max_results)
        except requests.RequestException as e:
            print(f"  [wikimedia] search failed: {query!r}: {e}")
            continue

        for title in titles:
            try:
                url = resolve_image_url(title)
            except requests.RequestException:
                continue
            if not url:
                continue
            img = fetch_image_array(url)
            if img is None:
                continue
            if not passes_filters(img, dedup, min_size, blur_thresh):
                continue
            idx += 1
            out_path = out_dir / f"wiki_{idx:05d}.jpg"
            cv2.imwrite(str(out_path), img)
            kept += 1

    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, choices=sorted(CATEGORY_QUERIES))
    parser.add_argument("--max-results", type=int, default=50, help="max file search results per query")
    args = parser.parse_args()

    kept = run(args.category, max_results=args.max_results)
    print(f"[wikimedia] {args.category}: kept {kept} images")


if __name__ == "__main__":
    main()
