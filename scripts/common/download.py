"""Shared HTTP image fetch used by crawl_ddg.py and crawl_wikimedia.py."""

from __future__ import annotations

import time

import cv2
import numpy as np
import requests

USER_AGENT = "CarlessWhisper/0.1 (data collection for personal project)"


def fetch_image_array(url: str, timeout: float = 10.0, max_retries: int = 2) -> np.ndarray | None:
    """Downloads url and decodes it as a BGR image array, or returns None on any failure.

    Retries once on 429 (respecting Retry-After, capped at 30s) since hosts like
    Wikimedia's upload CDN rate-limit aggressively.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 429 and attempt < max_retries - 1:
                wait = min(int(resp.headers.get("Retry-After", 5)), 30)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                return None
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except (requests.RequestException, cv2.error):
            return None
    return None
