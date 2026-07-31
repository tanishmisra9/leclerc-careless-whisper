"""Shared image quality filtering and dedup, used by video_parser.py."""

from __future__ import annotations

import cv2
import imagehash
import numpy as np
from PIL import Image


def is_too_small(img: np.ndarray, min_size: int) -> bool:
    h, w = img.shape[:2]
    return min(h, w) < min_size


def is_too_blurry(img: np.ndarray, blur_thresh: float) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < blur_thresh


class DedupTracker:
    """Tracks perceptual hashes of kept images and flags near-duplicates."""

    def __init__(self, hamming_thresh: int = 5, window: int = 50):
        self.hamming_thresh = hamming_thresh
        self.window = window
        self._hashes: list[imagehash.ImageHash] = []

    def is_duplicate(self, img: np.ndarray) -> bool:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h = imagehash.phash(pil_img)
        for existing in self._hashes:
            if h - existing <= self.hamming_thresh:
                return True
        self._hashes.append(h)
        if len(self._hashes) > self.window:
            self._hashes.pop(0)
        return False
