"""Measure head-pose (yaw) coverage across a folder of face images, to check whether a crawl
actually added profile-angle diversity rather than trusting search-query text.

Usage:
    uv run scripts/measure_pose_diversity.py data/web/face
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PAD_FRACTION = 0.5
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg")

# Bucket edges in degrees of |yaw|. face.pose is [pitch, yaw, roll] (empirically confirmed:
# yaw is the component that grows as a head turns left/right, tracks with visually "more profile").
BUCKETS = [(0, 15, "frontal"), (15, 35, "turned"), (35, 55, "3/4"), (55, 999, "profile")]


def pad(img: np.ndarray, fraction: float = PAD_FRACTION) -> np.ndarray:
    h, w = img.shape[:2]
    dy, dx = int(h * fraction), int(w * fraction)
    return cv2.copyMakeBorder(img, dy, dy, dx, dx, cv2.BORDER_REPLICATE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--top-n", type=int, default=5, help="show the N strongest-profile images found")
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        sys.exit(f"Not a directory: {args.image_dir}")

    paths = sorted(p for pattern in IMAGE_EXTENSIONS for p in args.image_dir.glob(pattern))
    if not paths:
        sys.exit(f"No images found in {args.image_dir}")

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(providers=["CPUExecutionProvider"])  # full modules -- pose needs landmark_3d_68
    app.prepare(ctx_id=-1, det_size=(640, 640))

    results = []
    no_detect = 0
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            no_detect += 1
            continue
        faces = app.get(pad(img))
        if not faces:
            no_detect += 1
            continue
        best = max(faces, key=lambda f: f.det_score)
        results.append((path, float(best.pose[1])))

    yaws = np.array([abs(y) for _, y in results])
    print(f"{args.image_dir}: {len(paths)} images, {len(results)} faces detected, {no_detect} no-detect")
    for lo, hi, label in BUCKETS:
        mask = (yaws >= lo) & (yaws < hi)
        print(f"  |yaw| in [{lo:3d},{hi:3d}) {label:8s}: {mask.sum():4d}  ({mask.mean()*100:5.1f}%)")
    if len(yaws):
        print(f"  max |yaw|: {yaws.max():.1f}")

    results.sort(key=lambda r: -abs(r[1]))
    print(f"\nTop {args.top_n} strongest-profile images:")
    for path, yaw in results[: args.top_n]:
        print(f"  {path}: yaw={yaw:.1f}")


if __name__ == "__main__":
    main()
