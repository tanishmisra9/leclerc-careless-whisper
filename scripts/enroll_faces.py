"""Build a reference embedding set from face crops, for live cosine-similarity matching.

Usage:
    uv run scripts/enroll_faces.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# The existing crops are tight face-only boxes; SCRFD finds almost nothing in them
# undoctored (measured 9/60 on a sample) but detects reliably once padded outward.
PAD_FRACTION = 0.5

# Measured on a 267-crop sample: genuine Leclerc crops sit at similarity >= 0.48 to the
# centroid, with one contaminant (a bystander's face) at 0.01. 0.3 splits them with margin.
OUTLIER_THRESH = 0.3


def pad(img: np.ndarray, fraction: float = PAD_FRACTION) -> np.ndarray:
    h, w = img.shape[:2]
    dy, dx = int(h * fraction), int(w * fraction)
    return cv2.copyMakeBorder(img, dy, dy, dx, dx, cv2.BORDER_REPLICATE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops-dir", type=Path, default=Path("data/crops/face/charles"))
    parser.add_argument("--out", type=Path, default=Path("data/leclerc_embeddings.npy"))
    args = parser.parse_args()

    if not args.crops_dir.is_dir():
        sys.exit(f"Crops directory not found: {args.crops_dir}")

    crop_paths = sorted(args.crops_dir.glob("*.png"))
    if not crop_paths:
        sys.exit(f"No .png crops found in {args.crops_dir}")

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    embeddings = []
    names = []
    no_detect = 0

    for crop_path in crop_paths:
        img = cv2.imread(str(crop_path))
        if img is None:
            no_detect += 1
            continue
        faces = app.get(pad(img))
        if not faces:
            no_detect += 1
            continue
        best = max(faces, key=lambda f: f.det_score)
        embeddings.append(best.normed_embedding)
        names.append(crop_path.name)

    if not embeddings:
        sys.exit("No faces detected in any crop after padding — nothing to enroll.")

    all_embeddings = np.stack(embeddings).astype(np.float32)
    centroid = all_embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    similarities = all_embeddings @ centroid

    keep_mask = similarities >= OUTLIER_THRESH
    dropped = [(names[i], float(similarities[i])) for i in range(len(names)) if not keep_mask[i]]
    if dropped:
        print(f"Dropped {len(dropped)} outlier(s) (similarity to centroid < {OUTLIER_THRESH}):")
        for name, sim in dropped:
            print(f"  {name}: {sim:.3f}")

    kept = all_embeddings[keep_mask]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, kept)

    print(
        f"crops={len(crop_paths)} embedded={len(embeddings)} no_detect={no_detect} "
        f"dropped={len(dropped)} kept={len(kept)} -> {args.out}"
    )


if __name__ == "__main__":
    main()
