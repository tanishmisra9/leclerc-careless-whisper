"""Build a reference embedding set from face crops, for live cosine-similarity matching.

Usage:
    uv run scripts/enroll_faces.py
    uv run scripts/enroll_faces.py --crops-dir data/crops/face/charles data/web/face
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

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg")


def pad(img: np.ndarray, fraction: float = PAD_FRACTION) -> np.ndarray:
    h, w = img.shape[:2]
    dy, dx = int(h * fraction), int(w * fraction)
    return cv2.copyMakeBorder(img, dy, dy, dx, dx, cv2.BORDER_REPLICATE)


def collect_image_paths(dirs: list[Path]) -> list[Path]:
    paths = []
    for d in dirs:
        if not d.is_dir():
            print(f"  (skipping {d}: not a directory)")
            continue
        for pattern in IMAGE_EXTENSIONS:
            paths.extend(d.glob(pattern))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crops-dir",
        type=Path,
        nargs="+",
        default=[Path("data/crops/face/charles"), Path("data/web/face")],
    )
    parser.add_argument("--out", type=Path, default=Path("data/leclerc_embeddings.npy"))
    args = parser.parse_args()

    crop_paths = collect_image_paths(args.crops_dir)
    if not crop_paths:
        sys.exit(f"No images found in {args.crops_dir}")

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # Pass 1: single-face images are unambiguous (the one face present is the subject) and
    # build a trusted centroid. Multi-face images are deferred — 26% of the web crawl has 2+
    # people in frame (measured), and "highest detection confidence" is not a reliable proxy
    # for "is Leclerc" once other people are in the shot.
    trusted_embeddings = []
    trusted_names = []
    ambiguous = []  # (path, [face embeddings]) for images with 2+ faces
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
        if len(faces) == 1:
            trusted_embeddings.append(faces[0].normed_embedding)
            trusted_names.append(crop_path.name)
        else:
            ambiguous.append((crop_path.name, [f.normed_embedding for f in faces]))

    if not trusted_embeddings:
        sys.exit("No single-face images found — nothing to build a trusted centroid from.")

    trusted_centroid = np.stack(trusted_embeddings).astype(np.float32).mean(axis=0)
    trusted_centroid /= np.linalg.norm(trusted_centroid)

    # Pass 2: for each multi-face image, pick whichever face is closest to the trusted
    # centroid, instead of highest detection confidence.
    resolved_embeddings = []
    resolved_names = []
    for name, face_embeddings in ambiguous:
        sims_to_trusted = np.stack(face_embeddings) @ trusted_centroid
        best_idx = int(np.argmax(sims_to_trusted))
        resolved_embeddings.append(face_embeddings[best_idx])
        resolved_names.append(name)

    embeddings = trusted_embeddings + resolved_embeddings
    names = trusted_names + resolved_names

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
        f"crops={len(crop_paths)} single_face={len(trusted_embeddings)} "
        f"multi_face_resolved={len(resolved_embeddings)} no_detect={no_detect} "
        f"dropped={len(dropped)} kept={len(kept)} -> {args.out}"
    )


if __name__ == "__main__":
    main()
