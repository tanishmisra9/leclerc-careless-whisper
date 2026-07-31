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

# Helmet-sourced images (data/web/helmet/) need their own, lower threshold: partial visor
# occlusion legitimately lowers similarity-to-centroid regardless of identity, so the 0.3 cut
# tuned for unoccluded faces was rejecting genuine helmet-on Leclerc shots. Measured by visually
# checking 7 images spanning the low end of the distribution: everything at or below 0.078 was
# either a spurious detection with no real face (dark/mirrored visor, a LEGO helmet sculpture) or
# genuine contamination (a background bystander); everything at or above 0.106 was confirmed him
# (car-number "16" visible, or a clearly visible eye through an open visor). 0.1 splits that gap.
HELMET_OUTLIER_THRESH = 0.1

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg")


def pad(img: np.ndarray, fraction: float = PAD_FRACTION) -> np.ndarray:
    h, w = img.shape[:2]
    dy, dx = int(h * fraction), int(w * fraction)
    return cv2.copyMakeBorder(img, dy, dy, dx, dx, cv2.BORDER_REPLICATE)


def is_helmet_path(path: Path) -> bool:
    return path.parent.name == "helmet"


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
        default=[Path("data/crops/face/charles"), Path("data/web/face"), Path("data/web/helmet")],
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
    trusted_is_helmet = []
    ambiguous = []  # (path, is_helmet, [face embeddings]) for images with 2+ faces
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
            trusted_is_helmet.append(is_helmet_path(crop_path))
        else:
            ambiguous.append((crop_path.name, is_helmet_path(crop_path), [f.normed_embedding for f in faces]))

    if not trusted_embeddings:
        sys.exit("No single-face images found — nothing to build a trusted centroid from.")

    trusted_centroid = np.stack(trusted_embeddings).astype(np.float32).mean(axis=0)
    trusted_centroid /= np.linalg.norm(trusted_centroid)

    # Pass 2: for each multi-face image, pick whichever face is closest to the trusted
    # centroid, instead of highest detection confidence.
    resolved_embeddings = []
    resolved_names = []
    resolved_is_helmet = []
    for name, helmet, face_embeddings in ambiguous:
        sims_to_trusted = np.stack(face_embeddings) @ trusted_centroid
        best_idx = int(np.argmax(sims_to_trusted))
        resolved_embeddings.append(face_embeddings[best_idx])
        resolved_names.append(name)
        resolved_is_helmet.append(helmet)

    embeddings = trusted_embeddings + resolved_embeddings
    names = trusted_names + resolved_names
    is_helmet = np.array(trusted_is_helmet + resolved_is_helmet)

    all_embeddings = np.stack(embeddings).astype(np.float32)
    centroid = all_embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    similarities = all_embeddings @ centroid

    thresholds = np.where(is_helmet, HELMET_OUTLIER_THRESH, OUTLIER_THRESH)
    keep_mask = similarities >= thresholds
    dropped = [(names[i], float(similarities[i]), float(thresholds[i])) for i in range(len(names)) if not keep_mask[i]]
    if dropped:
        print(f"Dropped {len(dropped)} outlier(s):")
        for name, sim, thresh in dropped:
            print(f"  {name}: {sim:.3f} (threshold {thresh:.2f})")

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
