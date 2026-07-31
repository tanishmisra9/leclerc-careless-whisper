"""Extract frames from a video and crop detected faces, filtering blurry/tiny/duplicate crops.

Usage:
    uv run scripts/video_parser.py data/raw_video/leclerc_interview.mp4 --fps 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from common.image_filters import DedupTracker, is_too_blurry, is_too_small


def extract_frames(video_path: Path, out_dir: Path, fps: float) -> list[Path]:
    out_pattern = str(out_dir / "frame_%06d.jpg")
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vf", f"fps={fps}", "-q:v", "2", out_pattern],
        check=True,
        capture_output=True,
    )
    return sorted(out_dir.glob("frame_*.jpg"))


def crop_with_margin(img, bbox, margin: float = 0.2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * margin))
    y1 = max(0, int(y1 - bh * margin))
    x2 = min(w, int(x2 + bw * margin))
    y2 = min(h, int(y2 + bh * margin))
    return img[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("data/crops/face"))
    parser.add_argument("--min-size", type=int, default=60)
    parser.add_argument("--blur-thresh", type=float, default=15.0)
    parser.add_argument("--det-thresh", type=float, default=0.6)
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"Video not found: {args.video}")

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1)

    out_dir = args.out / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup = DedupTracker()

    frames_processed = 0
    faces_found = 0
    faces_kept = 0
    rejected_small = 0
    rejected_blurry = 0
    rejected_duplicate = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frame_paths = extract_frames(args.video, tmp_dir, args.fps)

        for frame_path in frame_paths:
            frames_processed += 1
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            faces = app.get(img)
            for i, face in enumerate(faces):
                if face.det_score < args.det_thresh:
                    continue
                faces_found += 1
                crop = crop_with_margin(img, face.bbox)
                if crop.size == 0:
                    continue
                if is_too_small(crop, args.min_size):
                    rejected_small += 1
                    continue
                if is_too_blurry(crop, args.blur_thresh):
                    rejected_blurry += 1
                    continue
                if dedup.is_duplicate(crop):
                    rejected_duplicate += 1
                    continue
                out_path = out_dir / f"{frame_path.stem}_face_{i}.png"
                cv2.imwrite(str(out_path), crop)
                faces_kept += 1

    print(
        f"{args.video.name}: {frames_processed} frames processed, "
        f"{faces_found} faces detected, {faces_kept} crops kept -> {out_dir}"
    )
    print(
        f"  rejected: {rejected_small} too small, {rejected_blurry} too blurry, "
        f"{rejected_duplicate} near-duplicate"
    )


if __name__ == "__main__":
    main()
