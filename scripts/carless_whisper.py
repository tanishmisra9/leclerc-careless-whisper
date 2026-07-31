"""Play careless-whisper.m4a whenever Charles Leclerc's face is on camera.

Usage:
    uv run scripts/carless_whisper.py [--camera 0] [--threshold 0.4] [--max-misses 3]

Requires data/leclerc_embeddings.npy (run scripts/enroll_faces.py first).
Press q or Esc to quit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DET_WIDTH = 640


class AudioPlayer:
    """Wraps afplay so restart() always plays from 0:00, never resumes."""

    def __init__(self, path: Path):
        self.path = path
        self._proc: subprocess.Popen | None = None

    def restart(self) -> None:
        self.stop()
        self._proc = subprocess.Popen(["afplay", str(self.path)])

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--max-misses", type=int, default=3)
    args = parser.parse_args()

    ref_path = REPO_ROOT / "data" / "leclerc_embeddings.npy"
    audio_path = REPO_ROOT / "careless-whisper.m4a"

    if not ref_path.exists():
        sys.exit(f"Reference embeddings not found at {ref_path} — run scripts/enroll_faces.py first.")
    refs = np.load(ref_path)
    if refs.shape[0] == 0:
        sys.exit(f"Reference embeddings file is empty: {ref_path}")

    if not audio_path.exists():
        sys.exit(f"Audio file not found: {audio_path}")

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(DET_WIDTH, DET_WIDTH))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera index {args.camera} (try a different --camera index)")

    audio = AudioPlayer(audio_path)
    present = False
    misses = 0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed, stopping.")
                break

            h, w = frame.shape[:2]
            scale = DET_WIDTH / w
            small = cv2.resize(frame, (DET_WIDTH, int(h * scale)))
            sh, sw = small.shape[:2]

            faces = app.get(small)
            face_scores = [float((refs @ face.normed_embedding).max()) for face in faces]
            best = max(face_scores) if face_scores else 0.0
            matched = best >= args.threshold

            if matched:
                misses = 0
                if not present:
                    present = True
                    audio.restart()
            else:
                misses += 1
                if present and misses >= args.max_misses:
                    present = False
                    audio.stop()

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            for face, score in zip(faces, face_scores):
                x1, y1, x2, y2 = face.bbox
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(sw - 1, int(x2)), min(sh - 1, int(y2))
                color = (0, 255, 0) if score >= args.threshold else (0, 0, 255)
                cv2.rectangle(small, (x1, y1), (x2, y2), color, 2)
                cv2.putText(small, f"{score:.2f}", (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            status = "PLAYING" if present else "idle"
            cv2.putText(
                small,
                f"best={best:.2f} thresh={args.threshold:.2f} {status} fps={fps:.1f}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("CarlessWhisper", small)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        audio.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
