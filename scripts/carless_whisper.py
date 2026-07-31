"""Play careless-whisper.m4a whenever Charles Leclerc's face is on camera.

Usage:
    uv run scripts/carless_whisper.py [--camera 0] [--threshold 0.4] [--max-misses 3]
                                       [--capture-width 1920] [--capture-height 1080]
                                       [--rotate 0|90|180|270] [--no-auto-rotate]

Requires data/leclerc_embeddings.npy (run scripts/enroll_faces.py first).

cv2.VideoCapture reads the raw sensor buffer and ignores device-orientation metadata, so a phone
held in portrait (e.g. via Continuity Camera) still delivers landscape frames -- OpenCV has no way
to read the phone's physical orientation directly (that's a native-AVFoundation-only API). Instead,
--rotate sets the starting guess, and by default the script auto-corrects using the face detector
itself as the "which way is up" signal: after a stretch of frames with no face detected at all, it
tries the other three rotations of the raw frame and switches if one of them finds a face. This
reacts when detection lapses, not the instant the phone physically turns -- pass --no-auto-rotate
to disable it and rely on a fixed --rotate value instead.

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
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
DET_WIDTH = 640  # detection always runs on a downscaled copy at this width, independent of display size

# Consecutive no-face-at-all frames before probing alternate rotations. Long enough that a normal
# brief look-away or occlusion doesn't trigger 3 extra detection passes; short enough to feel
# responsive for a demo (roughly 1-1.5s at the fps this pipeline typically runs at).
REORIENT_AFTER_MISSES = 15

# Palette (RGB). Deliberately not default red/green: amber reads as "active/confident" for a
# match rather than alarm-adjacent, coral is a softer/informational below-threshold signal.
COLOR_IDLE = (0x3A, 0x3F, 0x4B)
COLOR_MATCH = (0xE8, 0xA3, 0x3D)
COLOR_MISS = (0xD6, 0x5A, 0x5A)
COLOR_HUD_BG = (0x14, 0x16, 0x1A)
COLOR_TEXT_PRIMARY = (0xF5, 0xF5, 0xF0)
COLOR_TEXT_SECONDARY = (0x9A, 0xA0, 0xAC)

FONT_CANDIDATES = ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/HelveticaNeue.ttc"]
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    return (b, g, r)


def get_font(size: int) -> ImageFont.ImageFont:
    if size in _font_cache:
        return _font_cache[size]
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            font = ImageFont.truetype(path, size)
            _font_cache[size] = font
            return font
    font = ImageFont.load_default(size=size)
    _font_cache[size] = font
    return font


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


class Hud:
    """Draws the face-tracking overlay + bottom status bar onto a display-resolution frame.

    cv2-native drawing (boxes, semi-transparent bars, the border glow) happens directly on the
    BGR numpy frame. Text goes through exactly one BGR<->PIL round-trip per frame (not one per
    string) so anti-aliased typography doesn't multiply the conversion cost.
    """

    def __init__(self, display_w: int, display_h: int):
        self.w = display_w
        self.h = display_h
        self.box_thickness = max(2, display_w // 400)
        self.border_thickness = max(6, display_w // 220)
        self.hud_h = int(display_h * 0.13)
        self.font_primary = get_font(max(16, display_w // 55))
        self.font_secondary = get_font(max(12, display_w // 90))
        self.font_score = get_font(max(12, display_w // 100))
        self.pad = max(12, display_w // 100)

    def render(
        self,
        frame: np.ndarray,
        faces: list[tuple[int, int, int, int, float]],  # x1,y1,x2,y2,score
        best: float,
        threshold: float,
        fps: float,
        present: bool,
    ) -> np.ndarray:
        text_items: list[tuple[int, int, str, ImageFont.ImageFont, tuple[int, int, int]]] = []

        # --- face boxes (cv2-native) + queue their score-chip text ---
        for x1, y1, x2, y2, score in faces:
            color = COLOR_MATCH if score >= threshold else COLOR_MISS
            cv2.rectangle(frame, (x1, y1), (x2, y2), rgb_to_bgr(color), self.box_thickness, cv2.LINE_AA)
            chip_text = f"{score:.2f}"
            chip_w, chip_h = self._text_size(chip_text, self.font_score)
            chip_x, chip_y = x1, max(0, y1 - chip_h - 10)
            cv2.rectangle(
                frame,
                (chip_x, chip_y),
                (chip_x + chip_w + 12, chip_y + chip_h + 8),
                rgb_to_bgr(COLOR_HUD_BG),
                -1,
                cv2.LINE_AA,
            )
            text_items.append((chip_x + 6, chip_y + 4, chip_text, self.font_score, color))

        # --- bottom HUD bar (cv2-native semi-transparent backdrop) ---
        bar_top = self.h - self.hud_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_top), (self.w, self.h), rgb_to_bgr(COLOR_HUD_BG), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)

        # The dot alone carries the state color; the label stays a fixed legible off-white so it
        # never has to fight for contrast against the dark backdrop (slate-on-near-black, in the
        # idle case, is nearly invisible -- caught by rendering a synthetic frame and measuring
        # pixel deltas before ever touching a camera).
        state_color = COLOR_MATCH if present else COLOR_IDLE
        state_text = "PLAYING" if present else "LISTENING"
        dot = "●" if present else "○"
        secondary_text = f"match {best:.2f}  ·  threshold {threshold:.2f}  ·  {fps:.0f}fps"

        text_y = bar_top + (self.hud_h - self._text_size(state_text, self.font_primary)[1]) // 2
        dot_w, _ = self._text_size(dot, self.font_primary)
        text_items.append((self.pad, text_y, dot, self.font_primary, state_color))
        text_items.append((self.pad + dot_w + self.pad // 2, text_y, state_text, self.font_primary, COLOR_TEXT_PRIMARY))

        sec_w, sec_h = self._text_size(secondary_text, self.font_secondary)
        sec_y = bar_top + (self.hud_h - sec_h) // 2
        text_items.append((self.w - self.pad - sec_w, sec_y, secondary_text, self.font_secondary, COLOR_TEXT_SECONDARY))

        # --- full-frame accent border when actively playing ---
        if present:
            t = self.border_thickness
            cv2.rectangle(frame, (t // 2, t // 2), (self.w - t // 2, self.h - t // 2), rgb_to_bgr(COLOR_MATCH), t, cv2.LINE_AA)

        return self._draw_text_batch(frame, text_items)

    def _text_size(self, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _draw_text_batch(self, frame: np.ndarray, items: list[tuple[int, int, str, ImageFont.ImageFont, tuple[int, int, int]]]) -> np.ndarray:
        if not items:
            return frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        for x, y, text, font, color in items:
            draw.text((x, y), text, font=font, fill=color)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


_ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    code = _ROTATIONS[degrees]
    return frame if code is None else cv2.rotate(frame, code)


def detect_faces(app, frame: np.ndarray) -> tuple[list, float]:
    """Runs detection on a downscaled copy of frame. Returns (faces, det_scale) where det_scale
    maps a detection-space coordinate back to frame-space (multiply by it)."""
    h, w = frame.shape[:2]
    det_scale = w / DET_WIDTH
    det_h = int(h / det_scale)
    small = cv2.resize(frame, (DET_WIDTH, det_h))
    return app.get(small), det_scale


def open_camera(index: int, want_w: int, want_h: int) -> tuple[cv2.VideoCapture, np.ndarray]:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera index {index} (try a different --camera index)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
    ret, frame = cap.read()
    if not ret:
        sys.exit(f"Camera index {index} opened but returned no frame.")
    h, w = frame.shape[:2]
    print(f"Camera: requested {want_w}x{want_h}, got {w}x{h}")
    return cap, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--max-misses", type=int, default=3)
    parser.add_argument("--capture-width", type=int, default=1920)
    parser.add_argument("--capture-height", type=int, default=1080)
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                         help="starting rotation guess; auto-adjusts from here unless --no-auto-rotate")
    parser.add_argument("--no-auto-rotate", action="store_false", dest="auto_rotate", default=True,
                         help="disable automatic re-orientation when no face is found for a while")
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

    cap, first_raw = open_camera(args.camera, args.capture_width, args.capture_height)
    current_rotation = args.rotate

    def build_display(raw: np.ndarray) -> tuple[np.ndarray, int, int, Hud]:
        rotated = rotate_frame(raw, current_rotation)
        h, w = rotated.shape[:2]
        return rotated, w, h, Hud(w, h)

    frame, display_w, display_h, hud = build_display(first_raw)
    if current_rotation:
        print(f"Rotation: {current_rotation}° -> display {display_w}x{display_h}")

    audio = AudioPlayer(audio_path)
    present = False
    misses = 0
    no_face_streak = 0
    prev_time = time.time()
    raw_frame: np.ndarray | None = first_raw

    try:
        while True:
            if raw_frame is None:
                ret, raw_frame = cap.read()
                if not ret:
                    print("Camera read failed, stopping.")
                    break
                frame = rotate_frame(raw_frame, current_rotation)

            faces_raw, det_scale = detect_faces(app, frame)

            if not faces_raw and args.auto_rotate:
                no_face_streak += 1
                if no_face_streak >= REORIENT_AFTER_MISSES:
                    for delta in (90, 180, 270):
                        trial_rotation = (current_rotation + delta) % 360
                        trial_frame = rotate_frame(raw_frame, trial_rotation)
                        trial_faces, trial_scale = detect_faces(app, trial_frame)
                        if trial_faces:
                            current_rotation = trial_rotation
                            frame, display_w, display_h, hud = build_display(raw_frame)
                            faces_raw, det_scale = trial_faces, trial_scale
                            print(f"Rotation: switched to {current_rotation}° -> display {display_w}x{display_h} (face found)")
                            break
                    no_face_streak = 0
            else:
                no_face_streak = 0

            scored_faces: list[tuple[int, int, int, int, float]] = []
            for face in faces_raw:
                score = float((refs @ face.normed_embedding).max())
                x1, y1, x2, y2 = face.bbox * det_scale
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(display_w - 1, int(x2)), min(display_h - 1, int(y2))
                scored_faces.append((x1, y1, x2, y2, score))

            face_scores = [s[4] for s in scored_faces]
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

            display = hud.render(frame, scored_faces, best, args.threshold, fps, present)

            cv2.imshow("CarlessWhisper", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            raw_frame = None
    finally:
        audio.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
