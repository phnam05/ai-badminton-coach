"""Interactive frame-by-frame review of detected swing events (Stage 2 checkpoint).

The events overlay video is hard to judge by eye: the swing lasts a fraction of
a second and ordinary video players cannot step backward frame by frame. This
viewer gives that control — step one frame at a time, jump straight to each
detected event, and record which frame YOU judge to be the real hit. That
judgment is saved to <clip>.groundtruth.json so detector accuracy can be scored
against it later, instead of re-eyeballing every clip after every tuning change.

Keys:
  <- / ->  (or , / .)   step one frame back / forward
  Home / End            first / last frame
  a  s  c               jump to backswing apex / speed peak / suspected contact
  [  ]                  jump to contact-window start / end
  f                     jump to follow-through end
  g                     mark current frame as the TRUE contact (saves ground truth)
  q / Esc               quit
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..pose.extractor import PoseSequence
from ..pose.overlay import COLOR_CONTACT, COLOR_TEXT, COLOR_WINDOW, annotate_frame
from .events import SwingEvents

MAX_DISPLAY_WIDTH = 1280   # shrink display so tall phone clips fit on screen
JPEG_QUALITY = 90          # frames are held in memory JPEG-encoded (raw would be GBs)
STRIP_HEIGHT = 56          # timeline strip below the video, px
STRIP_MARGIN = 12

COLOR_STRIP_BG = (28, 28, 28)
COLOR_STRIP_AXIS = (90, 90, 90)
COLOR_CURSOR = (255, 255, 255)
COLOR_TRUTH = (80, 220, 80)   # green — the user's own judged contact

# cv2.waitKeyEx codes (Windows and Linux variants), plus plain-ASCII fallbacks.
KEYS_PREV = {2424832, 65361, ord(",")}
KEYS_NEXT = {2555904, 65363, ord(".")}
KEYS_HOME = {2359296, 65360}
KEYS_END = {2621440, 65367}
KEYS_QUIT = {27, ord("q")}


def _load_annotated_frames(video_path: Path, seq: PoseSequence, events: SwingEvents) -> list[np.ndarray]:
    """Read the clip once, annotate every frame, return them JPEG-encoded."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    scale = min(1.0, MAX_DISPLAY_WIDTH / seq.width)
    encoded: list[np.ndarray] = []
    for frame_pose in seq.frames:
        ok, bgr = cap.read()
        if not ok:
            break
        annotate_frame(bgr, frame_pose, seq, events)  # full-res, same as events.mp4
        if scale < 1.0:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise IOError("JPEG encoding failed while caching frames")
        encoded.append(buf)
    cap.release()
    if not encoded:
        raise IOError(f"No frames read from video: {video_path}")
    return encoded


def _render_strip(width: int, n_frames: int, current: int, events: SwingEvents,
                  truth_frame: int | None, message: str) -> np.ndarray:
    """Timeline strip: every detected event as a colored tick, cursor on top."""
    strip = np.full((STRIP_HEIGHT, width, 3), COLOR_STRIP_BG, dtype=np.uint8)
    x0, x1 = STRIP_MARGIN, width - STRIP_MARGIN
    y_axis = 18

    def to_x(frame: int) -> int:
        return x0 + round((x1 - x0) * frame / max(1, n_frames - 1))

    cv2.line(strip, (x0, y_axis), (x1, y_axis), COLOR_STRIP_AXIS, 1, cv2.LINE_AA)

    w0, w1 = events.contact_window
    cv2.rectangle(strip, (to_x(w0), y_axis - 5), (to_x(w1), y_axis + 5), COLOR_WINDOW, -1)
    for frame, color in [
        (events.backswing_apex, COLOR_STRIP_AXIS),
        (events.speed_peak, COLOR_TEXT),
        (events.follow_through_end, COLOR_STRIP_AXIS),
        (events.contact_frame, COLOR_CONTACT),
    ]:
        if frame is not None:
            cv2.line(strip, (to_x(frame), y_axis - 7), (to_x(frame), y_axis + 7), color, 2)
    if truth_frame is not None:
        x = to_x(truth_frame)
        cv2.drawMarker(strip, (x, y_axis + 12), COLOR_TRUTH, cv2.MARKER_TRIANGLE_UP, 10, 2)

    x = to_x(current)
    cv2.line(strip, (x, 4), (x, y_axis + 8), COLOR_CURSOR, 1, cv2.LINE_AA)

    hint = message or "arrows/,. step | a apex  s peak  c contact  [ ] window  f follow | g mark true hit | q quit"
    cv2.putText(strip, hint, (x0, STRIP_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                COLOR_TRUTH if message else COLOR_STRIP_AXIS, 1, cv2.LINE_AA)
    return strip


def _save_ground_truth(path: Path, video_path: Path, frame: int, events: SwingEvents) -> None:
    """Record the user's judged hit frame next to what the detector said."""
    w0, w1 = events.contact_window
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "clip": str(video_path),
        "true_contact_frame": frame,
        "detected": {
            "contact_frame": events.contact_frame,
            "contact_window": [w0, w1],
            "speed_peak": events.speed_peak,
            "confidence": events.confidence,
        },
        "error_frames": events.contact_frame - frame,
        "true_frame_in_window": w0 <= frame <= w1,
        "marked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))


def review_clip(video_path: str | Path, seq: PoseSequence, events: SwingEvents,
                groundtruth_path: str | Path) -> int | None:
    """Open the interactive viewer. Returns the marked true-contact frame, if any."""
    video_path, groundtruth_path = Path(video_path), Path(groundtruth_path)
    frames = _load_annotated_frames(video_path, seq, events)
    n = len(frames)

    jumps = {
        ord("a"): events.backswing_apex,
        ord("s"): events.speed_peak,
        ord("c"): events.contact_frame,
        ord("["): events.contact_window[0],
        ord("]"): events.contact_window[1],
        ord("f"): events.follow_through_end,
    }

    window = f"review - {video_path.stem}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    current = min(events.contact_frame, n - 1)  # start where it matters
    truth_frame: int | None = None
    message = ""
    while True:
        bgr = cv2.imdecode(frames[current], cv2.IMREAD_COLOR)
        strip = _render_strip(bgr.shape[1], n, current, events, truth_frame, message)
        cv2.imshow(window, np.vstack([bgr, strip]))

        key = cv2.waitKeyEx(0)
        if key in KEYS_QUIT or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        message = ""
        if key in KEYS_PREV:
            current = max(0, current - 1)
        elif key in KEYS_NEXT:
            current = min(n - 1, current + 1)
        elif key in KEYS_HOME:
            current = 0
        elif key in KEYS_END:
            current = n - 1
        elif key == ord("g"):
            truth_frame = current
            _save_ground_truth(groundtruth_path, video_path, current, events)
            message = f"ground truth saved: frame {current} -> {groundtruth_path.name}"
        elif key in jumps and jumps[key] is not None:
            current = min(jumps[key], n - 1)

    cv2.destroyWindow(window)
    return truth_frame
