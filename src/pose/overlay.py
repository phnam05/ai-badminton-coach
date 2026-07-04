"""Render a skeleton-overlay video from a PoseSequence.

This is the Stage 1 checkpoint artifact: the user eyeballs it to confirm
tracking is solid before anything downstream is trusted.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from .extractor import PoseSequence
from .landmarks import POSE_CONNECTIONS

# Joints drawn dimmer when MediaPipe is unsure about them.
LOW_VISIBILITY = 0.5

COLOR_BONE = (80, 220, 80)        # BGR green
COLOR_JOINT = (60, 160, 255)      # BGR orange
COLOR_LOW_VIS = (100, 100, 100)   # gray for uncertain joints
COLOR_TEXT = (255, 255, 255)
COLOR_WARN = (60, 60, 230)        # red for "no detection"


def render_overlay(video_path: str | Path, seq: PoseSequence, out_path: str | Path) -> Path:
    """Draw the skeleton over each frame of the source video."""
    video_path, out_path = Path(video_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        seq.fps,
        (seq.width, seq.height),
    )
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Cannot open video writer for: {out_path}")

    thickness = max(2, seq.width // 640)
    radius = max(3, seq.width // 420)

    for frame_pose in seq.frames:
        ok, bgr = cap.read()
        if not ok:
            break

        if frame_pose.detected:
            pts = [
                (int(lm[0] * seq.width), int(lm[1] * seq.height), lm[3])
                for lm in frame_pose.landmarks
            ]
            for a, b in POSE_CONNECTIONS:
                color = COLOR_BONE if min(pts[a][2], pts[b][2]) >= LOW_VISIBILITY else COLOR_LOW_VIS
                cv2.line(bgr, pts[a][:2], pts[b][:2], color, thickness, cv2.LINE_AA)
            for x, y, vis in pts:
                color = COLOR_JOINT if vis >= LOW_VISIBILITY else COLOR_LOW_VIS
                cv2.circle(bgr, (x, y), radius, color, -1, cv2.LINE_AA)
            label, label_color = f"frame {frame_pose.index}", COLOR_TEXT
        else:
            label, label_color = f"frame {frame_pose.index}  NO DETECTION", COLOR_WARN

        cv2.putText(bgr, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2, cv2.LINE_AA)
        writer.write(bgr)

    cap.release()
    writer.release()
    return out_path
