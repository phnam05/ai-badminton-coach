"""Stage 1: video -> per-frame pose keypoints.

Uses MediaPipe's Tasks API (PoseLandmarker) in VIDEO mode, which tracks the
person across frames instead of re-detecting from scratch each frame.

Output is a PoseSequence: per-frame normalized image landmarks (x, y in [0,1]
relative to frame size) plus MediaPipe's world landmarks (meters, hip-centered
pseudo-3D). Downstream stages derive angles/ratios from these; raw coordinates
never leave the feature layer (see docs/feature-spec.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .landmarks import NUM_LANDMARKS

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_full.task"


@dataclass
class FramePose:
    """Pose for one video frame. Landmarks are None when nobody was detected."""

    index: int
    timestamp_ms: int
    # 33 x [x, y, z, visibility, presence]; x/y normalized to frame size.
    landmarks: list[list[float]] | None
    # 33 x [x, y, z, visibility, presence]; meters, hip-centered (pseudo-3D).
    world_landmarks: list[list[float]] | None

    @property
    def detected(self) -> bool:
        return self.landmarks is not None


@dataclass
class PoseSequence:
    """All frames of one clip, plus the video metadata needed to interpret them."""

    source: str
    fps: float
    width: int
    height: int
    frames: list[FramePose] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        if not self.frames:
            return 0.0
        return sum(f.detected for f in self.frames) / len(self.frames)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": self.source,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frames": [
                {
                    "index": f.index,
                    "timestamp_ms": f.timestamp_ms,
                    "landmarks": f.landmarks,
                    "world_landmarks": f.world_landmarks,
                }
                for f in self.frames
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PoseSequence":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        seq = cls(source=data["source"], fps=data["fps"], width=data["width"], height=data["height"])
        seq.frames = [
            FramePose(
                index=f["index"],
                timestamp_ms=f["timestamp_ms"],
                landmarks=f["landmarks"],
                world_landmarks=f["world_landmarks"],
            )
            for f in data["frames"]
        ]
        return seq


def _landmarks_to_list(landmark_list) -> list[list[float]]:
    out = [
        [lm.x, lm.y, lm.z, lm.visibility, lm.presence]
        for lm in landmark_list
    ]
    assert len(out) == NUM_LANDMARKS, f"expected {NUM_LANDMARKS} landmarks, got {len(out)}"
    return out


class PoseExtractor:
    """Runs PoseLandmarker over a whole clip."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. Run tests/smoke_test.py once to download it."
            )
        self._options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
        )

    def extract(self, video_path: str | Path, progress: bool = True) -> PoseSequence:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        seq = PoseSequence(source=str(video_path), fps=fps, width=width, height=height)

        # VIDEO mode requires monotonically increasing timestamps.
        with vision.PoseLandmarker.create_from_options(self._options) as landmarker:
            index = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                timestamp_ms = int(round(index * 1000.0 / fps))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
                )
                if result.pose_landmarks:
                    frame = FramePose(
                        index=index,
                        timestamp_ms=timestamp_ms,
                        landmarks=_landmarks_to_list(result.pose_landmarks[0]),
                        world_landmarks=_landmarks_to_list(result.pose_world_landmarks[0]),
                    )
                else:
                    frame = FramePose(index=index, timestamp_ms=timestamp_ms, landmarks=None, world_landmarks=None)
                seq.frames.append(frame)
                index += 1
                if progress and total > 0 and index % 30 == 0:
                    print(f"  pose: frame {index}/{total}", flush=True)

        cap.release()
        return seq
