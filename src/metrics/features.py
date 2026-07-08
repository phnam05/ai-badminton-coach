"""Stage 3: pose sequence -> per-frame, view-invariant feature time series.

This is the shared feature layer both evaluation engines consume
(docs/feature-spec.md). Everything is a joint angle (degrees) or a
torso-length ratio, so the numbers mean the same thing regardless of camera
distance, zoom, or player size — raw pixel coordinates never leave this
module. The sequence is canonicalized first (right-handed frame, detected
facing), so "front" and "racket arm" are unambiguous downstream.

No judgments here: this module MEASURES. Thresholds and verdicts live in
evaluate/, where they must trace to reference footage or a cited coaching
principle (CLAUDE.md ground-truth rule).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from ..pose import landmarks as lm
from ..pose.extractor import PoseSequence
from .canonical import canonicalize, detect_facing
from .geometry import incline_vs_vertical, joint_angle, vector_angle

# Feature name -> (unit, one-line meaning). Order here is the display order.
FEATURE_INFO: dict[str, tuple[str, str]] = {
    "right_wrist_height": ("torso lengths", "wrist above (+) / below (-) the racket shoulder"),
    "right_wrist_frontness": ("torso lengths", "wrist in front of (+) / behind (-) the racket shoulder"),
    "right_elbow_angle": ("degrees", "arm extension; 180 = straight"),
    "right_arm_elevation": ("degrees", "racket upper arm vs trunk; 0 = hanging, 180 = up"),
    "left_arm_elevation": ("degrees", "non-racket upper arm vs trunk"),
    "trunk_incline": ("degrees", "trunk vs vertical; + = leaning forward, - = arched back"),
    "right_knee_angle": ("degrees", "leg loading; 180 = straight"),
    "right_wrist_speed": ("torso lengths / s", "racket-wrist speed (event anchor)"),
    "wrist_cut_path": ("degrees", "wrist motion vs straight down; weak slice proxy (S2), read near contact only"),
}


@dataclass
class FeatureSeries:
    """All per-frame features of one clip, in the canonical right-handed frame."""

    source: str
    fps: float
    handedness: str                      # as filmed, before canonicalization
    facing: int                          # +1 = player faces increasing image-x (post-mirror)
    facing_agreement: float              # fraction of frames agreeing with `facing`
    detected: list[bool] = field(default_factory=list)
    features: dict[str, list[float]] = field(default_factory=dict)

    def array(self, name: str) -> np.ndarray:
        return np.asarray(self.features[name])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": self.source,
            "fps": self.fps,
            "handedness": self.handedness,
            "facing": self.facing,
            "facing_agreement": self.facing_agreement,
            "detected": self.detected,
            "features": self.features,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSeries":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def _interp_gaps(values: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Linearly fill frames where the pose was lost."""
    if detected.all():
        return values
    idx = np.arange(len(values))
    return np.interp(idx, idx[detected], values[detected])


def compute_features(seq: PoseSequence, handedness: str = "right") -> FeatureSeries:
    """Turn a pose sequence into the per-frame feature layer."""
    n = len(seq.frames)
    if n < 10:
        raise ValueError(f"clip too short for features ({n} frames)")

    seq = canonicalize(seq, handedness)
    facing, agreement = detect_facing(seq)

    detected = np.array([f.detected for f in seq.frames])

    def track(index: int, coord: int, scale: float) -> np.ndarray:
        vals = np.array(
            [f.landmarks[index][coord] * scale if f.detected else np.nan for f in seq.frames]
        )
        vals = _interp_gaps(np.nan_to_num(vals), detected)
        # Same jitter treatment as segment/events.py: smooth positions before
        # deriving anything from them.
        return savgol_filter(vals, 7, 2) if n >= 7 else vals

    def xy(index: int) -> tuple[np.ndarray, np.ndarray]:
        return track(index, 0, seq.width), track(index, 1, seq.height)

    r_sh = xy(lm.RIGHT_SHOULDER)
    l_sh = xy(lm.LEFT_SHOULDER)
    r_el = xy(lm.RIGHT_ELBOW)
    l_el = xy(lm.LEFT_ELBOW)
    r_wr = xy(lm.RIGHT_WRIST)
    r_hip = xy(lm.RIGHT_HIP)
    l_hip = xy(lm.LEFT_HIP)
    r_knee = xy(lm.RIGHT_KNEE)
    r_ankle = xy(lm.RIGHT_ANKLE)

    mid_sh = ((r_sh[0] + l_sh[0]) / 2, (r_sh[1] + l_sh[1]) / 2)
    mid_hip = ((r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2)

    # Scale unit: median torso length in pixels (mid-shoulder to mid-hip).
    torso = np.hypot(mid_sh[0] - mid_hip[0], mid_sh[1] - mid_hip[1])
    torso_len = float(np.median(torso[detected]))
    if torso_len <= 0:
        raise ValueError("degenerate torso length — pose data unusable")

    # Trunk axis pointing DOWN the body (shoulders -> hips): elevation 0 means
    # the upper arm hangs alongside the trunk, 180 means straight overhead.
    trunk_down = (mid_hip[0] - mid_sh[0], mid_hip[1] - mid_sh[1])

    features: dict[str, np.ndarray] = {}
    # Image y grows downward, so "above the shoulder" is shoulder_y - wrist_y.
    features["right_wrist_height"] = (r_sh[1] - r_wr[1]) / torso_len
    features["right_wrist_frontness"] = (r_wr[0] - r_sh[0]) * facing / torso_len
    features["right_elbow_angle"] = joint_angle(*r_sh, *r_el, *r_wr)
    features["right_arm_elevation"] = vector_angle(
        r_el[0] - r_sh[0], r_el[1] - r_sh[1], *trunk_down
    )
    features["left_arm_elevation"] = vector_angle(
        l_el[0] - l_sh[0], l_el[1] - l_sh[1], *trunk_down
    )
    features["trunk_incline"] = incline_vs_vertical(
        forward=(mid_sh[0] - mid_hip[0]) * facing,
        up=mid_hip[1] - mid_sh[1],
    )
    features["right_knee_angle"] = joint_angle(*r_hip, *r_knee, *r_ankle)

    speed = np.zeros(n)
    speed[1:] = np.hypot(np.diff(r_wr[0]), np.diff(r_wr[1])) * seq.fps / torso_len
    features["right_wrist_speed"] = speed

    # Direction of wrist motion vs straight down: 0 = driving straight down,
    # +/- = veering toward/away from the facing direction. Undefined when the
    # wrist is near-still, hence "read near contact only" (FEATURE_INFO).
    vx = np.zeros(n)
    vy = np.zeros(n)
    vx[1:] = np.diff(r_wr[0]) * facing
    vy[1:] = np.diff(r_wr[1])
    features["wrist_cut_path"] = np.degrees(np.arctan2(vx, vy))

    return FeatureSeries(
        source=seq.source,
        fps=seq.fps,
        handedness=handedness,
        facing=facing,
        facing_agreement=agreement,
        detected=detected.tolist(),
        features={k: np.asarray(v).tolist() for k, v in features.items()},
    )
