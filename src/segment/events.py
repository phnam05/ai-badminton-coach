"""Stage 2: locate swing events (backswing apex, CONTACT WINDOW, follow-through).

Contact is anchored to a WINDOW around the racket-wrist speed peak, never a
single frame — the 2D speed peak does not reliably coincide with true contact,
and one wrong frame would corrupt every scored metric (docs/feature-spec.md).
Detection carries a confidence verdict with reasons; on LOW confidence the
caller must surface "re-film / re-trim" instead of silently scoring.

Speeds are expressed in torso-lengths per second: scale-invariant, so the same
thresholds apply regardless of camera distance or player size.

Heuristic constants below are event-DETECTION plumbing (where in time the swing
is), not technique judgments — the ground-truth rule in CLAUDE.md governs
technique thresholds, which do not live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from ..pose import landmarks as lm
from ..pose.extractor import PoseSequence

# --- detection tuning (plumbing, not technique) ---
WINDOW_SPEED_FRACTION = 0.6   # window = contiguous frames >= this fraction of peak speed
WINDOW_MAX_HALF = 4           # hard cap on window half-width, frames
APEX_SPEED_FRACTION = 0.3     # backswing apex = last lull below this before the peak
FOLLOW_END_FRACTION = 0.2     # follow-through ends when speed stays below this
PEAK_MEDIAN_RATIO_MIN = 3.0   # peak must stand this far above the clip's median speed
# Absolute swing-existence floor. Measured: real smashes peak at 29-41 tl/s
# (smash_coachhan 29.3, smash_2 40.7); non-swing camera motion peaks at ~0.5.
# 5.0 splits those populations with a wide margin on both sides.
MIN_PEAK_SPEED = 5.0          # torso-lengths/sec; below this, no swing happened
EDGE_MARGIN = 3               # peak this close to clip edge => swing likely cut off
MIN_WRIST_VISIBILITY = 0.5
MAX_MISSING_NEAR_CONTACT = 0.2


@dataclass
class SwingEvents:
    """Detected swing timeline for one clip, with detection confidence."""

    contact_peak: int                      # frame of max wrist speed (window anchor)
    contact_window: tuple[int, int]        # inclusive frame range around the peak
    backswing_apex: int | None
    follow_through_end: int | None
    confidence: str                        # "high" | "low"
    reasons: list[str] = field(default_factory=list)   # why confidence is low
    wrist_speed: np.ndarray = field(default=None, repr=False)  # torso-lengths/sec, per frame
    fps: float = 30.0

    def frame_to_seconds(self, frame: int) -> float:
        return frame / self.fps

    def summary(self) -> str:
        lines = []
        w0, w1 = self.contact_window
        if self.backswing_apex is not None:
            lines.append(f"backswing apex   frame {self.backswing_apex}  ({self.frame_to_seconds(self.backswing_apex):.2f}s)")
        lines.append(
            f"CONTACT window   frames {w0}-{w1}  ({self.frame_to_seconds(w0):.2f}-{self.frame_to_seconds(w1):.2f}s), "
            f"speed peak at frame {self.contact_peak}"
        )
        if self.follow_through_end is not None:
            lines.append(f"follow-through   ends frame {self.follow_through_end}  ({self.frame_to_seconds(self.follow_through_end):.2f}s)")
        lines.append(f"confidence       {self.confidence.upper()}")
        for r in self.reasons:
            lines.append(f"  - {r}")
        return "\n".join(lines)


def _interp_gaps(values: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Linearly fill frames where the pose was lost."""
    if detected.all():
        return values
    idx = np.arange(len(values))
    return np.interp(idx, idx[detected], values[detected])


def detect_events(seq: PoseSequence, side: str = "right") -> SwingEvents:
    """Locate the swing events in a pose sequence. `side` = racket hand."""
    n = len(seq.frames)
    if n < 10:
        raise ValueError(f"clip too short to segment ({n} frames)")

    wrist_i = lm.RIGHT_WRIST if side == "right" else lm.LEFT_WRIST
    shoulder_i = lm.RIGHT_SHOULDER if side == "right" else lm.LEFT_SHOULDER

    detected = np.array([f.detected for f in seq.frames])
    if not detected.any():
        raise ValueError("no pose detected in any frame")

    def track(index: int, coord: int, scale: float) -> np.ndarray:
        vals = np.array([f.landmarks[index][coord] * scale if f.detected else np.nan for f in seq.frames])
        return _interp_gaps(np.nan_to_num(vals), detected)

    # Pixel coordinates so x/y distances are isotropic.
    wx, wy = track(wrist_i, 0, seq.width), track(wrist_i, 1, seq.height)
    sy = track(shoulder_i, 1, seq.height)
    wrist_vis = np.array([f.landmarks[wrist_i][3] if f.detected else 0.0 for f in seq.frames])

    # Scale unit: median torso length in pixels (mid-shoulder to mid-hip).
    def mid(a: int, b: int, coord: int, scale: float) -> np.ndarray:
        return (track(a, coord, scale) + track(b, coord, scale)) / 2

    torso = np.hypot(
        mid(lm.LEFT_SHOULDER, lm.RIGHT_SHOULDER, 0, seq.width) - mid(lm.LEFT_HIP, lm.RIGHT_HIP, 0, seq.width),
        mid(lm.LEFT_SHOULDER, lm.RIGHT_SHOULDER, 1, seq.height) - mid(lm.LEFT_HIP, lm.RIGHT_HIP, 1, seq.height),
    )
    torso_len = float(np.median(torso[detected]))
    if torso_len <= 0:
        raise ValueError("degenerate torso length — pose data unusable")

    # Smooth positions, then differentiate -> speed in torso-lengths/sec.
    if n >= 7:
        wx, wy = savgol_filter(wx, 7, 2), savgol_filter(wy, 7, 2)
    speed = np.zeros(n)
    speed[1:] = np.hypot(np.diff(wx), np.diff(wy)) * seq.fps / torso_len

    reasons: list[str] = []

    # --- contact peak ---
    peaks, props = find_peaks(speed, prominence=np.max(speed) * 0.3)
    overhead = [p for p in peaks if wy[p] < sy[p]]  # image y grows downward
    if overhead:
        peak = int(max(overhead, key=lambda p: speed[p]))
    elif len(peaks):
        peak = int(max(peaks, key=lambda p: speed[p]))
        reasons.append("wrist is not above the shoulder at the speed peak — overhead swing not confirmed")
    else:
        peak = int(np.argmax(speed))
        reasons.append("no distinct speed peak found — clip may not contain a swing")

    peak_speed = speed[peak]
    if peak_speed < MIN_PEAK_SPEED:
        reasons.append(
            f"peak wrist speed ({peak_speed:.1f} torso-lengths/s) is far below any real swing — "
            "this clip does not appear to contain a shot"
        )
    median_speed = float(np.median(speed))
    if median_speed > 0 and peak_speed / median_speed < PEAK_MEDIAN_RATIO_MIN:
        reasons.append("speed peak barely stands out from baseline motion — contact timing unreliable")
    if peak < EDGE_MARGIN or peak > n - 1 - EDGE_MARGIN:
        reasons.append("speed peak is at the very edge of the clip — the swing looks cut off; re-trim with margin")

    # --- contact window: contiguous frames near peak speed, hard-capped ---
    lo = peak
    while lo > 0 and peak - lo < WINDOW_MAX_HALF and speed[lo - 1] >= peak_speed * WINDOW_SPEED_FRACTION:
        lo -= 1
    hi = peak
    while hi < n - 1 and hi - peak < WINDOW_MAX_HALF and speed[hi + 1] >= peak_speed * WINDOW_SPEED_FRACTION:
        hi += 1

    # --- reliability near contact ---
    ctx_lo, ctx_hi = max(0, lo - 3), min(n, hi + 4)
    missing = 1.0 - detected[ctx_lo:ctx_hi].mean()
    if missing > MAX_MISSING_NEAR_CONTACT:
        reasons.append(f"pose lost in {missing:.0%} of frames around contact — re-film with better lighting/framing")
    if wrist_vis[lo : hi + 1].mean() < MIN_WRIST_VISIBILITY:
        reasons.append("racket wrist poorly visible around contact — metrics at contact would be guesses")

    # --- backswing apex: last low-speed lull before the acceleration ---
    apex = None
    for i in range(lo - 1, 0, -1):
        if speed[i] <= peak_speed * APEX_SPEED_FRACTION:
            apex = i
            break
    if apex is None:
        reasons.append("no backswing lull found before contact — clip may start mid-swing")

    # --- follow-through end: speed stays low after the window ---
    follow_end = None
    below = 0
    for i in range(hi + 1, n):
        below = below + 1 if speed[i] <= peak_speed * FOLLOW_END_FRACTION else 0
        if below >= 3:
            follow_end = i - 2
            break

    return SwingEvents(
        contact_peak=peak,
        contact_window=(lo, hi),
        backswing_apex=apex,
        follow_through_end=follow_end,
        confidence="low" if reasons else "high",
        reasons=reasons,
        wrist_speed=speed,
        fps=seq.fps,
    )
