"""Stage 2: locate swing events (backswing apex, CONTACT WINDOW, follow-through).

Two separate questions, answered by two separate signals:

  * "Did an overhead swing happen, and roughly WHEN?"  -> racket-wrist SPEED.
    A fast, above-the-shoulder wrist movement is a swing. The speed peak locates
    the swing in time; it is NOT contact (see below).
  * "Which frame is the HIT?"  -> ARM EXTENSION (wrist-to-shoulder distance over
    torso length). Contact happens when the arm is stretched out to meet the
    shuttle, so the suspected contact frame is the moment of maximum extension
    near the swing. Wrist speed peaks BEFORE contact (whip on a smash) or is a
    lull AT contact (a slice is deliberately decelerated), so speed alone
    mistimes the hit — badly for slices. Extension is a geometric property of
    contact itself and lands on it for both shots.

Contact is anchored to a WINDOW, never a single frame — the 2D signals do not
pin the exact hit, and one wrong frame would corrupt every scored metric
(docs/feature-spec.md). The window is the plateau of near-maximum extension. A
single suspected-contact frame is also reported, for display only. Detection
carries a confidence verdict with reasons; on LOW confidence the caller must
surface "re-film / re-trim" instead of silently scoring.

Speeds and extensions are scale-invariant (torso-lengths and torso-length
ratios), so the same thresholds apply regardless of camera distance or player
size.

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
CONTACT_SEARCH_BACK_S = 0.12   # look for peak extension this far BEFORE the speed peak
CONTACT_SEARCH_FWD_S = 0.22    # ... and this far after (contact sits on/just after the peak)
WINDOW_EXTENSION_FRACTION = 0.95  # window = frames within this fraction of peak arm-extension
WINDOW_MAX_HALF = 4            # hard cap on window half-width, frames
APEX_SPEED_FRACTION = 0.3      # backswing apex = last lull below this before the peak
FOLLOW_END_FRACTION = 0.2      # follow-through ends when speed stays below this
PEAK_MEDIAN_RATIO_MIN = 3.0    # peak must stand this far above the clip's median speed
# Absolute swing-existence floor. Measured: real smashes peak at 29-41 tl/s
# (smash_coachhan 29.3, smash_2 40.7); non-swing camera motion peaks at ~0.5.
# 5.0 splits those populations with a wide margin on both sides.
MIN_PEAK_SPEED = 5.0          # torso-lengths/sec; below this, no swing happened
MULTI_SWING_MIN_GAP_S = 0.3   # overhead swings farther apart than this count as separate shots
MULTI_SWING_PEAK_FRACTION = 0.6  # a 2nd "swing" must be at least this fraction of the main one
                                 # (else it's a prep lift/blip, not a separate shot)
EDGE_MARGIN = 3               # peak this close to clip edge => swing likely cut off
MIN_WRIST_VISIBILITY = 0.5
MAX_MISSING_NEAR_CONTACT = 0.2


@dataclass
class SwingEvents:
    """Detected swing timeline for one clip, with detection confidence."""

    speed_peak: int                        # frame of max wrist speed (locates the swing)
    contact_frame: int                     # suspected hit: max arm extension (display only)
    contact_window: tuple[int, int]        # inclusive frame range that does the real measuring
    backswing_apex: int | None
    follow_through_end: int | None
    confidence: str                        # "high" | "low"
    reasons: list[str] = field(default_factory=list)   # why confidence is low
    wrist_speed: np.ndarray = field(default=None, repr=False)     # torso-lengths/sec, per frame
    arm_extension: np.ndarray = field(default=None, repr=False)   # wrist-shoulder / torso, per frame
    fps: float = 30.0

    def frame_to_seconds(self, frame: int) -> float:
        return frame / self.fps

    def summary(self) -> str:
        lines = []
        w0, w1 = self.contact_window
        if self.backswing_apex is not None:
            lines.append(f"backswing apex   frame {self.backswing_apex}  ({self.frame_to_seconds(self.backswing_apex):.2f}s)")
        lines.append(
            f"CONTACT window   frames {w0}-{w1}  ({self.frame_to_seconds(w0):.2f}-{self.frame_to_seconds(w1):.2f}s)"
        )
        lines.append(
            f"suspected hit    frame {self.contact_frame}  ({self.frame_to_seconds(self.contact_frame):.2f}s, "
            f"max arm extension); swing speed peak at frame {self.speed_peak}"
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


def _cluster_peaks(peaks: list[int], gap: int) -> list[list[int]]:
    """Group frame indices that are within `gap` frames of each other.

    One swing produces several nearby peaks (upswing, follow-through); those
    collapse to one cluster. Two clusters => two separate shots in the clip.
    """
    clusters: list[list[int]] = []
    for p in sorted(peaks):
        if clusters and p - clusters[-1][-1] <= gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return clusters


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
    sx, sy = track(shoulder_i, 0, seq.width), track(shoulder_i, 1, seq.height)
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

    # Smooth positions before differentiating / measuring, to kill pose jitter.
    if n >= 7:
        wx, wy = savgol_filter(wx, 7, 2), savgol_filter(wy, 7, 2)
        sx, sy = savgol_filter(sx, 7, 2), savgol_filter(sy, 7, 2)

    # Wrist speed (torso-lengths/sec) — locates the swing in time.
    speed = np.zeros(n)
    speed[1:] = np.hypot(np.diff(wx), np.diff(wy)) * seq.fps / torso_len
    # Arm extension (wrist-to-shoulder / torso) — pinpoints the hit.
    extension = np.hypot(wx - sx, wy - sy) / torso_len

    reasons: list[str] = []

    # --- locate the swing: strongest OVERHEAD wrist-speed peak ---
    peaks, _ = find_peaks(speed, prominence=np.max(speed) * 0.3)
    overhead = [p for p in peaks if wy[p] < sy[p]]  # image y grows downward
    if overhead:
        speed_peak = int(max(overhead, key=lambda p: speed[p]))
    elif len(peaks):
        speed_peak = int(max(peaks, key=lambda p: speed[p]))
        reasons.append("wrist is not above the shoulder at the speed peak — overhead swing not confirmed")
    else:
        speed_peak = int(np.argmax(speed))
        reasons.append("no distinct speed peak found — clip may not contain a swing")

    peak_speed = speed[speed_peak]
    if peak_speed < MIN_PEAK_SPEED:
        reasons.append(
            f"peak wrist speed ({peak_speed:.1f} torso-lengths/s) is far below any real swing — "
            "this clip does not appear to contain a shot"
        )
    median_speed = float(np.median(speed))
    if median_speed > 0 and peak_speed / median_speed < PEAK_MEDIAN_RATIO_MIN:
        reasons.append("speed peak barely stands out from baseline motion — contact timing unreliable")
    if speed_peak < EDGE_MARGIN or speed_peak > n - 1 - EDGE_MARGIN:
        reasons.append("speed peak is at the very edge of the clip — the swing looks cut off; re-trim with margin")

    # --- more-than-one-swing check: the clip should hold a single shot ---
    # Only count peaks that are swing-sized relative to the main swing, so a
    # preparatory racket lift just over MIN_PEAK_SPEED is not read as a 2nd shot.
    swing_floor = max(MIN_PEAK_SPEED, MULTI_SWING_PEAK_FRACTION * peak_speed)
    strong = [p for p in overhead if speed[p] >= swing_floor]
    clusters = _cluster_peaks(strong, gap=int(round(MULTI_SWING_MIN_GAP_S * seq.fps)))
    if len(clusters) >= 2:
        reasons.append(
            f"this clip appears to contain {len(clusters)} overhead swings — trim to one shot "
            "(or split the clip) so a single contact can be scored"
        )

    # --- suspected contact: max arm extension in a window around the swing ---
    back = max(3, int(round(CONTACT_SEARCH_BACK_S * seq.fps)))
    fwd = max(4, int(round(CONTACT_SEARCH_FWD_S * seq.fps)))
    s0, s1 = max(0, speed_peak - back), min(n, speed_peak + fwd + 1)
    contact_frame = s0 + int(np.argmax(extension[s0:s1]))

    # --- contact window: contiguous frames at near-peak extension, hard-capped ---
    ext_thresh = extension[contact_frame] * WINDOW_EXTENSION_FRACTION
    lo = contact_frame
    while lo > 0 and contact_frame - lo < WINDOW_MAX_HALF and extension[lo - 1] >= ext_thresh:
        lo -= 1
    hi = contact_frame
    while hi < n - 1 and hi - contact_frame < WINDOW_MAX_HALF and extension[hi + 1] >= ext_thresh:
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
    for i in range(min(lo, speed_peak) - 1, 0, -1):
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
        speed_peak=speed_peak,
        contact_frame=contact_frame,
        contact_window=(lo, hi),
        backswing_apex=apex,
        follow_through_end=follow_end,
        confidence="low" if reasons else "high",
        reasons=reasons,
        wrist_speed=speed,
        arm_extension=extension,
        fps=seq.fps,
    )
