"""Canonicalization: every clip becomes a right-handed player with a known facing.

Handedness is resolved ONCE here (docs/feature-spec.md): a left-handed clip is
mirrored horizontally and its left/right landmarks swapped, so downstream code
always reads the racket arm from the RIGHT_* indices and never thinks about
handedness again. User and reference clips both pass through this, so a
left-handed reference can score a right-handed user and vice versa.

Facing — which way along image-x the player faces — is detected per clip so
"in front of the body" has a sign that is body-relative, not net/court-relative
(a locked decision in CLAUDE.md). The feet are the primary signal: toes sit in
front of heels in any playing stance. The nose-vs-shoulders offset is the
fallback when the feet are edge-on or poorly tracked.
"""

from __future__ import annotations

import numpy as np

from ..pose import landmarks as lm
from ..pose.extractor import FramePose, PoseSequence

# (left, right) index pairs exchanged when a clip is mirrored.
_SWAP_PAIRS = [
    (lm.LEFT_EYE_INNER, lm.RIGHT_EYE_INNER),
    (lm.LEFT_EYE, lm.RIGHT_EYE),
    (lm.LEFT_EYE_OUTER, lm.RIGHT_EYE_OUTER),
    (lm.LEFT_EAR, lm.RIGHT_EAR),
    (lm.MOUTH_LEFT, lm.MOUTH_RIGHT),
    (lm.LEFT_SHOULDER, lm.RIGHT_SHOULDER),
    (lm.LEFT_ELBOW, lm.RIGHT_ELBOW),
    (lm.LEFT_WRIST, lm.RIGHT_WRIST),
    (lm.LEFT_PINKY, lm.RIGHT_PINKY),
    (lm.LEFT_INDEX, lm.RIGHT_INDEX),
    (lm.LEFT_THUMB, lm.RIGHT_THUMB),
    (lm.LEFT_HIP, lm.RIGHT_HIP),
    (lm.LEFT_KNEE, lm.RIGHT_KNEE),
    (lm.LEFT_ANKLE, lm.RIGHT_ANKLE),
    (lm.LEFT_HEEL, lm.RIGHT_HEEL),
    (lm.LEFT_FOOT_INDEX, lm.RIGHT_FOOT_INDEX),
]

# Facing votes must mostly agree, or the frontness sign is a coin flip and the
# caller has to say so. Detection plumbing, not a technique threshold.
FACING_MIN_AGREEMENT = 0.8


def canonicalize(seq: PoseSequence, handedness: str = "right") -> PoseSequence:
    """Return `seq` in the canonical right-handed frame.

    Right-handed input is returned as-is. Left-handed input gets a mirrored
    COPY: normalized x -> 1 - x (world x -> -x) and left/right landmarks
    swapped, so the racket arm lands on the RIGHT_* indices.
    """
    if handedness == "right":
        return seq
    if handedness != "left":
        raise ValueError(f"handedness must be 'right' or 'left', got {handedness!r}")

    def mirror(points: list[list[float]] | None, world: bool) -> list[list[float]] | None:
        if points is None:
            return None
        out = [p.copy() for p in points]
        for a, b in _SWAP_PAIRS:
            out[a], out[b] = out[b], out[a]
        for p in out:
            p[0] = -p[0] if world else 1.0 - p[0]
        return out

    mirrored = PoseSequence(source=seq.source, fps=seq.fps, width=seq.width, height=seq.height)
    mirrored.frames = [
        FramePose(
            index=f.index,
            timestamp_ms=f.timestamp_ms,
            landmarks=mirror(f.landmarks, world=False),
            world_landmarks=mirror(f.world_landmarks, world=True),
        )
        for f in seq.frames
    ]
    return mirrored


def detect_facing(seq: PoseSequence) -> tuple[int, float]:
    """Which way the player faces along image-x, from a canonicalized sequence.

    Returns (facing, agreement): facing = +1 if the player faces increasing x,
    -1 otherwise; agreement = fraction of voting frames that agree with the
    verdict. Each detected frame votes with its heel->toe direction (both feet
    summed); frames where the feet are edge-on (near-zero) vote with the
    nose-vs-mid-shoulder offset instead.
    """
    votes: list[float] = []
    for f in seq.frames:
        if not f.detected:
            continue
        p = f.landmarks
        feet = (p[lm.LEFT_FOOT_INDEX][0] - p[lm.LEFT_HEEL][0]) + (
            p[lm.RIGHT_FOOT_INDEX][0] - p[lm.RIGHT_HEEL][0]
        )
        if feet != 0.0:
            votes.append(feet)
        else:
            mid_shoulder = (p[lm.LEFT_SHOULDER][0] + p[lm.RIGHT_SHOULDER][0]) / 2
            votes.append(p[lm.NOSE][0] - mid_shoulder)

    if not votes:
        raise ValueError("no pose detected in any frame — cannot determine facing")

    v = np.array(votes)
    # Magnitude-weighted verdict: strong, clear frames outvote edge-on jitter.
    facing = 1 if float(v.sum()) >= 0 else -1
    agreement = float(np.mean(np.sign(v) == facing)) if len(v) else 0.0
    return facing, agreement
