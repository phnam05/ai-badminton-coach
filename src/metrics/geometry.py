"""Angle/ratio primitives for the feature layer.

All helpers are vectorized over time: inputs are per-frame coordinate arrays
(pixels, image y grows downward), outputs are per-frame values. Angles are in
DEGREES. These are pure geometry — no badminton knowledge lives here.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def joint_angle(
    ax: np.ndarray, ay: np.ndarray,
    bx: np.ndarray, by: np.ndarray,
    cx: np.ndarray, cy: np.ndarray,
) -> np.ndarray:
    """Interior angle at vertex B of A-B-C, degrees in [0, 180].

    180 = A, B, C in a straight line (a fully extended joint).
    """
    return vector_angle(ax - bx, ay - by, cx - bx, cy - by)


def vector_angle(ux: np.ndarray, uy: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Unsigned angle between two per-frame vectors, degrees in [0, 180]."""
    dot = ux * vx + uy * vy
    norm = np.hypot(ux, uy) * np.hypot(vx, vy)
    cos = np.clip(dot / np.maximum(norm, _EPS), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def incline_vs_vertical(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Signed tilt of a segment vs vertical, degrees.

    `forward` and `up` are the segment's components along the caller's forward
    and upward directions. 0 = perfectly upright; positive = tipped toward
    forward; negative = tipped away (e.g. a trunk arching back).
    """
    return np.degrees(np.arctan2(forward, up))
