"""At-contact metrics: collapse the feature time series onto the contact window.

The window (from segment/events.py) does the measuring, never a single trusted
frame. The anchor is the max-reach frame — highest racket wrist — WITHIN the
window (docs/feature-spec.md: "contact height uses the max-reach frame"), and
the other instantaneous metrics are read at that same frame so they all
describe one moment.

These are MEASUREMENTS with recorded context, not judgments. Contact height's
gating inputs (P6 frontness, P7 elbow extension) are reported alongside it so
the rule engine (Stage 4a) can refuse to credit height earned by reaching
behind the head — but the crediting itself happens there, against sourced
thresholds, not here. Detection confidence is carried through unchanged: a LOW
confidence window can never silently produce trusted numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..segment.events import SwingEvents
from .canonical import FACING_MIN_AGREEMENT
from .features import FeatureSeries


@dataclass
class ContactMetrics:
    """Scored quantities for one swing, measured across the contact window."""

    anchor_frame: int                    # max-reach frame within the window
    window: tuple[int, int]
    contact_height: float                # torso lengths above (+) the racket shoulder
    elbow_angle: float                   # degrees; 180 = straight (P7 gate input)
    frontness: float                     # torso lengths in front (+) of the shoulder (P6 gate input)
    wrist_speed: float                   # torso lengths / s at the anchor
    # Swing-shape metrics; None when the anchoring event was not detected.
    trunk_incline_backswing: float | None    # degrees at backswing apex (- = arched back)
    trunk_incline_followthrough: float | None  # degrees at follow-through end (+ = flexed forward)
    trunk_arch_to_flex: float | None     # follow-through minus backswing incline
    left_arm_drop: float | None          # non-racket elevation lost, apex -> contact (+ = pulled down)
    confidence: str                      # "high" | "low", carried from event detection
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        w0, w1 = self.window

        def fmt(v: float | None, spec: str, missing: str) -> str:
            return format(v, spec) if v is not None else missing

        lines = [
            f"at-contact metrics  (anchor frame {self.anchor_frame}, window {w0}-{w1})",
            f"  contact height    {self.contact_height:+.2f} torso lengths vs shoulder",
            f"  elbow angle       {self.elbow_angle:.0f} deg  (180 = straight)",
            f"  frontness         {self.frontness:+.2f} torso lengths vs shoulder",
            f"  wrist speed       {self.wrist_speed:.1f} torso lengths/s",
            f"  trunk incline     backswing {fmt(self.trunk_incline_backswing, '+.0f', '--')} deg"
            f" -> follow-through {fmt(self.trunk_incline_followthrough, '+.0f', '--')} deg"
            f"  (arch-to-flex {fmt(self.trunk_arch_to_flex, '+.0f', '--')} deg)",
            f"  left-arm pull     {fmt(self.left_arm_drop, '+.0f', '--')} deg drop, backswing apex -> contact",
            f"confidence        {self.confidence.upper()}",
        ]
        lines += [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def measure_contact(features: FeatureSeries, events: SwingEvents) -> ContactMetrics:
    """Read the at-contact metrics off the feature layer at the detected window."""
    lo, hi = events.contact_window
    height = features.array("right_wrist_height")
    anchor = lo + int(np.argmax(height[lo : hi + 1]))

    reasons = list(events.reasons)
    if features.facing_agreement < FACING_MIN_AGREEMENT:
        reasons.append(
            f"facing direction ambiguous ({features.facing_agreement:.0%} agreement) — "
            "frontness sign unreliable; re-film side-on"
        )

    def at_event(name: str, frame: int | None) -> float | None:
        return float(features.array(name)[frame]) if frame is not None else None

    trunk_back = at_event("trunk_incline", events.backswing_apex)
    trunk_follow = at_event("trunk_incline", events.follow_through_end)
    left_elev_apex = at_event("left_arm_elevation", events.backswing_apex)

    return ContactMetrics(
        anchor_frame=anchor,
        window=(lo, hi),
        contact_height=float(height[anchor]),
        elbow_angle=float(features.array("right_elbow_angle")[anchor]),
        frontness=float(features.array("right_wrist_frontness")[anchor]),
        wrist_speed=float(features.array("right_wrist_speed")[anchor]),
        trunk_incline_backswing=trunk_back,
        trunk_incline_followthrough=trunk_follow,
        trunk_arch_to_flex=(
            trunk_follow - trunk_back if trunk_back is not None and trunk_follow is not None else None
        ),
        left_arm_drop=(
            left_elev_apex - float(features.array("left_arm_elevation")[anchor])
            if left_elev_apex is not None
            else None
        ),
        confidence="low" if reasons else "high",
        reasons=reasons,
    )
