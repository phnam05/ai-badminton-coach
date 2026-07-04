"""Wrist-speed plot: shows WHY the contact window was chosen.

This is the Stage 2 checkpoint artifact — the user compares the marked window
against the overlay video to confirm the contact frame is right.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .events import SwingEvents

# Reference dataviz palette (light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"     # wrist speed line
CONTACT = "#e34948"    # contact peak marker


def render_speed_plot(events: SwingEvents, out_path: str | Path, title: str = "") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    speed = events.wrist_speed
    t = [i / events.fps for i in range(len(speed))]
    w0, w1 = events.contact_window

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Recessive chrome: hairline grid, muted axes, no top/right spines.
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)

    # Contact window band + peak marker.
    ax.axvspan(t[w0], t[w1], color=SERIES, alpha=0.12, zorder=1)
    ax.axvline(t[events.contact_peak], color=CONTACT, linewidth=1.6, zorder=3)
    y_top = float(speed.max())
    ax.annotate(
        "contact", (t[events.contact_peak], y_top), xytext=(4, 2),
        textcoords="offset points", color=INK_SECONDARY, fontsize=9,
    )

    # Secondary event markers, direct-labeled in ink (never series color).
    for frame, label in ((events.backswing_apex, "backswing apex"), (events.follow_through_end, "follow-through end")):
        if frame is not None:
            ax.axvline(t[frame], color=BASELINE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
            ax.annotate(
                label, (t[frame], y_top * 0.92), xytext=(4, 0),
                textcoords="offset points", color=MUTED, fontsize=8.5,
            )

    # The single series (title names it — no legend needed).
    ax.plot(t, speed, color=SERIES, linewidth=2, zorder=4)

    ax.set_xlabel("time (s)", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel("wrist speed (torso lengths / s)", color=INK_SECONDARY, fontsize=9.5)
    head = "Racket-wrist speed"
    if title:
        head += f" — {title}"
    conf = f"detection confidence: {events.confidence.upper()}"
    ax.set_title(f"{head}\n{conf}", color=INK, fontsize=11, loc="left", pad=10)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path
