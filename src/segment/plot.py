"""Contact-checkpoint chart: shows WHERE the hit is marked and WHY.

This is the Stage 2 checkpoint artifact — the user compares the marked contact
against the overlay video to confirm it is right.

Two stacked panels share one time axis:

  * top  — racket-wrist SPEED (torso-lengths/sec): locates the swing.
  * bottom — ARM EXTENSION (wrist-to-shoulder / torso): pinpoints the hit.

A single red "suspected hit" line drops through both panels at the contact
frame, so the eye can see the gap: speed peaks first (top), and a few frames
later extension reaches its max right under the red line (bottom). That gap is
exactly why wrist speed alone mistimed contact. The two measures have different
units, so they get their own panels sharing the time axis — never two y-scales
on one plot, which would let the alignment be faked by rescaling.
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
SERIES_SPEED = "#2a78d6"   # slot 1 blue  — wrist speed
SERIES_EXT = "#1baf7a"     # slot 2 aqua  — arm extension
CONTACT = "#e34948"        # slot 6 red   — suspected hit / contact window


def _style(ax) -> None:
    """Recessive chrome: hairline grid, muted axes, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)


def render_speed_plot(events: SwingEvents, out_path: str | Path, title: str = "") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fps = events.fps
    speed = events.wrist_speed
    extension = events.arm_extension
    t = [i / fps for i in range(len(speed))]
    w0, w1 = events.contact_window
    # Half-frame padding so even a one-frame window is a visible band.
    band_lo, band_hi = max(0.0, (w0 - 0.5) / fps), (w1 + 0.5) / fps
    t_contact, t_swing = t[events.contact_frame], t[events.speed_peak]

    fig, (ax_s, ax_e) = plt.subplots(
        2, 1, figsize=(9, 5.8), dpi=150, sharex=True,
        gridspec_kw={"hspace": 0.14},
    )
    fig.patch.set_facecolor(SURFACE)
    _style(ax_s)
    _style(ax_e)

    # Shared markers on BOTH panels: contact window band + swing/contact guides.
    for ax in (ax_s, ax_e):
        ax.axvspan(band_lo, band_hi, color=CONTACT, alpha=0.08, zorder=1)
        ax.axvline(t_swing, color=BASELINE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.axvline(t_contact, color=CONTACT, linewidth=1.6, zorder=3)

    # --- top: wrist speed, with backswing/follow-through landmarks ---
    # Apex label grows left, follow-through grows right, so neither collides
    # with the "swing (speed peak)" label sitting between them at the top.
    y_s = float(speed.max())
    for frame, label, dx, ha in ((events.backswing_apex, "backswing apex", -4, "right"),
                                 (events.follow_through_end, "follow-through end", 4, "left")):
        if frame is not None:
            ax_s.axvline(t[frame], color=BASELINE, linewidth=1.0, linestyle=(0, (2, 3)), zorder=2)
            ax_s.annotate(label, (t[frame], y_s * 0.86), xytext=(dx, 0),
                          textcoords="offset points", color=MUTED, fontsize=8.5,
                          horizontalalignment=ha)
    ax_s.plot(t, speed, color=SERIES_SPEED, linewidth=2, zorder=4)
    ax_s.plot(t_swing, speed[events.speed_peak], "o", markersize=7,
              color=SERIES_SPEED, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=5)
    ax_s.annotate("swing (speed peak)", (t_swing, y_s), xytext=(4, 2),
                  textcoords="offset points", color=INK_SECONDARY, fontsize=9)
    ax_s.set_ylabel("wrist speed\n(torso lengths / s)", color=INK_SECONDARY, fontsize=9.5)

    # --- bottom: arm extension, contact marked at its max ---
    e_lo, e_hi = float(extension.min()), float(extension.max())
    ax_e.plot(t, extension, color=SERIES_EXT, linewidth=2, zorder=4)
    ax_e.plot(t_contact, extension[events.contact_frame], "o", markersize=8,
              color=CONTACT, markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=5)
    ax_e.annotate("suspected hit\n(max extension)", (t_contact, e_hi), xytext=(4, 2),
                  textcoords="offset points", color=INK_SECONDARY, fontsize=9,
                  verticalalignment="top")
    ax_e.set_ylim(e_lo - 0.05 * (e_hi - e_lo), e_hi + 0.18 * (e_hi - e_lo))
    ax_e.set_ylabel("arm extension\n(wrist-to-shoulder / torso)", color=INK_SECONDARY, fontsize=9.5)
    ax_e.set_xlabel("time (s)", color=INK_SECONDARY, fontsize=9.5)

    head = "Swing speed and arm extension"
    if title:
        head += f" — {title}"
    conf = f"detection confidence: {events.confidence.upper()}"
    ax_s.set_title(f"{head}\n{conf}", color=INK, fontsize=11, loc="left", pad=10)

    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out_path
