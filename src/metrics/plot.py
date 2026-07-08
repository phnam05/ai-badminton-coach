"""Stage 3 checkpoint artifact: one sheet, every feature curve.

The user eyeballs this against the overlay video before Stage 4a is built on
top: are the curves smooth (no tracking jitter), does the shaded contact
window sit on the visually-confirmed hit, do the values move the way the swing
looks (elbow straightens into contact, wrist rises then falls, trunk goes
arch -> flex)?

Every panel shares the time axis, with the contact window shaded and the
anchor (max-reach) frame marked, so a metric that misbehaves is caught at the
moment that matters.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..segment.events import SwingEvents
from .contact import ContactMetrics
from .features import FEATURE_INFO, FeatureSeries

# Same reference dataviz palette as segment/plot.py (light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"        # slot 1 blue — feature curves
CONTACT = "#e34948"       # slot 6 red  — contact window / anchor frame
WEAK = "#898781"          # flagged-weak features drawn muted


def _style(ax) -> None:
    """Recessive chrome: hairline grid, muted axes, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)


def render_feature_sheet(
    features: FeatureSeries,
    events: SwingEvents,
    contact: ContactMetrics,
    out_path: str | Path,
    title: str = "",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(FEATURE_INFO)
    fps = features.fps
    w0, w1 = contact.window
    # Half-frame padding so even a one-frame window is a visible band.
    band = (max(0.0, (w0 - 0.5) / fps), (w1 + 0.5) / fps)
    t_anchor = contact.anchor_frame / fps

    ncols = 3
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12.5, 2.6 * nrows), dpi=150, sharex=True,
        gridspec_kw={"hspace": 0.55, "wspace": 0.32},
    )
    fig.patch.set_facecolor(SURFACE)

    for ax, name in zip(axes.flat, names):
        unit, meaning = FEATURE_INFO[name]
        series = features.array(name)
        t = [i / fps for i in range(len(series))]
        weak = "weak" in meaning

        _style(ax)
        ax.axvspan(*band, color=CONTACT, alpha=0.08, zorder=1)
        ax.axvline(t_anchor, color=CONTACT, linewidth=1.3, zorder=3)
        ax.plot(t, series, color=WEAK if weak else SERIES, linewidth=1.6, zorder=4)
        ax.plot(t_anchor, series[contact.anchor_frame], "o", markersize=5.5,
                color=CONTACT, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5)
        ax.set_title(f"{name}\n{meaning}", color=INK_SECONDARY, fontsize=8.5,
                     loc="left", pad=4)
        ax.set_ylabel(unit, color=MUTED, fontsize=8)

    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color=INK_SECONDARY, fontsize=8.5)

    head = "Feature layer"
    if title:
        head += f" — {title}"
    fig.suptitle(
        f"{head}   ·   contact window shaded, anchor frame {contact.anchor_frame} marked"
        f"   ·   detection confidence: {events.confidence.upper()}",
        color=INK, fontsize=11, x=0.01, horizontalalignment="left",
    )

    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out_path
