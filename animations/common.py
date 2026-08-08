"""Shared plotting helpers for all animation scripts.

Every video is built the same way: a script assembles a list of "frame"
dicts describing what should be on screen, and a single draw function
repaints the whole figure for each frame.  Redrawing from scratch is a
little wasteful but keeps every animation script dead simple.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render off-screen; we only write video files
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch

BG = "#ffffff"
INK = "#222222"
MUTED = "#888888"
ACCENT = "#d62728"       # highlight color (merges, chosen tokens, ...)
HEAT_CMAP = "viridis"    # for attention weights (0..1)
VAL_CMAP = "RdBu_r"      # for signed value matrices (embeddings, Q/K/V)

_PALETTE = plt.get_cmap("tab20")


def token_color(token_id):
    """A stable pastel color per token id, so tokens are recognizable."""
    r, g, b, _ = _PALETTE(token_id % 20)
    # Blend toward white so black text stays readable on top.
    mix = 0.55
    return (1 - mix + mix * r, 1 - mix + mix * g, 1 - mix + mix * b)


def draw_token_row(ax, tokens, ids=None, y=0.5, x0=0.02, char_w=0.0135,
                   pad=0.006, height=0.28, fontsize=11, highlight=(),
                   highlight_color=ACCENT, dim=(), show_ids=False,
                   max_x=0.98, line_gap=0.42):
    """Draw tokens as rounded boxes laid out left to right, wrapping.

    ``highlight`` and ``dim`` are sets of token *positions*.  Returns the
    (x_center, y_center) of every box so callers can point at them.
    """
    centers = []
    x, yy = x0, y
    for i, tok in enumerate(tokens):
        label = tok.replace(" ", "·").replace(chr(10), "\\n")  # make spaces visible as ·
        w = char_w * max(len(label), 1) + 2 * pad
        if x + w > max_x:                  # wrap to the next line
            x = x0
            yy -= line_gap
        color = token_color(ids[i]) if ids is not None else "#dddddd"
        edge = highlight_color if i in highlight else "#aaaaaa"
        lw = 2.2 if i in highlight else 0.8
        alpha = 0.25 if i in dim else 1.0
        box = FancyBboxPatch(
            (x, yy - height / 2), w, height,
            boxstyle="round,pad=0.004,rounding_size=0.01",
            linewidth=lw, edgecolor=edge, facecolor=color, alpha=alpha,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, yy, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=fontsize,
                family="monospace", color=INK, alpha=alpha)
        if show_ids and ids is not None:
            ax.text(x + w / 2, yy - height / 2 - 0.10, str(ids[i]),
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=fontsize - 3, color=MUTED, alpha=alpha)
        centers.append((x + w / 2, yy))
        x += w + 0.008
    return centers


def blank(ax):
    """Strip an axes down to an empty canvas."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def heatmap(ax, matrix, title=None, cmap=VAL_CMAP, vmin=None, vmax=None,
            xticklabels=None, yticklabels=None, fontsize=7):
    """Small labeled heatmap used for all the matrix views."""
    if vmin is None and cmap == VAL_CMAP:
        vmax = max(abs(matrix.min()), abs(matrix.max()), 1e-6)
        vmin = -vmax
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=9, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels([t.replace(" ", "·").replace(chr(10), "\\n") for t in xticklabels],
                           fontsize=fontsize, family="monospace", rotation=90)
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels([t.replace(" ", "·").replace(chr(10), "\\n") for t in yticklabels],
                           fontsize=fontsize, family="monospace")
    return im


def caption(fig, text):
    """One shared explanatory line at the bottom of the figure."""
    fig.text(0.5, 0.035, text, ha="center", va="center",
             fontsize=11.5, color=INK, wrap=True)


def save_animation(fig, draw_frame, num_frames, out_path, fps, dpi):
    """Render ``num_frames`` calls of ``draw_frame`` to an mp4 (or gif)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    anim = animation.FuncAnimation(fig, draw_frame, frames=num_frames,
                                   interval=1000 / fps)
    if animation.FFMpegWriter.isAvailable():
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    else:  # fallback when ffmpeg is missing
        out_path = out_path.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out_path} ({num_frames} frames @ {fps} fps)")
    return out_path
