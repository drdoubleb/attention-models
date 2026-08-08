"""Video 2 — Embeddings and positions.

Token IDs are just numbers; the model turns each one into a vector by
looking up a row of the embedding matrix E, then adds a positional
vector so the model can tell *where* each token sits.  The video walks
through the probe sentence one token at a time, showing the lookup, the
addition, and the input matrix X filling up row by row.
"""

import numpy as np
import matplotlib.pyplot as plt

from .common import (INK, MUTED, ACCENT, blank, caption, draw_token_row,
                     heatmap, save_animation, VAL_CMAP)


def render(run, cfg):
    tokenizer = run["tokenizer"]
    model = run["model"]
    probe_ids = run["probe_ids"]
    probe_tokens = [tokenizer.vocab[i] for i in probe_ids]
    cache = run["probe_cache"]

    E = model.params["E"]
    P = model.params["P"]
    T = len(probe_ids)
    D = model.d_model

    # frames: intro, then 2 per token (lookup / add position), then outro
    frames = [("intro", None)]
    for i in range(T):
        frames.append(("lookup", i))
        frames.append(("add", i))
    frames += [("outro", None)] * 3

    fig = plt.figure(figsize=(12.8, 7.2))
    ax_tokens = fig.add_axes([0.02, 0.80, 0.96, 0.16])
    ax_E = fig.add_axes([0.05, 0.12, 0.20, 0.60])       # embedding matrix
    ax_tok_vec = fig.add_axes([0.33, 0.60, 0.28, 0.07])  # E[id]
    ax_pos_vec = fig.add_axes([0.33, 0.44, 0.28, 0.07])  # P[i]
    ax_sum_vec = fig.add_axes([0.33, 0.26, 0.28, 0.07])  # sum
    ax_X = fig.add_axes([0.70, 0.14, 0.26, 0.58])        # growing input matrix

    vmax = float(max(abs(E).max(), abs(P).max()))

    def draw(t):
        fig.texts.clear()
        for a in (ax_tokens, ax_E, ax_tok_vec, ax_pos_vec, ax_sum_vec, ax_X):
            a.clear()
        blank(ax_tokens)
        fig.suptitle("From token IDs to vectors: embeddings + positions",
                     fontsize=15, color=INK, y=0.985)

        kind, i = frames[t]
        done = T if kind == "outro" else (0 if kind == "intro" else i + (kind == "add"))

        draw_token_row(ax_tokens, probe_tokens, ids=probe_ids, y=0.45,
                       highlight={i} if i is not None else set(),
                       show_ids=True, char_w=0.011, fontsize=11)

        heatmap(ax_E, E, title=f"embedding matrix E   ({E.shape[0]} tokens × {D} dims)",
                vmin=-vmax, vmax=vmax)
        ax_E.set_ylabel("one row per vocabulary token", fontsize=8, color=MUTED)

        # Growing input matrix X: rows appear as tokens are processed.
        X_shown = np.full((T, D), np.nan)
        if done:
            X_shown[:done] = cache["x0"][:done]
        heatmap(ax_X, np.nan_to_num(X_shown),
                title=f"input matrix X   ({T} tokens × {D} dims)",
                vmin=-vmax, vmax=vmax,
                yticklabels=probe_tokens, fontsize=8)
        for r in range(done, T):   # mask not-yet-filled rows
            ax_X.axhspan(r - 0.5, r + 0.5, color="white", zorder=3)

        if kind == "intro":
            for a in (ax_tok_vec, ax_pos_vec, ax_sum_vec):
                blank(a)
            caption(fig, "Each token ID selects one row of the embedding matrix E — "
                         "a learned vector of numbers that represents that token.")
            return
        if kind == "outro":
            for a in (ax_tok_vec, ax_pos_vec, ax_sum_vec):
                blank(a)
            caption(fig, "The full input matrix X: one vector per token, position "
                         "information included. This is what flows into attention.")
            return

        tok_id = probe_ids[i]
        tok = probe_tokens[i].replace(" ", "·").replace(chr(10), "\\n")

        # Highlight the row of E being read.
        ax_E.add_patch(plt.Rectangle((-0.5, tok_id - 0.5), E.shape[1], 1,
                                     fill=False, edgecolor=ACCENT, lw=2, zorder=4))

        heatmap(ax_tok_vec, E[tok_id][None, :],
                title=f"E[{tok_id}]  —  embedding of '{tok}'", vmin=-vmax, vmax=vmax)
        if kind == "lookup":
            for a in (ax_pos_vec, ax_sum_vec):
                blank(a)
            caption(fig, f"Position {i}: token '{tok}' has ID {tok_id}, "
                         f"so we read row {tok_id} of E.")
        else:
            heatmap(ax_pos_vec, P[i][None, :],
                    title=f"P[{i}]  —  positional vector for slot {i}",
                    vmin=-vmax, vmax=vmax)
            heatmap(ax_sum_vec, cache["x0"][i][None, :],
                    title=f"X[{i}] = E[{tok_id}] + P[{i}]", vmin=-vmax, vmax=vmax)
            for a in (ax_pos_vec, ax_sum_vec):
                a.set_xticks([])
            ax_pos_vec.text(-0.03, 0.5, "+", transform=ax_pos_vec.transAxes,
                            ha="right", va="center", fontsize=20, color=INK)
            ax_sum_vec.text(-0.03, 0.5, "=", transform=ax_sum_vec.transAxes,
                            ha="right", va="center", fontsize=20, color=INK)
            caption(fig, f"Add the positional vector P[{i}] so the model knows this "
                         f"'{tok}' sits in slot {i} — the sum becomes row {i} of X.")

    return save_animation(fig, draw, len(frames),
                          f"{cfg.OUTPUT_DIR}/videos/2_embeddings.mp4",
                          fps=cfg.VIDEO_FPS, dpi=cfg.VIDEO_DPI)
