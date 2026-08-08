"""Video 3 — Attention, step by step (the centerpiece).

Using the trained weights on the probe sentence, the video first shows
the input X being projected into queries, keys and values, then walks
through every position: the query is compared against all earlier keys
(dot products), the scores go through softmax to become weights, and the
value vectors are blended by those weights.  The attention matrix fills
in row by row and is held at the end.
"""

import numpy as np
import matplotlib.pyplot as plt

from .common import (INK, MUTED, ACCENT, blank, caption, draw_token_row,
                     heatmap, save_animation, HEAT_CMAP)


def render(run, cfg):
    tokenizer = run["tokenizer"]
    model = run["model"]
    probe_ids = run["probe_ids"]
    tokens = [tokenizer.vocab[i] for i in probe_ids]
    cache = run["probe_cache"]
    T = len(probe_ids)
    D = model.d_model

    # Frame plan: 4 intro frames (X, Q, K, V), then 3 per query position
    # (scores, softmax, blend), then 3 hold frames on the finished matrix.
    frames = [("proj", 0), ("proj", 1), ("proj", 2), ("proj", 3)]
    for i in range(T):
        frames += [("scores", i), ("softmax", i), ("blend", i)]
    frames += [("final", None)] * 3

    fig = plt.figure(figsize=(12.8, 7.6))
    ax_tokens = fig.add_axes([0.02, 0.83, 0.96, 0.13])
    ax_A = fig.add_axes([0.11, 0.22, 0.30, 0.52])        # attention matrix
    ax_bar = fig.add_axes([0.52, 0.47, 0.44, 0.27])      # scores / weights
    ax_low = fig.add_axes([0.52, 0.14, 0.44, 0.22])      # V blend / projections

    proj_specs = [
        ("X", cache["x0"], "input matrix X (from video 2)",
         "Attention starts from X — the embedded tokens."),
        ("Q", cache["Q"], "Q = X · Wq   (queries)",
         "Each token makes a QUERY: a vector describing what it is looking for."),
        ("K", cache["K"], "K = X · Wk   (keys)",
         "Each token makes a KEY: a vector describing what it can offer."),
        ("V", cache["V"], "V = X · Wv   (values)",
         "Each token makes a VALUE: the actual information it will hand over if attended to."),
    ]

    def draw_A(upto_row, active_row=None, show_weights_in_active=True):
        """Attention matrix filled up to (and optionally including) a row."""
        A = cache["A"]
        M = np.zeros((T, T))
        if upto_row > 0:
            M[:upto_row] = A[:upto_row]
        if active_row is not None and show_weights_in_active:
            M[active_row] = A[active_row]
        heatmap(ax_A, M, title="attention weights A (each row sums to 1)",
                cmap=HEAT_CMAP, vmin=0, vmax=1,
                xticklabels=tokens, yticklabels=tokens, fontsize=8)
        ax_A.set_xlabel("keys (looked at)", fontsize=9, color=MUTED)
        ax_A.set_ylabel("queries (looking)", fontsize=9, color=MUTED)
        if active_row is not None:
            ax_A.add_patch(plt.Rectangle((-0.5, active_row - 0.5), T, 1,
                                         fill=False, edgecolor=ACCENT, lw=2.2,
                                         zorder=4))

    def draw(t):
        fig.texts.clear()
        for a in (ax_tokens, ax_A, ax_bar, ax_low):
            a.clear()
        blank(ax_tokens)
        fig.suptitle("Attention: every token decides which earlier tokens to look at",
                     fontsize=15, color=INK, y=0.985)

        kind, i = frames[t]
        tok_labels = [tk.replace(" ", "·").replace(chr(10), "\\n") for tk in tokens]

        if kind == "proj":
            name, mat, title, cap = proj_specs[i]
            draw_token_row(ax_tokens, tokens, ids=probe_ids, y=0.45, char_w=0.011)
            draw_A(0)
            heatmap(ax_bar, mat, title=title, yticklabels=tokens, fontsize=8)
            blank(ax_low)
            if name != "X":
                ax_low.text(0.5, 0.6, f"{name} = X · W{name.lower()}",
                            ha="center", fontsize=16, family="monospace", color=INK)
                ax_low.text(0.5, 0.3, f"({T}×{D}) = ({T}×{D}) · ({D}×{D})",
                            ha="center", fontsize=10, color=MUTED, family="monospace")
            caption(fig, cap)
            return

        if kind == "final":
            draw_token_row(ax_tokens, tokens, ids=probe_ids, y=0.45, char_w=0.011)
            draw_A(T)
            heatmap(ax_bar, cache["C"], title="output C = A · V  (blended values)",
                    yticklabels=tokens, fontsize=8)
            blank(ax_low)
            caption(fig, "The finished attention matrix: bright cells show which earlier "
                         "tokens each position pulled information from.")
            return

        # Per-query-position frames.
        qtok = tok_labels[i]
        draw_token_row(ax_tokens, tokens, ids=probe_ids, y=0.45,
                       highlight={i}, dim=set(range(i + 1, T)), char_w=0.011)

        scores = cache["S"][i, : i + 1]
        weights = cache["A"][i, : i + 1]
        xs = np.arange(i + 1)

        if kind == "scores":
            draw_A(i, active_row=i, show_weights_in_active=False)
            ax_bar.bar(xs, scores, color="#7f9fc4")
            ax_bar.set_title(f"scores: Q[{i}] · K[j] / √{D}   (query '{qtok}' vs each key)",
                             fontsize=9)
            ax_bar.set_xticks(xs)
            ax_bar.set_xticklabels(tok_labels[: i + 1], family="monospace",
                                   fontsize=8, rotation=45)
            ax_bar.axhline(0, color=MUTED, lw=0.6)
            ax_bar.set_xlim(-0.6, T - 0.4)
            blank(ax_low)
            ax_low.text(0.5, 0.5,
                        "future tokens are masked out —\na token may only look backwards",
                        ha="center", va="center", fontsize=11, color=MUTED)
            caption(fig, f"Position {i} ('{qtok}') compares its query with the key of every "
                         "earlier token — a dot product measures how well they match.")
        elif kind == "softmax":
            draw_A(i, active_row=i)
            ax_bar.bar(xs, weights, color="#2ca02c")
            ax_bar.set_title("softmax(scores) → attention weights (sum to 1)", fontsize=9)
            ax_bar.set_xticks(xs)
            ax_bar.set_xticklabels(tok_labels[: i + 1], family="monospace",
                                   fontsize=8, rotation=45)
            ax_bar.set_ylim(0, 1.05)
            ax_bar.set_xlim(-0.6, T - 0.4)
            blank(ax_low)
            best = int(np.argmax(weights))
            ax_low.text(0.5, 0.5,
                        f"'{qtok}' pays {weights[best]:.0%} of its attention to "
                        f"'{tok_labels[best]}'",
                        ha="center", va="center", fontsize=12, color=INK)
            caption(fig, "Softmax turns the raw scores into positive weights that sum to 1 — "
                         f"this becomes row {i} of the attention matrix.")
        else:  # blend
            draw_A(i + 1, active_row=i)
            ax_bar.bar(xs, weights, color="#2ca02c", alpha=0.9)
            ax_bar.set_title("attention weights", fontsize=9)
            ax_bar.set_xticks(xs)
            ax_bar.set_xticklabels(tok_labels[: i + 1], family="monospace",
                                   fontsize=8, rotation=45)
            ax_bar.set_ylim(0, 1.05)
            ax_bar.set_xlim(-0.6, T - 0.4)
            heatmap(ax_low, cache["C"][i][None, :],
                    title=f"C[{i}] = weighted mix of value vectors  "
                          f"(Σ weight[j] × V[j])")
            caption(fig, f"The value vectors are averaged using those weights: position {i} "
                         "walks away with a blend of the tokens it attended to.")

    return save_animation(fig, draw, len(frames),
                          f"{cfg.OUTPUT_DIR}/videos/3_attention.mp4",
                          fps=cfg.VIDEO_FPS, dpi=cfg.VIDEO_DPI)
