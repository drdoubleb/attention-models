"""Video 5 — Generation.

The trained model writes text one token at a time.  Each step gets two
frames: first the model's probability distribution for the next token
(with the attention pattern that produced it), then the sampled token
being appended to the text.  The context window slides once the text
outgrows it.
"""

import numpy as np
import matplotlib.pyplot as plt

from .common import (INK, MUTED, ACCENT, blank, caption, draw_token_row,
                     heatmap, save_animation, HEAT_CMAP)

TOP_K = 10  # how many candidate tokens to show in the probability bars


def render(run, cfg):
    tokenizer = run["tokenizer"]
    prompt_len = len(run["prompt_ids"])
    steps = run["generation_steps"]

    frames = [("intro", None, None)]
    for s, step in enumerate(steps):
        frames.append(("predict", s, step))
        frames.append(("append", s, step))
    frames += [("final", None, None)] * 3

    fig = plt.figure(figsize=(12.8, 7.2))
    ax_text = fig.add_axes([0.02, 0.60, 0.96, 0.34])
    ax_A = fig.add_axes([0.08, 0.20, 0.32, 0.34])
    ax_bar = fig.add_axes([0.52, 0.16, 0.44, 0.34])

    def ids_so_far(s, appended):
        n = prompt_len + s + (1 if appended else 0)
        return run["generated_ids"][:n]

    def draw_text_panel(ids, context_ids, new_pos=None):
        toks = [tokenizer.vocab[i] for i in ids]
        n = len(ids)
        ctx_start = n - len(context_ids)
        dim = set(range(0, max(0, ctx_start)))          # fell out of the window
        hi = {new_pos} if new_pos is not None else set()
        draw_token_row(ax_text, toks, ids=ids, y=0.78, char_w=0.0105,
                       fontsize=10, height=0.22, line_gap=0.34,
                       highlight=hi, dim=dim)

    def draw(t):
        fig.texts.clear()
        for a in (ax_text, ax_A, ax_bar):
            a.clear()
        blank(ax_text)
        fig.suptitle("Generation: the model writes, one token at a time",
                     fontsize=15, color=INK, y=0.985)
        kind, s, step = frames[t]

        if kind == "intro":
            draw_text_panel(ids_so_far(0, False), run["prompt_ids"])
            blank(ax_A), blank(ax_bar)
            caption(fig, f'We hand the trained model a prompt: "{tokenizer.decode(run["prompt_ids"])}" '
                         "— and ask it to continue.")
            return
        if kind == "final":
            draw_text_panel(run["generated_ids"], run["generated_ids"])
            blank(ax_A), blank(ax_bar)
            caption(fig, f'Final output: "{tokenizer.decode(run["generated_ids"])}"')
            return

        ctx = step["context_ids"]
        ctx_toks = [tokenizer.vocab[i] for i in ctx]

        # Attention used for this prediction (last row = the predicting position).
        heatmap(ax_A, step["attention"], title="attention inside the context window",
                cmap=HEAT_CMAP, vmin=0, vmax=1,
                xticklabels=ctx_toks, yticklabels=ctx_toks, fontsize=7)
        last = len(ctx) - 1
        ax_A.add_patch(plt.Rectangle((-0.5, last - 0.5), len(ctx), 1, fill=False,
                                     edgecolor=ACCENT, lw=2, zorder=4))

        # Top-k candidates.
        probs = step["probs"]
        top = np.argsort(probs)[::-1][:TOP_K]
        labels = [tokenizer.vocab[i].replace(" ", "·").replace(chr(10), "\\n") for i in top]
        chosen = step["chosen"]
        colors = ["#7f9fc4"] * len(top)
        if kind == "append":
            colors = [ACCENT if i == chosen else "#c9d4e4" for i in top]
        ax_bar.bar(range(len(top)), probs[top], color=colors)
        ax_bar.set_xticks(range(len(top)))
        ax_bar.set_xticklabels(labels, family="monospace", fontsize=9, rotation=45)
        ax_bar.set_ylim(0, 1.0)
        ax_bar.set_title("probability of each candidate next token", fontsize=10)
        ax_bar.tick_params(labelsize=8)

        if kind == "predict":
            draw_text_panel(ids_so_far(s, False), ctx)
            caption(fig, f"Step {s + 1}: the model runs on its context (highlighted row = "
                         "the position doing the predicting) and scores every possible next token.")
        else:
            ids = ids_so_far(s, True)
            draw_text_panel(ids, ctx, new_pos=len(ids) - 1)
            tok = tokenizer.vocab[chosen].replace(" ", "·").replace(chr(10), "\\n")
            caption(fig, f"A token is sampled from those probabilities — '{tok}' — appended "
                         "to the text, and the whole process repeats.")

    return save_animation(fig, draw, len(frames),
                          f"{cfg.OUTPUT_DIR}/videos/5_generation.mp4",
                          fps=cfg.VIDEO_FPS, dpi=cfg.VIDEO_DPI)
