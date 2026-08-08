"""Video 4 — Training.

One frame per recorded snapshot.  Three synchronized panels show the
model learning: the loss falling, the attention pattern on the probe
sentence organizing itself, and the predicted next-token distribution
sharpening toward the correct answer.
"""

import numpy as np
import matplotlib.pyplot as plt

from .common import (INK, MUTED, ACCENT, caption, heatmap, save_animation,
                     HEAT_CMAP)


def render(run, cfg):
    tokenizer = run["tokenizer"]
    tokens = [tokenizer.vocab[i] for i in run["probe_ids"]]
    snapshots = run["snapshots"]
    losses = run["losses"]

    # What token actually follows the probe in the training text?
    from pathlib import Path
    text = Path(cfg.TRAINING_TEXT_PATH).read_text().strip()
    corpus_ids = tokenizer.encode(text)
    probe_ids = list(run["probe_ids"])
    if corpus_ids[: len(probe_ids)] == probe_ids:
        # The probe is a prefix of the corpus stream: the next token there.
        target_id = corpus_ids[len(probe_ids)]
    else:
        # Custom probe: walk the corpus by character count and take the
        # first token extending past the probe's end.
        pos, target_id = 0, corpus_ids[-1]
        boundary = len(run["probe_text"])
        for tid in corpus_ids:
            pos += len(tokenizer.vocab[tid])
            if pos > boundary:
                target_id = tid
                break

    # Track a fixed set of candidate tokens across all frames: the ones the
    # trained model ranks highest, plus the correct answer.
    final_probs = snapshots[-1]["next_probs"]
    top = list(np.argsort(final_probs)[::-1][:8])
    if target_id not in top:
        top[-1] = target_id
    bar_labels = [tokenizer.vocab[i].replace(" ", "·").replace(chr(10), "\\n") for i in top]

    fig = plt.figure(figsize=(12.8, 6.4))
    ax_loss = fig.add_axes([0.06, 0.18, 0.27, 0.62])
    ax_A = fig.add_axes([0.40, 0.24, 0.25, 0.56])
    ax_bar = fig.add_axes([0.72, 0.18, 0.25, 0.62])

    def draw(t):
        fig.texts.clear()
        for a in (ax_loss, ax_A, ax_bar):
            a.clear()
        snap = snapshots[t]
        step = snap["step"]
        fig.suptitle(f"Training the model — step {step} of {cfg.TRAIN_STEPS}",
                     fontsize=15, color=INK, y=0.97)

        # Loss curve so far.
        ax_loss.plot(range(1, len(losses) + 1), losses, color="#cccccc", lw=1)
        shown = losses[:step] if step else []
        if shown:
            ax_loss.plot(range(1, len(shown) + 1), shown, color="#1f77b4", lw=1.6)
            ax_loss.plot(len(shown), shown[-1], "o", color="#1f77b4", ms=6)
        ax_loss.set_title("loss: how wrong the next-token guesses are", fontsize=10)
        ax_loss.set_xlabel("training step", fontsize=9)
        ax_loss.set_xlim(0, len(losses) + 1)
        ax_loss.set_ylim(0, max(losses) * 1.05)
        ax_loss.tick_params(labelsize=8)

        # Attention on the probe sentence.
        heatmap(ax_A, snap["attention"], title="attention on the probe sentence",
                cmap=HEAT_CMAP, vmin=0, vmax=1,
                xticklabels=tokens, yticklabels=tokens, fontsize=7)

        # Next-token prediction for the probe.
        probs = snap["next_probs"][top]
        colors = [ACCENT if i == target_id else "#7f9fc4" for i in top]
        ax_bar.bar(range(len(top)), probs, color=colors)
        ax_bar.set_xticks(range(len(top)))
        ax_bar.set_xticklabels(bar_labels, family="monospace", fontsize=8,
                               rotation=40, ha="right")
        ax_bar.set_ylim(0, 1.0)
        ax_bar.set_title(f'what comes after "{run["probe_text"]}" ?', fontsize=10)
        ax_bar.tick_params(labelsize=8)

        caption(fig, "Every step: guess the next token everywhere, measure the error, "
                     "nudge all weights downhill (gradient descent). "
                     f"Red bar = the actual next token in the training text "
                     f"('{tokenizer.vocab[target_id].replace(' ', '·')}').")

    fps = getattr(cfg, "TRAINING_VIDEO_FPS", 8)
    return save_animation(fig, draw, len(snapshots),
                          f"{cfg.OUTPUT_DIR}/videos/4_training.mp4",
                          fps=fps, dpi=cfg.VIDEO_DPI)
