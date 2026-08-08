"""Video 1 — Tokenization.

Shows the probe sentence being split into characters, then replays the
tokenizer's learned BPE merges one at a time: on every frame the pair
being fused is highlighted, and the sidebar explains which merge rule is
firing and how often that pair appeared in the training text.
"""

import matplotlib.pyplot as plt

from .common import (INK, MUTED, ACCENT, blank, caption, draw_token_row,
                     save_animation)


def build_frames(run):
    tokenizer = run["tokenizer"]
    text = run["probe_text"]
    steps = tokenizer.encode_with_history(text)

    # Look up training stats for each merge so the sidebar can cite them.
    merge_info = {tuple(h["pair"]): h for h in tokenizer.merge_history}
    merge_rank = {tuple(h["pair"]): r + 1 for r, h in enumerate(tokenizer.merge_history)}

    frames = []
    frames.append({
        "tokens": list(text), "highlight": set(),
        "note": "",
        "cap": f'We start with the raw text: "{text}"',
    })
    frames.append({
        "tokens": steps[0]["tokens"], "highlight": set(),
        "note": f"{len(steps[0]['tokens'])} tokens",
        "cap": "Step 1: split the text into single characters — one token each. (· marks a space)",
    })

    for step in steps[1:]:
        pair = step["merged_pair"]
        info = merge_info[pair]
        rank = merge_rank[pair]
        # Highlight every occurrence of the freshly created token.
        new_tok = info["new_token"]
        hi = {i for i, t in enumerate(step["tokens"]) if t == new_tok}
        l, r = (p.replace(" ", "·").replace(chr(10), "\\n") for p in pair)
        frames.append({
            "tokens": step["tokens"], "highlight": hi,
            "note": f"merge #{rank}:  '{l}' + '{r}'  →  '{new_tok.replace(' ', '·')}'",
            "cap": f"Merge rule #{rank} fires: '{l}'+'{r}' were seen together "
                   f"{info['count']}× in the training text, so they fused into one token.",
        })

    final = steps[-1]["tokens"]
    ids = [tokenizer.token_to_id[t] for t in final]
    frames.append({
        "tokens": final, "highlight": set(), "ids": ids, "show_ids": True,
        "note": f"{len(final)} tokens",
        "cap": "Done. Every token gets an ID number — these IDs are what the model actually sees.",
    })
    frames.append(frames[-1])  # hold the final frame
    frames.append(frames[-1])
    return frames


def render(run, cfg):
    frames = build_frames(run)
    tokenizer = run["tokenizer"]

    fig = plt.figure(figsize=(12.8, 5.2))
    ax = fig.add_axes([0, 0.12, 1, 0.82])

    def draw(t):
        fig.texts.clear()
        ax.clear()
        blank(ax)
        f = frames[t]
        fig.suptitle("Tokenization: byte-pair encoding", fontsize=15,
                     color=INK, y=0.96)

        ids = f.get("ids")
        draw_token_row(ax, f["tokens"], ids=ids, y=0.62,
                       highlight=f["highlight"], show_ids=f.get("show_ids", False),
                       char_w=0.011, fontsize=11, line_gap=0.30)
        if f["note"]:
            ax.text(0.5, 0.16, f["note"], transform=ax.transAxes, ha="center",
                    fontsize=13, color=ACCENT if "merge" in f["note"] else MUTED,
                    family="monospace")
        caption(fig, f["cap"])

    return save_animation(fig, draw, len(frames),
                          f"{cfg.OUTPUT_DIR}/videos/1_tokenization.mp4",
                          fps=cfg.VIDEO_FPS, dpi=cfg.VIDEO_DPI)
