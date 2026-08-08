"""Training pipeline: tokenize the corpus, train the model, record history.

Besides updating weights, the loop periodically freezes a "snapshot" of
what the model currently believes about a fixed probe sentence — its
attention pattern and next-token predictions — so the training video can
show those beliefs sharpening over time.
"""

import pickle
from pathlib import Path

import numpy as np

from .model import TinyTransformer
from .tokenizer import MiniBPE


def make_batches(token_ids, context, batch_size, rng):
    """Sample ``batch_size`` random (input, target) windows from the stream.

    Targets are simply the inputs shifted one token to the left: at every
    position the model learns to predict what comes next.
    """
    max_start = len(token_ids) - context - 1
    starts = rng.integers(0, max_start + 1, size=batch_size)
    xs = np.stack([token_ids[s: s + context] for s in starts])
    ys = np.stack([token_ids[s + 1: s + context + 1] for s in starts])
    return xs, ys


def train(cfg):
    """Run the full pipeline described by the ``config`` module ``cfg``.

    Returns a dict with the tokenizer, model, and everything recorded for
    the animations, and pickles it to ``<OUTPUT_DIR>/run_data.pkl``.
    """
    rng = np.random.default_rng(cfg.SEED)
    text = Path(cfg.TRAINING_TEXT_PATH).read_text().strip()

    # ------------------------------------------------------------ tokenize
    print(f"training tokenizer ({cfg.NUM_MERGES} merges) ...")
    tokenizer = MiniBPE().train(text, cfg.NUM_MERGES)
    token_ids = np.array(tokenizer.encode(text))
    print(f"  vocab size {tokenizer.vocab_size}, corpus {len(token_ids)} tokens")

    if cfg.PROBE_TEXT:
        probe_ids = tokenizer.encode(cfg.PROBE_TEXT)[: cfg.CONTEXT]
    else:
        # Default probe: the corpus's own first sentence, using the corpus
        # tokenization so the probe is a true prefix of the training stream
        # (BPE merges can span sentence boundaries, so encoding the sentence
        # on its own could yield different tokens).
        first_sentence_len = len(text.split(".")[0]) + 1
        k, pos = 0, 0
        while (k < len(token_ids) - 1 and k < cfg.CONTEXT
               and pos < first_sentence_len):
            pos += len(tokenizer.vocab[token_ids[k]])
            k += 1
        probe_ids = [int(i) for i in token_ids[:k]]
    probe_text = tokenizer.decode(probe_ids)
    print(f"  probe sentence: {probe_text!r} ({len(probe_ids)} tokens)")

    # --------------------------------------------------------------- train
    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.D_MODEL,
        context=cfg.CONTEXT,
        mlp_hidden=cfg.MLP_HIDDEN,
        seed=cfg.SEED,
    )

    losses = []
    snapshots = []

    def take_snapshot(step):
        """Freeze the model's current view of the probe sentence."""
        cache = model.forward(probe_ids)
        snapshots.append({
            "step": step,
            "loss": losses[-1] if losses else None,
            "attention": cache["A"].copy(),
            # Distribution over the token following the full probe context.
            "next_probs": cache["probs"][-1].copy(),
        })

    print(f"training model for {cfg.TRAIN_STEPS} steps ...")
    take_snapshot(step=0)
    for step in range(1, cfg.TRAIN_STEPS + 1):
        xs, ys = make_batches(token_ids, cfg.CONTEXT, cfg.BATCH_SIZE, rng)

        # Accumulate loss and gradients over the batch, then average.
        batch_loss = 0.0
        grads = None
        for x, y in zip(xs, ys):
            cache = model.forward(x)
            batch_loss += model.loss(cache, y)
            g = model.backward(cache, y)
            if grads is None:
                grads = g
            else:
                for k in grads:
                    grads[k] += g[k]
        batch_loss /= len(xs)
        for k in grads:
            grads[k] /= len(xs)

        model.adam_step(grads, lr=cfg.LEARNING_RATE)
        losses.append(batch_loss)

        if step % cfg.SNAPSHOT_EVERY == 0 or step == cfg.TRAIN_STEPS:
            take_snapshot(step)
        if step % max(1, cfg.TRAIN_STEPS // 10) == 0:
            print(f"  step {step:5d}  loss {batch_loss:.4f}")

    # ------------------------------------------------------------ generate
    print(f"generating from prompt {cfg.PROMPT!r} ...")
    prompt_ids = tokenizer.encode(cfg.PROMPT)
    all_ids, gen_steps = model.generate(
        prompt_ids, cfg.GENERATE_TOKENS,
        temperature=cfg.TEMPERATURE, rng=np.random.default_rng(cfg.SEED),
    )
    print(f"  -> {tokenizer.decode(all_ids)!r}")

    run = {
        "tokenizer": tokenizer,
        "model": model,
        "probe_text": probe_text,
        "probe_ids": probe_ids,
        "probe_cache": model.forward(probe_ids),  # final trained internals
        "losses": losses,
        "snapshots": snapshots,
        "prompt_ids": prompt_ids,
        "generated_ids": all_ids,
        "generation_steps": gen_steps,
    }

    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_data.pkl", "wb") as f:
        pickle.dump(run, f)
    print(f"saved run data to {out_dir / 'run_data.pkl'}")
    return run


def load_run(cfg):
    with open(Path(cfg.OUTPUT_DIR) / "run_data.pkl", "rb") as f:
        return pickle.load(f)
