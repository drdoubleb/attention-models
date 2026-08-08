# Attention, animated

Code that builds a tiny attention-based language model **completely from
scratch** — tokenizer, forward pass, backpropagation, optimizer, all in
plain NumPy — and renders a series of MP4 videos explaining how it works
at a base level. No ML frameworks: every number shown on screen is
computed by code you can read in an afternoon.

## The video series

| # | Video | What it shows |
|---|-------|---------------|
| 1 | `1_tokenization.mp4` | Text splitting into characters, then BPE merge rules fusing frequent pairs into tokens with IDs |
| 2 | `2_embeddings.mp4` | Token IDs looking up embedding vectors, positional vectors being added, the input matrix X filling up |
| 3 | `3_attention.mp4` | The centerpiece: Q/K/V projections, dot-product scores, causal masking, softmax, and value blending — one query position at a time |
| 4 | `4_training.mp4` | Loss falling, the attention pattern organizing itself, and next-token predictions sharpening toward the right answer |
| 5 | `5_generation.mp4` | The trained model writing text token by token: context window, probabilities, sampling |

## Quick start

```bash
pip install -r requirements.txt   # numpy + matplotlib
# ffmpeg is needed for MP4 output (falls back to GIF without it)

python main.py all                # train + render every video into out/videos/
```

Other commands:

```bash
python main.py train                  # just train; saves out/run_data.pkl
python main.py animate                # re-render all videos from the saved run
python main.py animate attention      # re-render a single video
python -m tests.test_gradients        # verify backprop against numerical gradients
```

## Using your own text

Everything configurable lives in [`config.py`](config.py):

* **Training text** — edit `data/training_text.txt` or point
  `TRAINING_TEXT_PATH` at your own file. Small, repetitive text works
  best: the model is tiny on purpose so it can be fully visualized.
* **Prompt** — set `PROMPT` to the starter text for the generation video.
* **Model size, training length, video fps/resolution** — all in the same file.

## What's inside

```
attention_model/
  tokenizer.py   mini-BPE: learn merges, encode/decode, record merge history
  model.py       the transformer: forward pass, hand-derived backprop, Adam
  training.py    training loop + snapshot recording for the animations
animations/
  common.py           shared drawing helpers (token boxes, heatmaps, video writer)
  anim_tokenization.py ... anim_generation.py   one script per video
tests/
  test_gradients.py    numerical gradient check for every parameter
main.py          command-line entry point
config.py        all user-facing knobs
```

### The model

One transformer block, single attention head, written so the math is
front and center (see the docstring in `attention_model/model.py`):

```
x0 = E[ids] + P                        embeddings + positions
A  = softmax(mask(Q Kᵀ / √d))          attention weights   (Q,K,V = x0·Wq, x0·Wk, x0·Wv)
x1 = x0 + (A V) Wo                     attended info, residual
x2 = x1 + relu(x1 W1) W2               feed-forward, residual
logits = x2 Wout                       next-token scores
```

Layer normalization and multi-head attention are deliberately omitted to
keep the math readable; at this scale training is stable without them.
Every gradient is derived by hand and verified against finite differences
by `tests/test_gradients.py`.

### How the animations get their data

`model.forward()` returns a cache containing **every** intermediate value
(embeddings, Q, K, V, scores, attention weights, blended values, logits,
probabilities). The training loop snapshots that cache on a fixed probe
sentence every few steps, and generation records it at every sampled
token. The animation scripts are pure consumers of this recorded data —
matplotlib draws each frame and ffmpeg stitches them into MP4s.
