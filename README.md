# Attention, by hand

Two ways into the same tiny attention-based language model, both built
**completely from scratch** — tokenizer, forward pass, backpropagation,
optimizer — with no ML frameworks:

* **`index.html` — "Look Back", an interactive tutorial.** One
  self-contained web page that trains a real transformer in your browser
  on a poem you pick, then has you take it apart piece by piece — by hand.
* **The Python pipeline** — the same model in plain NumPy, rendering a
  series of MP4 videos of its internals.

Every number shown in either is computed by code you can read in an
afternoon.

## The interactive tutorial (`index.html`)

Open `index.html` in any modern browser — no build step, no server, no
dependencies (fonts load from Google Fonts, with system fallbacks
offline). To publish it the way the NGS tutorial is published, enable
GitHub Pages for this repository (branch `main`, folder `/`); the page will
then be served at `drdoubleb.com/attention-models`.

You pick the training text first — the cat-and-dog story from the videos,
Christina Rossetti's *Who Has Seen the Wind?*, the opening of *A Tale of Two
Cities*, *The House That Jack Built*, Blake's *The Tyger*, or anything you
paste — and every number on every later step comes from that text. A
word-level transformer (d = 16, one block, one head, ~3–5k parameters)
trains in the background in about ten seconds; the story is one continuous
game played on one 16-token window of your text:

| # | Step | What you do |
|---|------|-------------|
| 1 | Cover & guess | Play next-token prediction yourself; lose at the slot where only *looking back* could have told you the answer |
| 2 | Chop | Cut punctuation off word tiles with the scissors; bet which token gets id 0; count the vocabulary |
| 3 | Look up | Fetch each slot's 16-number strip from the table E; notice two identical rows; drag position strips P onto them |
| 4 | Pour | Pour 100 % of "your" attention over the earlier slots — your row becomes a sealed bet |
| 5 | Ask & offer | Drag the query onto keys; watch the 16 products sum to a score; wrong targets (Q on Q, Q on V, the raw strip) explain themselves; then find the top key on the trained model |
| 6 | The card | On the full 16×16 score matrix, confirm where each row's future begins — the causal mask, and why it exists |
| 7 | Share out | Softmax as a fixed budget: push one slot above 0.5, raise another *without touching its slider*, guess the trained top share, snap back |
| 8 | Blend | Pour values into the bowl in the amounts the weights allow, ·Wo, add back to the slot (the residual) |
| 9 | Bet | Two taps of feed-forward polish, then bet on the trained model's top pick; see the newborn's flat odds and the loss |
| 10 | Train | Move three of the model's numbers by hand (with the real gradient printed), then hold to train — live loss curve, live attention map, your bets judged |
| 11 | Read its mind | Predict-then-reveal rows of the learned attention map, your poured row beside the machine's |
| 12 | Let it play | A proportional wheel spins on the model's odds; temperature 0 / 0.5 / 1 / 2; tap any generated token to see where it looked |
| ★ | Experiments | Two contexts with the same last words and different answers; the unmasked "cheater" twin caught by its attention stripe; which words drifted together |

Every drawn number can be tapped for its receipt (the formula with the real
values that produced it); every sentence about the trained model is
generated from the live numbers, never written in advance. Progress and
the trained weights are kept in `localStorage`. Works with mouse, touch or
keyboard (click a chip, then click its destination), respects
`prefers-reduced-motion`, and follows the light/dark theme.

`node tests/test_web_model.js` extracts the model script from `index.html`
and checks it the way `tests/test_gradients.py` checks the NumPy model:
finite-difference gradients, tokenizer round trip, deterministic training.

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
index.html       the interactive tutorial: model + tutorial in one file (vanilla HTML/CSS/JS)
attention_model/
  tokenizer.py   mini-BPE: learn merges, encode/decode, record merge history
  model.py       the transformer: forward pass, hand-derived backprop, Adam
  training.py    training loop + snapshot recording for the animations
animations/
  common.py           shared drawing helpers (token boxes, heatmaps, video writer)
  anim_tokenization.py ... anim_generation.py   one script per video
tests/
  test_gradients.py    numerical gradient check for every parameter (NumPy)
  test_web_model.js    the same check for the JavaScript model inside index.html
main.py          command-line entry point
config.py        all user-facing knobs
```

The JavaScript model in `index.html` is a line-by-line port of
`attention_model/model.py` (same architecture, same −10⁹ mask, same Adam),
with a word-level tokenizer instead of BPE so that attention patterns read
as words looking at words.

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
