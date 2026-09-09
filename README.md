# Attention, by hand

Two ways into the same tiny attention-based language model, both built
**completely from scratch** — tokenizer, forward pass, backpropagation,
optimizer — with no ML frameworks:

* **`index.html` — "Look Back", an interactive tutorial.** Start with
  14 guided lessons explaining attention and LLMs from first principles.
  Open the optional model lab to train a real transformer in your browser
  and take it apart piece by piece.
* **The Python pipeline** — the same model in plain NumPy, rendering a
  series of MP4 videos of its internals.

The model and its arithmetic are implemented in readable JavaScript and
NumPy. Guided examples explicitly label hand-chosen values and scripted
scenarios; these must not be mistaken for measurements of the trained model.

## The interactive tutorial (`index.html`)

Open `index.html` in any modern browser — no build step, no server, no
dependencies (fonts load from Google Fonts, with system fallbacks
offline). To publish it the way the NGS tutorial is published, enable
GitHub Pages for this repository (branch `main`, folder `/`); the page will
then be served at `drdoubleb.com/attention-models`.

### Start with the guided lessons

The default page explains each concept before its activity. No coding or
calculus is required. Every lesson includes an experiment, a takeaway,
a checkpoint with feedback for every answer, optional deeper math, and
links to primary sources. Lessons and checkpoints never lock navigation.

| Part | Lessons |
|------|---------|
| Get your bearings | Next-token prediction; tokenization and subwords; embeddings and position |
| Understand attention | Queries/keys/values; scores, causal masking, softmax and weighted values; multi-head attention, residuals, MLPs and normalization; encoder, decoder and cross-attention architectures |
| Learn and generate | Targets, cross-entropy, backpropagation and optimization; held-out evaluation and overfitting; temperature, top-k/top-p, context windows and KV caches |
| From a model to an assistant | Scaling, RoPE, FlashAttention, GQA, MoE and quantization; pretraining, SFT, RLHF, DPO and LoRA; prompts, RAG and tools; hallucinations, interpretability, multimodality and evaluation |

The guide uses a short cat sentence as a recurring example, plus two optional
connections to the NGS tutorial that explain where the analogy stops.
Reading location and successful checkpoints are saved locally, and the guide
still works when browser storage is unavailable. Link directly to a lesson
with a hash such as `index.html#learn/attention` or `index.html#learn/assistant`.
The model does not begin training until a lab is opened.

### Explore the original model lab

Choose a training text — the cat-and-dog story from the videos,
Christina Rossetti's *Who Has Seen the Wind?*, the opening of *A Tale of Two
Cities*, *The House That Jack Built*, Blake's *The Tyger*, or anything you
paste — and inspect computations from that text. A
word-level transformer (d = 16, one block, one head, ~3–5k parameters)
trains in the background (speed depends on the device); the story is one continuous
game played on one 16-token window of your text:

| # | Step | What you do |
|---|------|-------------|
| 1 | Cover & guess | Play next-token prediction yourself; lose at the slot where only *looking back* could have told you the answer |
| 2 | Tokenize | Cut punctuation off word tiles with the scissors; bet which token gets id 0; count the vocabulary |
| 3 | Embeddings | Fetch each slot's 16-number strip from the table E; notice two identical rows; drag position strips P onto them |
| 4 | Choose context | Pour 100 % of "your" attention over the available slots — your row becomes a sealed bet |
| 5 | Queries & keys | Drag the query onto keys; watch the 16 products sum to a score; then find the top key on the trained model |
| 6 | Causal mask | On the full 16×16 score matrix, confirm where each row's future begins |
| 7 | Softmax weights | Explore how changing one score changes the whole weight distribution |
| 8 | Value mixture | Mix values in the amounts the weights allow, project with Wo, then add the residual |
| 9 | Next-token scores | Apply the feed-forward network and output projection; inspect output probabilities and loss |
| 10 | Train parameters | Move parameters with real gradients; train and inspect the training-corpus loss |
| 11 | Inspect attention | Reveal input-dependent mixing weights without treating them as a complete explanation |
| 12 | Generate tokens | Sample on the model's probabilities at different temperatures |
| ★ | Experiments | Two contexts with the same last words and different answers; the unmasked "cheater" twin caught by its attention stripe; which words drifted together |

The original numerical receipts and live computations remain available.
Each lab step now identifies its purpose and links back to a relevant
guided lesson. All experiments are accessible after selecting text; use
“Continue without finishing” to move past a puzzle without marking it complete.
Guided links use the cat story by default if no text has been selected.
Lab deep links use a hash such as `index.html#lab/5`.

The trained weights are kept in `localStorage`. Works with mouse, touch or
keyboard (click a chip, then click its destination), respects
`prefers-reduced-motion`, and follows the light/dark theme.

`node tests/test_web_model.js` extracts the model script from `index.html`
and checks it the way `tests/test_gradients.py` checks the NumPy model:
finite-difference gradients, tokenizer round trip, deterministic training.

`node tests/test_guide.js` checks all inline scripts, curriculum completeness,
lab mappings, masked weighted mixtures, softmax stability, temperature,
top-p filtering, context truncation, loss, and recovery from unavailable storage.
Both JavaScript test commands require only Node.js.

See [the review and coverage notes](docs/tutorial-review.md) for the teaching
problems addressed, scope, and validation limits. The NumPy model and video
pipeline are unchanged. The page remains one file with no build step.

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
