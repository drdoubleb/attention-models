# Tutorial review and learning-path changes

Reviewed `drdoubleb/attention-models` at commit
`0020e861fa83ff66aa16b8f7ece4b4688078d220`, with `drdoubleb/ngs-tutorial`
as the teaching reference.

The original has a valuable real, inspectable model and detailed arithmetic
receipts. Its difficulty for a novice is largely a sequencing problem: a
learner must understand the mechanism to understand why the manipulation matters.
The NGS tutorial introduces a physical problem and names the operation that
solves it before asking the learner to carry it out. Someone familiar with
sequencing can also supply missing context; an LLM beginner cannot assume that.

## Findings and changes

| Finding in the original | Resulting change |
|---|---|
| Picking a corpus and solving puzzles comes before an overview of a language model, transformer, or assistant. | A default guided entry establishes input, objective, representations, and output before the optional lab. |
| Labels such as “pour,” “polish,” and “the card” require remembering metaphors while learning new mechanisms. | Mechanism names appear in lab navigation; each step has a concise purpose and input/output explanation. The existing tactile experiments remain. |
| `renderRail` disables later steps based on `unlocked`, and `go` disables Next until an activity succeeds. | All guide lessons are open, lab scenes are accessible after text selection, and an explicit skip action does not record puzzle completion. |
| The final “What GPT adds” paragraph compresses most LLM material and incorrectly says none of the five lines changes. | Separate lessons explain architecture variants, normalization, scaling, post-training, retrieval, tools, and limitations, with precise boundaries around the implemented toy. |
| “Read its mind” suggests an attention map explains the complete prediction. | “Inspect attention” describes mixing weights and explains the limits of interpretability. |
| Falling loss on the training corpus can look like proof of general language ability. | A held-out evaluation lesson distinguishes optimization, memorization, generalization, contamination, and task evaluation. The lab explicitly labels its training-corpus loss. |
| “It stored no text” and “it can't say anything new” are too absolute. | Explain that parameters can encode memorized sequences and reusable patterns, and that novel token combinations do not demonstrate general ability. |

## Design of the beginner path

Four parts contain 14 lessons. Each explains terms before its controls, then
offers an activity, a takeaway, a checkpoint with answer-specific feedback,
optional detail, and primary-source links. Reading and answering are independent:
the user can revisit or skip any lesson. Numerical examples are either computed
with the real word tokenizer or clearly identified as hand-chosen illustrations.
Scripted retrieval and assistant scenarios do not impersonate a live LLM.

The original one-file architecture, light/dark theme, fonts, mouse/touch/keyboard
lab interactions, model implementation, storage keys, custom training texts,
numerical receipts, and NumPy/video pipeline are retained. Guide state uses
separate `lookback-guide-*` keys. Starting the guide does not train a model;
entering the lab restores the saved model or chooses the recommended cat story
when a deep link needs text. Changing training text resets experiment completion.

The NGS analogy is limited to following transformations and evaluating without
leakage. It explicitly avoids equating tokens with DNA bases or attention with
sequence alignment.

## Coverage and boundaries

| Area | Coverage |
|---|---|
| Representations | Tokens, vocabulary IDs, word/subword/byte tokenization, BPE, embeddings, vectors/matrices, position, contextual representations, RoPE |
| Attention | Q/K/V projections, dot products and scaling, causal masks including the current position, softmax, weighted values, multiple heads, self/cross-attention |
| Transformer | Residuals, position-wise MLPs, logits and output softmax, LayerNorm/RMSNorm, depth, encoder/decoder/encoder–decoder families |
| Training | Shifted targets, self-supervision, teacher forcing, cross-entropy, backpropagation, gradients, optimizer/Adam, learning rate, batch/step/epoch, parameters versus activations |
| Evaluation | Train/validation/test splits, held-out loss, overfitting, memorization, perplexity, data quality, contamination, task-focused assessment |
| Generation | Autoregression, temperature/greedy decoding, top-k/top-p, context budgets, truncation, stopping conditions, KV caches |
| Modern LLM engineering | Parameters versus training tokens, scaling, quadratic pair counts, prefill versus decode, FlashAttention, GQA, MoE, quantization |
| Assistants | Pretraining, SFT, RLHF, DPO, LoRA, in-context learning, prompts, RAG, chunking/retrieval, tools and agent loops |
| Limits and extensions | Hallucinations, factual support, attention interpretation, bias/robustness/privacy, prompt injection, multimodality, reasoning and inference-time compute |

This is a conceptual foundation with experiments, not an exhaustive ML course.
It does not implement a production tokenizer, a multi-layer LLM, distributed
training, preference optimization, retrieval infrastructure, or live tools.
Advanced topics are introduced with appropriate scope rather than simulated as
capabilities of the one-head browser model.

## Validation

- Existing `node tests/test_web_model.js` passes: 167 gradient comparisons
  (worst relative error 3.40e-6), tokenizer round trip, deterministic training,
  loss reduction, generation, and causal/unmasked behavior.
- New `node tests/test_guide.js` passes: all scripts parse, 14 lessons and 15
  lab mappings are complete, and the teaching calculations satisfy numerical
  invariants, including exact future exclusion and normalized top-p sampling.
- A temporary LinkeDOM harness loaded the shipped scripts and exercised all
  14 lesson renderers, controls at their limits, every quiz answer, empty and
  markup-like tokenizer input, progress storage, malformed/deep links, all
  15 original scene builders, and skip/return navigation. This is an emulated
  document test, not a real-browser layout or interaction audit.
- The local HTTP preview responds successfully. No rendered-browser visual,
  mobile, screen-reader, or end-to-end puzzle-completion audit was performed.
- The model core was preserved byte-for-byte. The Python pipeline was not
  changed or rerun.

The existing GitHub Pages setup is preserved. A proposed branch does not
update the public tutorial until its changes are merged/deployed by the owner.
