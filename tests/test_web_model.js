#!/usr/bin/env node
/**
 * Test the JavaScript model that lives inside index.html (the interactive tutorial).
 *
 * The tutorial embeds a from-scratch transformer in <script id="model-core">.
 * This test extracts that script, evaluates it in a bare VM context (no DOM),
 * and checks it the same way tests/test_gradients.py checks the NumPy model:
 *
 *   1. hand-written backprop matches finite differences for every parameter
 *   2. the tokenizer round-trips a poem with punctuation and line breaks
 *   3. training is deterministic and actually lowers the loss
 *   4. the "cheating" (unmasked) variant really can see the future
 *
 * Run with:  node tests/test_web_model.js
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
const m = html.match(/<script id="model-core">([\s\S]*?)<\/script>/);
if (!m) { console.error("could not find <script id=\"model-core\"> in index.html"); process.exit(1); }

const ctx = { console, Math, JSON, Map, Set, Array, Object, Number, String, Error, Infinity, NaN };
vm.createContext(ctx);
vm.runInContext(m[1], ctx);
// class declarations are lexical bindings, not properties of the context, so pull them out explicitly
const { TinyTransformer, WordTokenizer, mulberry32, trainStep, corpusLoss } =
  vm.runInContext("({ TinyTransformer, WordTokenizer, mulberry32, trainStep, corpusLoss })", ctx);

let failures = 0;
function check(ok, msg) { console.log((ok ? "  ok    " : "  FAIL  ") + msg); if (!ok) failures++; }

/* ------------------------------------------------------------ 1. gradients */
console.log("1. gradient check (finite differences vs backward)");
{
  const rng = mulberry32(7);
  // a wider init and a few training steps make every gradient comfortably non-zero
  const model = new TinyTransformer({ vocabSize: 11, dModel: 6, context: 9, mlpHidden: 10, seed: 3, initStd: 0.25 });
  const ids = Array.from({ length: 8 }, () => Math.floor(rng() * 11));
  const targets = Array.from({ length: 8 }, () => Math.floor(rng() * 11));
  const cache = model.forward(ids);
  const analytic = model.backward(cache, targets);
  const eps = 1e-5;
  let worst = 0, compared = 0;
  for (const name of Object.keys(model.params)) {
    const P = model.params[name];
    const rows = P.length, cols = P[0].length;
    for (let n = 0; n < 20; n++) {
      const i = Math.floor(rng() * rows), j = Math.floor(rng() * cols);
      const orig = P[i][j];
      P[i][j] = orig + eps; const lp = model.loss(model.forward(ids), targets);
      P[i][j] = orig - eps; const lm = model.loss(model.forward(ids), targets);
      P[i][j] = orig;
      const numeric = (lp - lm) / (2 * eps), exact = analytic[name][i][j];
      if (Math.abs(numeric) + Math.abs(exact) < 1e-7) continue;
      const rel = Math.abs(numeric - exact) / Math.max(Math.abs(numeric) + Math.abs(exact), 1e-8);
      worst = Math.max(worst, rel); compared++;
      if (rel > 1e-4) { check(false, `${name}[${i}][${j}] numeric=${numeric.toExponential(4)} analytic=${exact.toExponential(4)} rel=${rel.toExponential(2)}`); }
    }
  }
  check(worst < 1e-4, `all parameters: ${compared} entries compared, worst relative error ${worst.toExponential(2)}`);
}

/* ------------------------------------------------------------ 2. tokenizer */
console.log("2. tokenizer round trip");
{
  const poem = "Who has seen the wind?\nNeither I nor you:\nBut when the leaves hang trembling,\nThe wind is passing through.\n\nWho has seen the wind?\nNeither you nor I:\nBut when the trees bow down their heads,\nThe wind is passing by.";
  const tok = new WordTokenizer(poem);
  const ids = tok.encode(poem);
  check(tok.decode(ids) === poem.toLowerCase(), `decode(encode(poem)) reproduces the poem (${ids.length} tokens, vocabulary ${tok.vocabSize})`);
  check(tok.vocab.includes("\n"), "the line break is a token");
  check(tok.counts.every((c, i) => i === 0 || c <= tok.counts[i - 1]), "ids are ordered by frequency");
  check(tok.encodeReport("the zebra").unknown.length === 1, "unknown words are reported, not silently mangled");
}

/* ------------------------------------------------------------- 3. training */
console.log("3. training: deterministic and learns");
{
  const text = fs.readFileSync(path.join(__dirname, "..", "data", "training_text.txt"), "utf8").trim();
  const tok = new WordTokenizer(text);
  const ids = tok.encode(text);
  function run(steps) {
    const model = new TinyTransformer({ vocabSize: tok.vocabSize, seed: 42 });
    const rng = mulberry32(1234);
    const losses = [];
    for (let s = 0; s < steps; s++) losses.push(trainStep(model, ids, { batchSize: 8, lr: 3e-3, rng }).loss);
    return { model, losses, corpus: corpusLoss(model, ids) };
  }
  const t0 = Date.now();
  const a = run(400), b = run(400);
  const ms = (Date.now() - t0) / 800;
  check(a.losses.every((l, i) => l === b.losses[i]), "same seeds → bit-identical loss curve");
  const start = corpusLoss(new TinyTransformer({ vocabSize: tok.vocabSize, seed: 42 }), ids);
  check(a.corpus < start * 0.5, `corpus loss ${start.toFixed(3)} → ${a.corpus.toFixed(3)} after 400 steps (${ms.toFixed(1)} ms per step)`);
  const gen = a.model.generate(tok.encode("the cat"), 10, { temperature: 0 });
  check(gen.ids.length === 12, `greedy generation runs: "${tok.decode(gen.ids)}"`);
}

/* ---------------------------------------------------------- 4. the cheater */
console.log("4. the unmasked twin");
{
  const model = new TinyTransformer({ vocabSize: 9, seed: 1 });
  const honest = model.forward([1, 2, 3, 4]).A;
  model.causal = false;
  const cheat = model.forward([1, 2, 3, 4]).A;
  check(honest[0][3] === 0 && cheat[0][3] > 0, "with causal=false a position can attend to its future");
  check(Math.abs(cheat[0].reduce((s, x) => s + x, 0) - 1) < 1e-12, "rows still sum to one");
}

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
