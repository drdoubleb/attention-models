"""A tiny transformer language model written in plain NumPy.

Nothing is hidden behind a framework: the forward pass, every gradient,
and the optimizer are spelled out by hand so that any intermediate value
can be pulled out and animated.

Architecture (one transformer block, single attention head):

    ids                  token ids, length T
    x0 = E[ids] + P      token embedding + positional embedding   (T, D)
    Q, K, V = x0 Wq, x0 Wk, x0 Wv                                 (T, D)
    S = Q K^T / sqrt(D)  attention scores, causally masked        (T, T)
    A = softmax(S)       attention weights                        (T, T)
    C = A V              each row is a blend of value vectors     (T, D)
    x1 = x0 + C Wo       attended information added back          (T, D)
    x2 = x1 + relu(x1 W1) W2      feed-forward layer              (T, D)
    logits = x2 Wout     one score per vocabulary token           (T, V)

The model is trained to predict the *next* token at every position with
cross-entropy loss.  Layer normalization is deliberately left out: at
this scale training is stable without it and the math stays readable.
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # subtract max for stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class TinyTransformer:
    def __init__(self, vocab_size, d_model=16, context=16, mlp_hidden=32, seed=0):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context = context
        self.mlp_hidden = mlp_hidden

        rng = np.random.default_rng(seed)
        init = lambda *shape: rng.normal(0.0, 0.08, shape)

        self.params = {
            "E": init(vocab_size, d_model),      # token embeddings
            "P": init(context, d_model),         # positional embeddings
            "Wq": init(d_model, d_model),        # query projection
            "Wk": init(d_model, d_model),        # key projection
            "Wv": init(d_model, d_model),        # value projection
            "Wo": init(d_model, d_model),        # attention output projection
            "W1": init(d_model, mlp_hidden),     # feed-forward in
            "W2": init(mlp_hidden, d_model),     # feed-forward out
            "Wout": init(d_model, vocab_size),   # final projection to logits
        }

        # Adam optimizer state, one pair of moment tensors per parameter.
        self._adam_m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_t = 0

    # ------------------------------------------------------------- forward
    def forward(self, ids):
        """Run the model on a list/array of token ids (length T <= context).

        Returns a cache dict holding *every* intermediate value.  The
        animations read these directly; backward() consumes the same cache.
        """
        ids = np.asarray(ids)
        T = len(ids)
        assert T <= self.context, "sequence longer than the model's context"
        p = self.params
        scale = 1.0 / np.sqrt(self.d_model)

        x_tok = p["E"][ids]                       # (T, D) look up each token's vector
        x_pos = p["P"][:T]                        # (T, D) one vector per position
        x0 = x_tok + x_pos

        Q = x0 @ p["Wq"]
        K = x0 @ p["Wk"]
        V = x0 @ p["Wv"]

        S = (Q @ K.T) * scale                     # (T, T) how much i "matches" j
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        S_masked = np.where(mask, -1e9, S)        # future positions are hidden
        A = softmax(S_masked, axis=-1)            # rows sum to 1

        C = A @ V                                 # (T, D) weighted blend of values
        attn_out = C @ p["Wo"]
        x1 = x0 + attn_out                        # residual connection

        pre = x1 @ p["W1"]
        h = np.maximum(pre, 0.0)                  # ReLU
        mlp_out = h @ p["W2"]
        x2 = x1 + mlp_out                         # residual connection

        logits = x2 @ p["Wout"]
        probs = softmax(logits, axis=-1)

        return {
            "ids": ids, "T": T, "x_tok": x_tok, "x_pos": x_pos, "x0": x0,
            "Q": Q, "K": K, "V": V, "S": S, "S_masked": S_masked, "A": A,
            "C": C, "attn_out": attn_out, "x1": x1, "pre": pre, "h": h,
            "mlp_out": mlp_out, "x2": x2, "logits": logits, "probs": probs,
        }

    def loss(self, cache, targets):
        """Mean cross-entropy of predicting ``targets`` (next tokens)."""
        T = cache["T"]
        return float(-np.mean(np.log(cache["probs"][np.arange(T), targets] + 1e-12)))

    # ------------------------------------------------------------ backward
    def backward(self, cache, targets):
        """Hand-derived gradients of the loss w.r.t. every parameter."""
        p = self.params
        ids, T = cache["ids"], cache["T"]
        targets = np.asarray(targets)
        scale = 1.0 / np.sqrt(self.d_model)
        g = {}

        # Softmax + cross-entropy together give the famous (probs - onehot).
        dlogits = cache["probs"].copy()
        dlogits[np.arange(T), targets] -= 1.0
        dlogits /= T

        g["Wout"] = cache["x2"].T @ dlogits
        dx2 = dlogits @ p["Wout"].T

        # Feed-forward block (x2 = x1 + relu(x1 W1) W2).
        dmlp_out = dx2
        dx1 = dx2.copy()                          # residual branch
        g["W2"] = cache["h"].T @ dmlp_out
        dh = dmlp_out @ p["W2"].T
        dpre = dh * (cache["pre"] > 0)            # ReLU gate
        g["W1"] = cache["x1"].T @ dpre
        dx1 += dpre @ p["W1"].T

        # Attention block (x1 = x0 + (A V) Wo).
        dattn_out = dx1
        dx0 = dx1.copy()                          # residual branch
        g["Wo"] = cache["C"].T @ dattn_out
        dC = dattn_out @ p["Wo"].T

        dA = dC @ cache["V"].T
        dV = cache["A"].T @ dC

        # Softmax backward, row by row: dS = A * (dA - sum(dA * A)).
        A = cache["A"]
        dS = A * (dA - np.sum(dA * A, axis=-1, keepdims=True))

        dQ = (dS @ cache["K"]) * scale
        dK = (dS.T @ cache["Q"]) * scale

        x0 = cache["x0"]
        g["Wq"] = x0.T @ dQ
        g["Wk"] = x0.T @ dK
        g["Wv"] = x0.T @ dV
        dx0 += dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T

        # Embeddings: scatter-add each position's gradient onto its token row.
        g["E"] = np.zeros_like(p["E"])
        np.add.at(g["E"], ids, dx0)
        g["P"] = np.zeros_like(p["P"])
        g["P"][:T] = dx0

        return g

    # ----------------------------------------------------------- optimizer
    def adam_step(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """One Adam update: momentum + per-parameter adaptive step size."""
        self._adam_t += 1
        t = self._adam_t
        for k, grad in grads.items():
            m = self._adam_m[k] = beta1 * self._adam_m[k] + (1 - beta1) * grad
            v = self._adam_v[k] = beta2 * self._adam_v[k] + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ---------------------------------------------------------- generation
    def next_token_distribution(self, ids):
        """Probabilities for the token that follows ``ids``."""
        cache = self.forward(ids[-self.context:])
        return cache["probs"][-1], cache

    def generate(self, ids, num_tokens, temperature=1.0, rng=None):
        """Autoregressive sampling.  Returns (all_ids, per-step records)."""
        rng = rng or np.random.default_rng(0)
        ids = list(ids)
        steps = []
        for _ in range(num_tokens):
            probs, cache = self.next_token_distribution(np.array(ids))
            if temperature <= 0:
                next_id = int(np.argmax(probs))
                sample_probs = probs
            else:
                logits = np.log(probs + 1e-12) / temperature
                sample_probs = softmax(logits)
                next_id = int(rng.choice(len(sample_probs), p=sample_probs))
            steps.append({
                "context_ids": list(ids[-self.context:]),
                "probs": sample_probs.copy(),
                "attention": cache["A"].copy(),
                "chosen": next_id,
            })
            ids.append(next_id)
        return ids, steps
