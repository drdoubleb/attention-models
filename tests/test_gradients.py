"""Verify the hand-written backprop against numerical gradients.

For every parameter tensor we nudge a handful of entries by +/-eps and
compare the resulting change in loss to the analytic gradient.  Run with:

    python -m tests.test_gradients
"""

import numpy as np

from attention_model.model import TinyTransformer


def main():
    rng = np.random.default_rng(7)
    model = TinyTransformer(vocab_size=11, d_model=6, context=9, mlp_hidden=10, seed=3)
    ids = rng.integers(0, 11, size=8)
    targets = rng.integers(0, 11, size=8)

    cache = model.forward(ids)
    analytic = model.backward(cache, targets)

    eps = 1e-5
    worst = 0.0
    for name, param in model.params.items():
        flat = param.reshape(-1)
        # Spot-check up to 20 random entries per tensor.
        idx = rng.choice(flat.size, size=min(20, flat.size), replace=False)
        for i in idx:
            orig = flat[i]
            flat[i] = orig + eps
            loss_plus = model.loss(model.forward(ids), targets)
            flat[i] = orig - eps
            loss_minus = model.loss(model.forward(ids), targets)
            flat[i] = orig

            numeric = (loss_plus - loss_minus) / (2 * eps)
            exact = analytic[name].reshape(-1)[i]
            if abs(numeric - exact) < 1e-7:
                continue  # both effectively zero; relative error is meaningless
            denom = max(abs(numeric) + abs(exact), 1e-8)
            rel_err = abs(numeric - exact) / denom
            worst = max(worst, rel_err)
            assert rel_err < 1e-4, (
                f"gradient mismatch in {name}[{i}]: "
                f"numeric={numeric:.8f} analytic={exact:.8f} rel_err={rel_err:.2e}"
            )
        print(f"  {name:5s} ok")

    print(f"all gradients match (worst relative error {worst:.2e})")


if __name__ == "__main__":
    main()
