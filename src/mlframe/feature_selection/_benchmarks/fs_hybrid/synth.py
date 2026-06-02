"""Controlled synthetic generator for feature-selection hybrid experiments.

Ground-truth structure (so recovery is measurable):
  - 8 base "causal" latent features z0..z7, iid N(0,1).
  - Signal = linear(z0..z3, z7) + multiplicative interaction (z4*z5)
            + quadratic (z6^2 - 1). Binary target via Bernoulli(sigmoid(.)).
  - Redundant clusters: noisy copies of z0 (linear), z4 (interaction operand),
    z6 (quadratic operand) -> correlated groups that test dedup/clustering.
  - Pure noise features.

Why this shape: a tree model captures z4*z5 and z6^2 natively from raw splits;
a linear model needs engineered product/square features -> exposes per-model
divergence and the value of MRMR feature-engineering. Redundant clusters let us
see whether a selector keeps correlated copies (hurts linear/kNN) or collapses them.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

N_BASE = 8
CLUSTER_PARENTS = {0: 4, 4: 4, 6: 4}  # parent base idx -> n noisy copies
N_NOISE = 32
CLUSTER_NOISE_SD = 0.30


def make_dataset(n_samples: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_samples, N_BASE))

    logit = (
        1.4 * z[:, 0] + 1.1 * z[:, 1] - 1.0 * z[:, 2] + 0.9 * z[:, 3]
        + 1.6 * z[:, 4] * z[:, 5]          # multiplicative interaction
        + 1.3 * (z[:, 6] ** 2 - 1.0)       # quadratic
        + 0.8 * z[:, 7]
    )
    logit = logit / 1.6                     # temperature -> non-degenerate classes
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_samples) < p).astype(int)

    cols = {}
    base_names = []
    for i in range(N_BASE):
        name = f"inf_{i}"
        cols[name] = z[:, i]
        base_names.append(name)

    redundant_names = []
    for parent, k in CLUSTER_PARENTS.items():
        for j in range(k):
            name = f"red_{parent}_{j}"
            cols[name] = z[:, parent] + CLUSTER_NOISE_SD * rng.standard_normal(n_samples)
            redundant_names.append(name)

    noise_names = []
    for i in range(N_NOISE):
        name = f"noise_{i}"
        cols[name] = rng.standard_normal(n_samples)
        noise_names.append(name)

    X = pd.DataFrame(cols)
    # shuffle column order so position carries no information
    order = list(X.columns)
    rng.shuffle(order)
    X = X[order]

    truth = {
        "base": base_names,                 # truly causal
        "relevant": base_names + redundant_names,  # causal + redundant copies
        "noise": noise_names,
        "interaction_operands": ["inf_4", "inf_5"],
        "quadratic_operand": ["inf_6"],
    }
    return X, pd.Series(y, name="target"), truth


if __name__ == "__main__":
    X, y, truth = make_dataset()
    print("shape", X.shape, "pos_rate", round(float(y.mean()), 3))
    print("base", truth["base"])
    print("n_relevant", len(truth["relevant"]), "n_noise", len(truth["noise"]))
