"""Harder synthetic where signal is SPLIT across regimes so no single selector wins (round-3 fix #2).

Blocks (each contributes ~comparable signal; designed so different selectors catch different ones):
  STRONG linear (4)        : large coefs -> every selector catches.
  FE-only (interaction+sq) : ia*ib and sq^2 -> only feature ENGINEERING recovers (MRMR-FE); pure selection blind.
  WEAK-SPARSE linear (8)   : small coefs buried in a LARGE noise pool -> RFECV/permutation backward-elimination
                             keeps them on collective CV gain, but greedy MRMR (marginal MI) drops the weakest and
                             a shallow Boruta shadow gate misses them. This is the block MRMR-FE alone loses.
  REDUNDANT copies         : noisy copies of strong[0] -> clustering/dedup territory.
  NOISE (200)              : large pool so the weak-sparse block is genuinely hard to find marginally.

Expected: mrmr_fe gets FE+strong+redundant but MISSES weak-sparse; rfecv gets weak-sparse+strong but MISSES FE;
the HYBRID (FE substrate + multi-gate recall) should get BOTH -> the only bed where compute-once-share-many can
beat a simple FE strategy. Per-block recovery is reported so we can see which selector misses what.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

N_STRONG = 4
N_WEAK = 8
N_REDUNDANT = 5          # noisy copies of strong[0]
N_NOISE = 200
WEAK_COEF = 0.40
CLUSTER_NOISE_SD = 0.30


def make_hard_dataset(n_samples: int = 6000, seed: int = 0):
    rng = np.random.default_rng(seed)
    strong = rng.standard_normal((n_samples, N_STRONG))
    ia, ib = rng.standard_normal(n_samples), rng.standard_normal(n_samples)   # interaction operands
    sq = rng.standard_normal(n_samples)                                       # quadratic operand
    weak = rng.standard_normal((n_samples, N_WEAK))

    logit = (1.3 * strong[:, 0] + 1.1 * strong[:, 1] - 1.0 * strong[:, 2] + 0.9 * strong[:, 3]
             + 1.5 * ia * ib                       # FE-only interaction
             + 1.2 * (sq ** 2 - 1.0)               # FE-only quadratic
             + WEAK_COEF * weak.sum(axis=1))        # 8 weak-sparse linear (collectively ~1.1, individually small)
    logit = logit / 1.7
    y = (rng.random(n_samples) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    cols, base = {}, []
    for i in range(N_STRONG):
        cols[f"str_{i}"] = strong[:, i]; base.append(f"str_{i}")
    cols["ia"] = ia; cols["ib"] = ib; cols["sq"] = sq
    base += ["ia", "ib", "sq"]
    for j in range(N_WEAK):
        cols[f"weak_{j}"] = weak[:, j]; base.append(f"weak_{j}")
    redundant = []
    for j in range(N_REDUNDANT):
        cols[f"red_0_{j}"] = strong[:, 0] + CLUSTER_NOISE_SD * rng.standard_normal(n_samples); redundant.append(f"red_0_{j}")
    noise = []
    for i in range(N_NOISE):
        cols[f"noise_{i}"] = rng.standard_normal(n_samples); noise.append(f"noise_{i}")

    X = pd.DataFrame(cols)
    order = list(X.columns); rng.shuffle(order); X = X[order]
    truth = {
        "base": base, "relevant": base + redundant, "noise": noise,
        "interaction_operands": ["ia", "ib"], "quadratic_operand": ["sq"],
        "strong": [f"str_{i}" for i in range(N_STRONG)],
        "weak_sparse": [f"weak_{j}" for j in range(N_WEAK)],
        "fe_block": ["ia", "ib", "sq"],
    }
    return X, pd.Series(y, name="target"), truth


if __name__ == "__main__":
    X, y, t = make_hard_dataset()
    print("shape", X.shape, "pos_rate", round(float(y.mean()), 3))
    print("n_base", len(t["base"]), "n_weak", len(t["weak_sparse"]), "n_noise", len(t["noise"]))
