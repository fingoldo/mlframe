"""Multi-scenario synthetic generators with KNOWN ground truth for FS benchmarking.

One parametric generator + named presets that stress different selector weaknesses, so a default flip
must win on the MAJORITY of scenarios x seeds (single-seed/single-scenario wins repeatedly mislead -
the selectors are high-variance). All classification (binary) so honest-holdout AUC is comparable across
LightGBM / Logistic / kNN downstreams.

Each preset returns (X: DataFrame, y: Series, truth: dict). truth keys:
  base                 - the causal latent feature names (recovery target)
  relevant             - base + redundant copies (an all-relevant method should keep these)
  noise                - pure-noise feature names
  interaction_operands - operands of multiplicative terms (marginal MI ~ 0; the synergy blindspot)
  quadratic_operands   - operands of squared terms
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _gen(seed, n, n_linear, interaction_pairs, n_quadratic, cluster_parents, cluster_size, cluster_noise_sd, n_noise, coef_scale, temperature, monotone):
    """Parametric builder. Latent z columns are the causal features; observed X = z + redundant copies + noise."""
    rng = np.random.default_rng(seed)
    n_base = n_linear + 2 * interaction_pairs + n_quadratic
    z = rng.standard_normal((n, n_base))
    logit = np.zeros(n)
    idx = 0
    lin_idx = list(range(idx, idx + n_linear)); idx += n_linear
    for k, j in enumerate(lin_idx):
        logit += coef_scale * (1.4 - 0.12 * k) * z[:, j]  # decreasing strengths -> stresses weak signals
    inter_idx = []
    for _ in range(interaction_pairs):
        a, b = idx, idx + 1; idx += 2
        logit += coef_scale * 1.6 * z[:, a] * z[:, b]
        inter_idx += [a, b]
    quad_idx = list(range(idx, idx + n_quadratic)); idx += n_quadratic
    for j in quad_idx:
        logit += coef_scale * 1.3 * (z[:, j] ** 2 - 1.0)

    p = 1.0 / (1.0 + np.exp(-logit / temperature))
    y = (rng.random(n) < p).astype(int)

    cols, base_names = {}, []
    for i in range(n_base):
        col = z[:, i]
        if monotone:  # MI-invariant monotone distortion: trees/MI unaffected, linear model sees a warped axis
            col = np.sign(col) * np.abs(col) ** 1.5
        name = f"inf_{i}"; cols[name] = col; base_names.append(name)

    redundant_names = []
    for parent in cluster_parents:
        for j in range(cluster_size):
            name = f"red_{parent}_{j}"
            cols[name] = z[:, parent] + cluster_noise_sd * rng.standard_normal(n)
            redundant_names.append(name)

    noise_names = []
    for i in range(n_noise):
        cols[name := f"noise_{i}"] = rng.standard_normal(n); noise_names.append(name)

    X = pd.DataFrame(cols)
    order = list(X.columns); rng.shuffle(order); X = X[order]
    truth = dict(
        base=base_names, relevant=base_names + redundant_names, noise=noise_names,
        interaction_operands=[f"inf_{i}" for i in inter_idx],
        quadratic_operands=[f"inf_{i}" for i in quad_idx],
    )
    return X, pd.Series(y, name="target"), truth


# Named presets. Each stresses a different selector failure mode.
def _base(seed, n=5000):  # 4 linear + 1 interaction + 1 quadratic + 3 clusters x4 + 32 noise (the original)
    return _gen(seed, n, 4, 1, 1, [0, 4, 6], 4, 0.30, 32, 1.0, 1.6, False)

def _xor2(seed, n=5000):       # interaction-heavy: 2 pairs + 1 quadratic + 2 linear -> synergy blindspot
    return _gen(seed, n, 2, 2, 1, [0], 4, 0.30, 28, 1.0, 1.6, False)

def _highnoise(seed, n=5000):  # 8 signals drowned in 100 noise -> noise-rejection stress
    return _gen(seed, n, 5, 1, 1, [0], 3, 0.30, 100, 1.0, 1.6, False)

def _manyredundant(seed, n=5000):  # big correlated clusters (8 copies x 4 parents) -> dedup stress
    return _gen(seed, n, 4, 1, 1, [0, 1, 4, 6], 8, 0.25, 24, 1.0, 1.6, False)

def _monotone(seed, n=5000):   # signals through monotone distortion -> linear-vs-tree divergence
    return _gen(seed, n, 5, 1, 1, [0, 4], 4, 0.30, 30, 1.0, 1.6, True)

def _weakmix(seed, n=5000):    # low SNR (small coef_scale, high temperature) -> threshold stress
    return _gen(seed, n, 6, 1, 1, [0], 4, 0.30, 30, 0.7, 2.2, False)


SCENARIOS = {
    "base": _base, "xor2": _xor2, "highnoise": _highnoise,
    "manyredundant": _manyredundant, "monotone": _monotone, "weakmix": _weakmix,
}


def make(scenario: str, seed: int = 0, n: int = 5000):
    return SCENARIOS[scenario](seed, n)


if __name__ == "__main__":
    for name in SCENARIOS:
        X, y, t = make(name, 0)
        print(f"{name:14s} shape={X.shape} pos={float(y.mean()):.3f} base={len(t['base'])} "
              f"inter={len(t['interaction_operands'])} quad={len(t['quadratic_operands'])} "
              f"relevant={len(t['relevant'])} noise={len(t['noise'])}")
