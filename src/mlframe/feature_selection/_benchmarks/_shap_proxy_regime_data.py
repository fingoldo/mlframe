"""Parametric synthetic-data generator for mapping WHERE ShapProxiedFS shines (Batch C foundation).

Builds the target from explicitly separable components so each regime axis is swept independently:
informative count, redundancy (correlated copies) and its strength, interaction order/strength,
SNR, class balance, nonlinearity. Ground-truth feature roles are returned so a benchmark can compute
regret-against-oracle (informatives + one representative per redundancy cluster), not just relative
metrics. Used by the where-it-shines sweep and the self-improvement loop's benchmarks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_regime_dataset(
    *, n_samples=4000, n_informative=5, n_redundant=0, redundancy_rho=0.8, n_noise=5,
    interaction_order=0, interaction_strength=0.0, interaction_has_main=True, snr=4.0,
    task="binary", class_balance=0.5, nonlinearity="linear", seed=0,
):
    """Return ``(X_df, y, roles)`` where roles maps each column name to
    'informative' | 'redundant' | 'noise'. See module docstring for the swept axes."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_samples, n_informative))

    def g(col):
        if nonlinearity == "monotone_nonlinear":
            return np.sign(col) * np.abs(col) ** 1.5
        if nonlinearity == "nonmonotone":
            return np.sin(2.0 * col)
        return col

    coefs = np.linspace(1.0, 0.4, n_informative)
    main = (g(Z) * coefs).sum(axis=1)

    inter = np.zeros(n_samples)
    interacting = []
    if interaction_order and interaction_strength > 0 and n_informative >= 2:
        if interaction_order == "xor":
            p, q = 0, 1
            inter = ((Z[:, p] > 0) ^ (Z[:, q] > 0)).astype(float) * 2 - 1
            interacting = [p, q]
        else:
            order = min(int(interaction_order), n_informative)
            interacting = list(range(order))
            inter = np.prod(Z[:, interacting], axis=1)
    f_signal = (1.0 - interaction_strength) * main + interaction_strength * inter
    if interacting and not interaction_has_main:  # pure interaction: zero the partners' main effect
        adj = (g(Z[:, interacting]) * coefs[interacting]).sum(axis=1)
        f_signal = f_signal - (1.0 - interaction_strength) * adj

    # Redundant correlated copies of random informatives.
    R_cols, R_names_src = [], []
    for j in range(n_redundant):
        src = int(rng.integers(0, n_informative))
        copy = redundancy_rho * Z[:, src] + np.sqrt(max(1e-9, 1 - redundancy_rho ** 2)) * rng.standard_normal(n_samples)
        R_cols.append(copy)
        R_names_src.append(src)
    R = np.column_stack(R_cols) if R_cols else np.empty((n_samples, 0))
    N = rng.standard_normal((n_samples, n_noise))

    X = np.column_stack([Z, R, N]) if (R.size or n_noise) else Z
    names = ([f"inf{i}" for i in range(n_informative)]
             + [f"red{i}_of{R_names_src[i]}" for i in range(n_redundant)]
             + [f"noise{i}" for i in range(n_noise)])
    roles = {**{f"inf{i}": "informative" for i in range(n_informative)},
             **{f"red{i}_of{R_names_src[i]}": "redundant" for i in range(n_redundant)},
             **{f"noise{i}": "noise" for i in range(n_noise)}}

    if np.std(f_signal) > 0:
        f_signal = f_signal / np.std(f_signal)
    if task == "regression":
        y = f_signal + rng.normal(0, np.sqrt(1.0 / max(snr, 1e-6)), n_samples)
    else:
        logits = f_signal * np.sqrt(snr)
        thr = np.quantile(logits, 1.0 - class_balance)
        p = 1.0 / (1.0 + np.exp(-(logits - thr)))
        y = (rng.random(n_samples) < p).astype(int)

    return pd.DataFrame(X, columns=names), y, roles


def oracle_subset(roles: dict) -> list[str]:
    """The honest-best attainable subset: all informatives + one representative per redundancy source.
    (For benchmark regret-against-oracle.)"""
    return [name for name, role in roles.items() if role == "informative"]
