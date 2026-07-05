"""Parametric synthetic-data generator for mapping WHERE ShapProxiedFS shines (Batch C foundation).

Builds the target from explicitly separable components so each regime axis is swept independently:
informative count, redundancy (correlated copies) and its strength, interaction order/strength,
SNR, class balance, nonlinearity. Ground-truth feature roles are returned so a benchmark can compute
regret-against-oracle (informatives + one representative per redundancy cluster), not just relative
metrics. Used by the where-it-shines sweep and the self-improvement loop's benchmarks.

Caveat (finite-sample noise-pool effect, iter23 recall investigation 2026-05-28):
Informative coefficients are ``np.linspace(1.0, 0.4, n_informative)`` -- the weakest informatives
sit near coef ~0.4. With ``n_samples`` modest (~2000) and ``n_noise`` large (e.g. ~7000), random
spurious correlations in the noise pool can empirically match or exceed the weakest informatives'
signal-to-noise ratio in any given sample, even at SNR=5.0. Width=7000 / n_rows=2000 / seed sweep
shows NON-DETERMINISTIC drop of weaker informatives (e.g. seed=0 drops inf4+inf6, seed=1 drops
inf5, seed=2 drops inf1+inf5+inf6); the dropped set is not coef-monotone (seed=2 dropped inf1 with
coef 0.91 while inf7 with coef 0.4 survived). This is a synthetic-data finite-sample limitation,
NOT a ShapProxiedFS pipeline bias. To benchmark recall at width>=5000 honestly, raise ``n_samples``
(>=5000 recommended) or use ``snr>=8`` so the informatives stay separable from random noise-pool
correlations. Recall comparisons at fixed ``n_samples=2000`` across widths 3k -> 7k mix algorithm
behaviour with synthetic-data variance.
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

    # Pre-allocate the full feature matrix and fill role blocks in place.
    # The wide noise pool at large widths (e.g. C4: 10k rows x 19960 noise cols)
    # made the prior `[Z, R, N]` + `column_stack` path peak above ~3 GiB; that triggered
    # OOM on dev boxes sharing memory with Postgres / Memory Compression. Chunked noise fill
    # caps the transient overhead at one chunk's worth.
    #
    # float32 halves the buffer (10000 x 20000 x 4 = 800 MiB vs 1.5 GiB) and
    # cuts the downstream pd.DataFrame block-manager copy in half too; the
    # synthetic-data contract only cares about std()~=1 within 5pp tolerance,
    # which is well above float32 precision. (Pre-fix C4 peak was ~4.3 GiB
    # with float64; float32 + DataFrame(copy=False) drops it under the 2.5 GiB cap.)
    total_cols = n_informative + n_redundant + n_noise
    X = np.empty((n_samples, total_cols), dtype=np.float32)
    X[:, :n_informative] = Z.astype(np.float32, copy=False)

    R_names_src = []
    rho_keep = np.sqrt(max(1e-9, 1 - redundancy_rho**2))
    for j in range(n_redundant):
        src = int(rng.integers(0, n_informative))
        X[:, n_informative + j] = (redundancy_rho * Z[:, src] + rho_keep * rng.standard_normal(n_samples)).astype(np.float32, copy=False)
        R_names_src.append(src)

    noise_start = n_informative + n_redundant
    if n_noise:
        # Stream noise into X in column-block chunks to bound peak transient allocation.
        # Chunk size targets ~32 MiB per scratch buffer regardless of n_samples / n_noise.
        target_chunk_bytes = 32 * 1024 * 1024
        bytes_per_col = n_samples * 4  # float32
        chunk_cols = max(1, min(n_noise, target_chunk_bytes // max(bytes_per_col, 1)))
        col = 0
        while col < n_noise:
            this = min(chunk_cols, n_noise - col)
            X[:, noise_start + col : noise_start + col + this] = rng.standard_normal((n_samples, this)).astype(np.float32, copy=False)
            col += this

    names = [f"inf{i}" for i in range(n_informative)] + [f"red{i}_of{R_names_src[i]}" for i in range(n_redundant)] + [f"noise{i}" for i in range(n_noise)]
    roles = {
        **{f"inf{i}": "informative" for i in range(n_informative)},
        **{f"red{i}_of{R_names_src[i]}": "redundant" for i in range(n_redundant)},
        **{f"noise{i}": "noise" for i in range(n_noise)},
    }

    if np.std(f_signal) > 0:
        f_signal = f_signal / np.std(f_signal)
    if task == "regression":
        y = f_signal + rng.normal(0, np.sqrt(1.0 / max(snr, 1e-6)), n_samples)
    else:
        logits = f_signal * np.sqrt(snr)
        thr = np.quantile(logits, 1.0 - class_balance)
        p = 1.0 / (1.0 + np.exp(-(logits - thr)))
        y = (rng.random(n_samples) < p).astype(int)

    # ``pd.DataFrame(X, columns=...)`` in pandas 2.x copies the backing buffer by
    # default, which doubles peak RSS for the wide-noise C4 regime (10k x 20k =
    # 1.5 GiB; the copy pushes peak above 4 GiB and trips the OOM sensor).
    # ``copy=False`` keeps the DataFrame as a zero-copy view of ``X``; the
    # caller treats the return as read-only so the shared buffer is safe.
    return pd.DataFrame(X, columns=names, copy=False), y, roles


def oracle_subset(roles: dict) -> list[str]:
    """The honest-best attainable subset: all informatives + one representative per redundancy source.
    (For benchmark regret-against-oracle.)"""
    return [name for name, role in roles.items() if role == "informative"]
