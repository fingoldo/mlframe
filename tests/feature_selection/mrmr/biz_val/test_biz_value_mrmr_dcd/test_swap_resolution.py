"""The DCD swap compares an aggregate with its anchor at the anchor's own resolution.

A plug-in MI grows with the number of bins, so an aggregate binned at the global ``quantization_nbins`` against an anchor binned by the adaptive default strategy (``quantile_spacing``, 2026-11-23) was judged at another
resolution. The swap on three PERFECT duplicates fired before that default for exactly this reason (the aggregate of identical copies carries nothing the anchor lacks), and the denoising it is built for was
never measured by it; with the aggregate binned as finely as the anchor, identical copies do not swap and noisy reflections of one latent do."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tests.feature_selection.conftest import LEGACY_FE_OFF


def _frame(noise: float, n: int = 1500, seed: int = 0):
    """Three reflections of one latent with the given noise, an unrelated strong column and a noise column."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "strong": other,
            "dup_a": latent + noise * rng.standard_normal(n),
            "dup_b": latent + noise * rng.standard_normal(n),
            "dup_c": latent + noise * rng.standard_normal(n),
            "noise_0": rng.standard_normal(n),
        }
    )
    return X, pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))


def _swaps(noise: float, tau: float, method: str = "mean_z") -> list:
    """The swap log of a DCD fit without feature engineering (isolates the swap)."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _frame(noise)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(
            **LEGACY_FE_OFF,
            dcd_enable=True,
            dcd_tau_cluster=tau,
            dcd_cluster_size_threshold=2,
            dcd_swap_method=method,
            full_npermutations=20,
            fe_max_steps=0,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
    return list((m.dcd_ or {}).get("swap_log", []))


def test_identical_copies_do_not_swap_for_an_aggregate():
    """Three reflections with noise 0.01 are the same column: no aggregate of them beats the anchor at equal resolution, so no aggregate swap fires."""
    assert [e for e in _swaps(0.01, 0.5) if e.get("branch") == "aggregate"] == []


def test_noisy_reflections_swap_for_their_denoised_aggregate():
    """Three reflections with noise 0.3 (correlation about 0.9) have a pooled component better than any one of them: the aggregate swap fires and records its combiner."""
    aggregates = [e for e in _swaps(0.3, 0.3) if e.get("branch") == "aggregate"]
    assert aggregates and all(e["aggregate_name"].startswith("_dcd_pc1_") and e.get("method") == "mean_z" for e in aggregates), aggregates
