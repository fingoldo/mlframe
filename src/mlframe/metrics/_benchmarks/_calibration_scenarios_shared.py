"""Shared synthetic calibration scenarios for the debiased-metric benches
(``bench_ece_debiased.py`` / ``bench_brier_decomp_debiased.py``): independently duplicated
across those scripts, consolidated here so a fix can't silently drift out of sync across copies.
Each returns ``(scores, accuracies, true_value_or_None)``.
"""
from __future__ import annotations

import numpy as np


def scn_calibrated_bimodal(rng, n):
    """Perfectly calibrated, bimodal scores. True ECE/REL = 0."""
    half = n // 2
    s = np.empty(n)
    s[:half] = np.clip(rng.beta(2.0, 12.0, half), 1e-6, 1 - 1e-6)
    s[half:] = np.clip(rng.beta(12.0, 2.0, n - half), 1e-6, 1 - 1e-6)
    return s, s.copy(), 0.0


def scn_miscal_sigmoid(rng, n):
    """Logit-shifted miscalibration; nonzero true ECE/REL (fine-grid reference)."""
    s = np.clip(rng.beta(1.2, 6.0, n), 1e-6, 1 - 1e-6)
    logit = np.log(s / (1 - s)) + 0.8
    acc = 1.0 / (1.0 + np.exp(-logit))
    return s, acc, None


def scn_miscal_overconf(rng, n):
    """Overconfident model; nonzero true ECE/REL."""
    s = np.clip(rng.uniform(0, 1, n), 1e-6, 1 - 1e-6)
    acc = np.clip(0.5 + 0.5 * (s - 0.5) * 0.4, 0, 1)  # squashed toward 0.5
    return s, acc, None


def ece_plugin(y, p, nbins):
    """Plug-in binned ECE via the prod headline kernel: ``sum_b (n_b/N) * |conf_b - acc_b|``."""
    from mlframe.metrics.calibration._calibration_metrics import compute_ece_and_brier_decomposition

    return compute_ece_and_brier_decomposition(np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64), nbins)[0]
