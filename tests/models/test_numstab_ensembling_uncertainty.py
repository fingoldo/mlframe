"""Numerical-stability regression test for mean-normalised ensemble uncertainty.

Pre-fix, with ``normalize_stds_by_mean_preds=True`` the per-class relative spread was
``std / mean`` with no guard; a class whose cross-member mean prediction is ~0 produced
inf/nan that poisoned the quantile threshold and corrupted the confident-index selection
for EVERY row. The guard treats a near-zero-mean class as zero relative spread.
"""

from __future__ import annotations

import warnings

import numpy as np

from mlframe.models.ensembling.predict import ensemble_probabilistic_predictions


def test_uncertainty_near_zero_mean_class_does_not_poison_selection():
    """Uncertainty near zero mean class does not poison selection."""
    rng = np.random.default_rng(0)
    n, k = 50, 3
    # Three members. Column 2 has ~0 mean across members for every row
    # (members straddle zero), so mean_preds[:, 2] ~ 0 -> std/mean blows up pre-fix.
    p1 = rng.uniform(0.2, 0.8, size=(n, k)).astype(np.float64)
    p2 = p1 + rng.normal(0, 0.02, size=(n, k))
    p3 = p1 + rng.normal(0, 0.02, size=(n, k))
    for p in (p1, p2, p3):
        p[:, 2] = rng.choice([-1e-9, 1e-9], size=n)  # near-zero mean column

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ens, uncertainty, confident_indices = ensemble_probabilistic_predictions(
            p1,
            p2,
            p3,
            ensemble_method="arithm",
            uncertainty_quantile=0.5,
            normalize_stds_by_mean_preds=True,
            verbose=False,
        )

    assert uncertainty is not None
    assert np.all(np.isfinite(uncertainty)), "near-zero-mean class poisoned uncertainty with inf/nan"
    # The threshold/selection must be well-defined: ~half the rows selected at q=0.5.
    assert confident_indices is not None
    assert 0 < len(confident_indices) <= n
