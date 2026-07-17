"""Regression: region-adaptive fit must NEVER store ``None`` region params.

When every candidate in a region scores ``-inf`` (degenerate region: e.g. all OOF scoring
folds fail), ``score > best_score`` is ``-inf > -inf == False``, so the pre-fix code left
``best_params`` at its ``None`` seed and stored it. At predict time the region's
``forward`` / ``inverse`` calls ``transform.forward(y, base, None)`` -> ``TypeError``.
The fix seeds ``best_params`` from a guaranteed full-region ``linear_residual`` fit.
"""

import numpy as np

from mlframe.training.composite.discovery import _region_adaptive as ra
from mlframe.training.composite.discovery._region_adaptive import fit_region_adaptive


def test_all_minus_inf_region_keeps_non_none_params_and_predicts(monkeypatch):
    # Force every candidate to score -inf in every region -> the failure condition.
    """All minus inf region keeps non none params and predicts."""
    monkeypatch.setattr(ra, "_oof_score_transform", lambda *a, **k: (-np.inf, None))

    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(size=n)
    y = 2.0 * base + rng.normal(scale=0.1, size=n)

    spec = fit_region_adaptive(y, base, k=4, n_folds=3, random_state=0)

    # No region may carry None params.
    assert all(p is not None for p in spec.region_params), spec.region_params

    # And predict round-trips without TypeError (the pre-fix crash).
    t = spec.forward(y, base)
    back = spec.inverse(t, base)
    assert np.all(np.isfinite(t))
    assert np.all(np.isfinite(back))
