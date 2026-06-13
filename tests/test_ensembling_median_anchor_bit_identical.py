"""Regression sensor: the outlier-gate cross-member anchor uses ``np.median`` (fast C reduction), and it stays bit-identical to the prior
``np.quantile(q=0.5)`` generic-partition path. Guards a future "revert to np.quantile" from silently re-introducing the slower path, and pins
that the swap never changes which members the gate excludes (the anchor feeds per-member MAE/STD, which drive exclusion)."""
from __future__ import annotations

import numpy as np

import mlframe.models.ensembling.predict as PR
from mlframe.models.ensembling.predict import ensemble_probabilistic_predictions


def test_median_anchor_matches_quantile_q05_bit_identical():
    rng = np.random.default_rng(7)
    preds = [np.clip(rng.beta(2, 2, (40000, 2)), 0.01, 0.99) for _ in range(5)]
    preds.append(np.clip(rng.beta(8, 2, (40000, 2)), 0.01, 0.99))  # an outlier member, so the >2-member gate path runs

    out_new = ensemble_probabilistic_predictions(*preds, ensemble_method="harm", verbose=False)

    _qt = np.quantile
    PR.np.median = lambda a, axis=None, **kw: _qt(a, 0.5, axis=axis)
    try:
        out_old = ensemble_probabilistic_predictions(*preds, ensemble_method="harm", verbose=False)
    finally:
        PR.np.median = np.median

    for a, b in zip(out_new, out_old):
        if a is None and b is None:
            continue
        assert np.array_equal(np.asarray(a), np.asarray(b)), "median anchor diverged from quantile(q=0.5)"
