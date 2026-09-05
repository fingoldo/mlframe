"""Regression: the paired-bootstrap p-value must never be exactly 0.0.

``compare_models``' bootstrap tail used to be a raw fraction ``2 * mean(boot_means <= 0)``, which returns exactly
0.0 whenever every resample lands on one side of zero -- the COMMON case for a clearly-better challenger. A p of
0.0 fed into Benjamini-Hochberg (or any multiple-testing correction) is treated as maximal significance with zero
uncertainty, when what actually happened is that the bootstrap ran out of resolution at ``n_boot``. The sibling
``discovery._eval_stats.bootstrap_gain_p_value`` already applied Davison-Hinkley add-one smoothing for exactly
this reason; this pins that the two agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.compare import compare_models


class _ConstPredictor:
    """Predicts a fixed constant for every row -- lets the loss difference be set exactly."""

    def __init__(self, value: float) -> None:
        """Stores the constant this predictor emits."""
        self.value = value

    def predict(self, X):
        """Returns ``value`` repeated once per row of ``X``."""
        return np.full(np.asarray(X).shape[0], self.value, dtype=np.float64)


@pytest.mark.parametrize("n_boot", [100, 1000])
def test_bootstrap_p_value_is_strictly_positive_when_every_resample_is_one_sided(n_boot: int) -> None:
    """A challenger that beats the champion on EVERY row saturates the bootstrap tail; the reported p-value must
    still be strictly positive and sit at the ``2 / (n_boot + 1)`` resolution floor, not at 0.0."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=400)
    # champion is far off, challenger is exact on the mean -> every resampled mean difference is > 0.
    res = compare_models(_ConstPredictor(50.0), _ConstPredictor(float(y.mean())), np.zeros((400, 1)), y, metric="mse", n_boot=n_boot)

    assert res["p_value"] > 0.0, "a saturated bootstrap tail must not report an impossible p = 0.0"
    assert res["p_value"] == pytest.approx(2.0 / (n_boot + 1.0)), "the floor must be the add-one smoothed two-sided tail"
    assert res["challenger_wins"] is True


def test_bootstrap_p_value_floor_is_reachable_by_should_promote() -> None:
    """At the default ``n_boot=1000`` the floor (~0.002) stays well under ``alpha=0.05``, so smoothing does not
    change any promotion decision at default settings."""
    from mlframe.training.composite.compare import should_promote

    rng = np.random.default_rng(1)
    y = rng.normal(size=400)
    res = should_promote(_ConstPredictor(50.0), _ConstPredictor(float(y.mean())), np.zeros((400, 1)), y, metric="mse")
    assert res["p_value"] < 0.05
    assert res["promote"] is True
