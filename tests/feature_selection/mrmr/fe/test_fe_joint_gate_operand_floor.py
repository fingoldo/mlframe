"""The joint-gate operand floor rejects degraded copies of one operand but not pairs whose partner carries joint information."""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_operand_floor import _beats_the_larger_operand

_MARGINALS = {0: 0.50, 1: 0.05}


def _marginal(var: int) -> float:
    """Fixed per-operand marginal MI table."""
    return _MARGINALS[var]


def test_noise_degraded_copy_of_strong_operand_is_rejected():
    """Engineered MI and the raw joint MI both sit ~1% above the strong operand, so the partner adds nothing: reject."""
    messages: list = []
    assert _beats_the_larger_operand(0.505, _marginal, (0, 1), messages, pair_mi=0.51, uplift_bar=1.10) is False
    assert messages


def test_partner_with_real_joint_information_is_not_rejected():
    """The raw pair's joint MI is 1.5x the larger operand, so a low engineered uplift is not a degraded copy: accept."""
    assert _beats_the_larger_operand(0.52, _marginal, (0, 1), None, pair_mi=0.75, uplift_bar=1.10) is True


def test_engineered_column_clearing_the_uplift_is_accepted():
    """An engineered MI 1.4x over the larger operand passes regardless of the joint MI."""
    assert _beats_the_larger_operand(0.70, _marginal, (0, 1), None, pair_mi=0.0, uplift_bar=1.10) is True


def test_plugin_mi_matches_sklearn_and_the_null_bar_is_at_least_one():
    """The njit plug-in MI is sklearn's ``mutual_info_score``; the null bar over a noise partner is never below 1 and stays near it for an uninformative pair."""
    import numpy as np
    from sklearn.metrics import mutual_info_score

    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_operand_floor import _plugin_mi, operand_uplift_null_bar

    rng = np.random.default_rng(0)
    x = rng.integers(0, 7, 4000).astype(np.int64)
    y = rng.integers(0, 3, 4000).astype(np.int64)
    assert _plugin_mi(x, y, 7, 3) == pytest.approx(mutual_info_score(x, y), rel=1e-9, abs=1e-12)
    a = rng.integers(0, 10, 4000).astype(np.int64)
    b = rng.integers(0, 10, 4000).astype(np.int64)
    yy = (a > 4).astype(np.int64)
    floor = _plugin_mi(a, yy, 10, 2)
    bar = operand_uplift_null_bar(a, b, yy, floor)
    assert bar >= 1.0
