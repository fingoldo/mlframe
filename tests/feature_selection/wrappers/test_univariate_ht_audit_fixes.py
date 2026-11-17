"""Univariate-HT prescreen: int64 overflow of the tie sums, string / nullable targets, throttled Cochran warning."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.wrappers import _univariate_ht as uht
from mlframe.utils.log_throttle import reset_throttle_counts


def test_rank_and_tiesum_does_not_overflow_int64_on_a_huge_tie_group():
    """A tie group above 2.1M rows has t**3 above int64; the tie sum must stay the exact float t**3 - t."""
    t = 2_200_000
    _, tie_sum = uht._rank_and_tiesum(np.zeros(t))
    assert tie_sum == pytest.approx(float(t) ** 3 - t, rel=1e-12)


def test_kruskal_wallis_tie_correction_matches_scipy_above_int64_cube_limit():
    """H on 2.2M rows with three heavy tie groups matches scipy (n**3 - n overflowed int64 before)."""
    scipy_stats = pytest.importorskip("scipy.stats")
    n = 2_200_000
    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, n).astype(np.float64)
    g = rng.integers(0, 2, n)
    H, df = uht._kruskal_wallis_h_v2(x, g.astype(np.int64), 2)
    ref = scipy_stats.kruskal(x[g == 0], x[g == 1]).statistic
    assert df == 1
    assert H == pytest.approx(ref, rel=1e-6)


def test_classify_target_accepts_string_labels():
    """String / object class labels are classification in auto mode instead of raising in np.floor."""
    assert uht._classify_target(np.array(["a", "b"] * 5)) == "binary"
    assert uht._classify_target(np.array(["a", "b", "c"] * 5, dtype=object)) == "multiclass"


def test_cochran_warning_is_throttled_and_logged(caplog):
    """A sparse contingency table logs the Cochran warning through log_throttle at WARNING."""
    reset_throttle_counts("univariate_ht_cochran_low_expected")
    x = np.array(["a", "b", "c", "d", "e", "f"])
    y = np.array([0, 1, 0, 1, 0, 1])
    with caplog.at_level(logging.WARNING, logger=uht.logger.name):
        for _ in range(10):
            uht._chi2_independence_p(x, y)
    recs = [r for r in caplog.records if "Cochran" in r.getMessage()]
    assert 1 <= len(recs) <= 5
