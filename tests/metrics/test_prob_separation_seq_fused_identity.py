"""Identity regression for the fused SEQ probability_separation_score kernel.

Pins ``_probability_separation_score_seq`` (now a zero-alloc two-pass scalar
kernel) bit-against a reference reimplementation of the prior alloc-heavy
form (boolean mask + fancy-index copy + np.mean + np.std). Catches a future
regression that changes the returned score, and confirms the seq path agrees
with the parallel path it dispatches into above n=50k.
"""

import numpy as np
import pytest

from mlframe.metrics._log_loss_and_separation import (
    _probability_separation_score_seq,
    _probability_separation_score_par,
    probability_separation_score,
)


def _reference_old(y_true, y_prob, class_label=1, std_weight=0.5):
    """The pre-fusion implementation (mask + masked copy + np.mean/np.std)."""
    idx = y_true == class_label
    if idx.sum() == 0:
        return np.nan
    res = float(np.mean(y_prob[idx]))
    if std_weight != 0.0:
        addend = float(np.std(y_prob[idx])) * std_weight
        res = res - addend if class_label == 1 else res + addend
    return res


@pytest.mark.parametrize("n", [1, 2, 100, 2_000, 49_000])
@pytest.mark.parametrize("class_label", [0, 1])
@pytest.mark.parametrize("std_weight", [0.0, 0.5, 1.0])
def test_seq_fused_matches_reference(n, class_label, std_weight):
    rng = np.random.default_rng(n + class_label)
    yt = rng.integers(0, 2, size=n).astype(np.float64)
    yp = rng.random(n)
    got = _probability_separation_score_seq(yt, yp, class_label, std_weight)
    ref = _reference_old(yt, yp, class_label, std_weight)
    if np.isnan(ref):
        assert np.isnan(got)
    else:
        # reduction-order tolerance (running sum vs np pairwise); contract ~1e-15 rel
        assert got == pytest.approx(ref, rel=1e-12, abs=1e-12)


def test_seq_empty_in_class_returns_nan():
    yt = np.zeros(50, dtype=np.float64)  # no class_label==1 present
    yp = np.random.default_rng(1).random(50)
    assert np.isnan(_probability_separation_score_seq(yt, yp, 1, 0.5))


def test_seq_agrees_with_par():
    rng = np.random.default_rng(7)
    n = 60_000
    yt = rng.integers(0, 2, size=n).astype(np.float64)
    yp = rng.random(n)
    for cl in (0, 1):
        for sw in (0.0, 0.5):
            seq = _probability_separation_score_seq(yt, yp, cl, sw)
            par = _probability_separation_score_par(yt, yp, cl, sw)
            assert seq == pytest.approx(par, rel=1e-10, abs=1e-10)


def test_public_dispatch_smoke():
    rng = np.random.default_rng(3)
    yt = rng.integers(0, 2, size=5_000).astype(np.float64)
    yp = rng.random(5_000)
    val = probability_separation_score(yt, yp, 1, 0.5)
    assert np.isfinite(val)
