"""Tests for mlframe.metrics.scoring — the salvaged Models.py symbols."""

import numpy as np
import pytest

from mlframe.metrics.scoring import (
    ProbaScoreProxy,
    fast_rmse,
    LogUniform,
    rmse_loss,
    rmse_score,
    rmsle_loss,
    rmsle_score,
)


def test_rmse_loss_zero_on_match():
    """Rmse loss zero on match."""
    y = np.array([1.0, 2.0, 3.0])
    assert rmse_loss(y, y) == 0.0


def test_rmse_loss_known_value():
    # sqrt(mean([1,1,1])) == 1
    """Rmse loss known value."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert rmse_loss(y_true, y_pred) == pytest.approx(1.0)


def test_fast_rmse_matches_rmse_loss():
    """iter367: fast_rmse is the numba single-pass kernel used inside the
    honest-diagnostics bootstrap regression path. Must produce the same
    scalar as the numpy rmse_loss for any aligned (y_true, y_pred) pair --
    fastmath=True is enabled but the simple sum-of-squares reduction
    converges to the same float64 value on aligned arrays."""
    rng = np.random.default_rng(20260527)
    for n in (10, 1000, 50_000):
        y = rng.normal(size=n)
        p = y + rng.normal(scale=0.5, size=n)
        a = fast_rmse(y, p)
        b = float(rmse_loss(y, p))
        assert abs(a - b) < 1e-9, f"n={n}: fast_rmse={a} vs rmse_loss={b}"


def test_fast_rmse_zero_on_match():
    """Fast rmse zero on match."""
    y = np.array([1.0, 2.0, 3.0])
    assert fast_rmse(y, y) == 0.0


def test_fast_rmse_handles_int_inputs():
    """Caller may pass int arrays (resampled from integer index arrays); the
    function must cast to float64 internally without raising."""
    y = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    p = np.array([2, 3, 4, 5, 6], dtype=np.int64)
    assert fast_rmse(y, p) == pytest.approx(1.0)


def test_fast_rmse_mixed_dtypes_bit_equivalent():
    """iter595: fast_rmse drops the unconditional ``dtype=np.float64`` cast
    in favour of numba's mixed-dtype dispatch. Pin bit-equivalence with the
    numpy reference across the dtype pairs that appear in the honest-
    diagnostics bootstrap loop: (int labels, float64 probs) when a
    classification target reaches the regression-ish fallback path, and
    (float64, float32) when a torch/MLP model emits float32 means."""
    rng = np.random.default_rng(20260530)
    n = 50_000
    y_int64 = rng.integers(0, 3, size=n, dtype=np.int64)
    y_int32 = y_int64.astype(np.int32)
    p_f64 = rng.random(n)
    p_f32 = p_f64.astype(np.float32)
    y_f64 = rng.random(n)

    for y_t, y_p, atol in [
        (y_int64, p_f64, 1e-9),
        (y_int32, p_f64, 1e-9),
        (y_f64, p_f64, 1e-9),
        (y_f64, p_f32, 1e-5),
    ]:
        a = fast_rmse(y_t, y_p)
        b = float(rmse_loss(y_t, y_p))
        assert abs(a - b) < atol, f"dtypes ({y_t.dtype}, {y_p.dtype}): fast_rmse={a} vs rmse_loss={b}"


def test_fast_rmse_handles_non_contiguous_input():
    """iter595: ``np.ascontiguousarray`` replaces the asarray+dtype cast and
    must still upgrade a strided view to a contiguous array so the numba
    kernel sees a valid layout. Use a view-of-larger-buffer (every-other
    element) to exercise the copying branch."""
    rng = np.random.default_rng(20260530)
    full = rng.random(200)
    y = full[::2]
    p = full[1::2] + 0.1
    assert not y.flags.c_contiguous
    a = fast_rmse(y, p)
    b = float(rmse_loss(np.ascontiguousarray(y), np.ascontiguousarray(p)))
    assert abs(a - b) < 1e-9


def test_rmsle_loss_clips_negative_predictions():
    """Rmsle loss clips negative predictions."""
    y_true = np.array([1.0, 2.0])
    y_pred_neg = np.array([-5.0, 2.0])
    y_pred_zero = np.array([0.0, 2.0])
    # negative preds clipped to 0 → same loss as predicting 0
    assert rmsle_loss(y_true, y_pred_neg) == pytest.approx(rmsle_loss(y_true, y_pred_zero))


def test_rmsle_loss_zero_on_match():
    """Rmsle loss zero on match."""
    y = np.array([0.5, 1.5, 10.0])
    assert rmsle_loss(y, y) == pytest.approx(0.0)


def test_rmse_and_rmsle_scorers_greater_is_better_false():
    # make_scorer with greater_is_better=False negates the output
    """Rmse and rmsle scorers greater is better false."""
    assert rmse_score._sign == -1
    assert rmsle_score._sign == -1


def test_log_uniform_bounds():
    """Log uniform bounds."""
    lu = LogUniform(a=-2, b=2, base=10)
    samples = lu.rvs(size=500, random_state=42)
    assert samples.shape == (500,)
    assert np.all(samples >= 10**-2 - 1e-9)
    assert np.all(samples <= 10**2 + 1e-9)


def test_log_uniform_scalar():
    """Log uniform scalar."""
    lu = LogUniform(a=0, b=1, base=10)
    val = lu.rvs(random_state=1)
    assert np.isscalar(val) or val.shape == ()
    assert 1 <= float(val) <= 10


def test_log_uniform_random_state_reproducible():
    """Log uniform random state reproducible."""
    lu = LogUniform(-1, 1, base=10)
    a = lu.rvs(size=10, random_state=7)
    b = lu.rvs(size=10, random_state=7)
    np.testing.assert_array_equal(a, b)


def test_proba_score_proxy_selects_column():
    """Proba score proxy selects column."""
    from sklearn.metrics import roc_auc_score

    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.3, 0.7], [0.6, 0.4]])
    # class 1 column == proba of positive class
    expected = roc_auc_score(y_true, y_probs[:, 1])
    assert ProbaScoreProxy(y_true, y_probs, class_idx=1, proxied_func=roc_auc_score) == expected
