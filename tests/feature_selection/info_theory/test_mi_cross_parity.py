"""Cross-parity tests for the three MI implementations in feature_selection.mi.

The module ships three parallel MI kernels (``grok_compute_mutual_information``,
``chatgpt_compute_mutual_information``, ``deepseek_compute_mutual_information``) that
the production dispatcher chooses between based on a runtime benchmark. Per the
module-level NOTE, the three implementations are load-bearing by design and exist to
catch numerical drift under ``@njit(fastmath=True)``. This test file verifies they
actually AGREE on the same input — silent divergence is exactly the bug the audit
flagged as missing coverage (U14 in tests-expand.md).

The single-pair kernel ``grok_mutual_information`` and the single-target kernel
``grok_mutual_information_old`` are also exercised at the boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.mi import (
    chatgpt_compute_mutual_information,
    deepseek_compute_mutual_information,
    grok_compute_joint_hist,
    grok_compute_mutual_information,
    grok_mutual_information,
    grok_mutual_information_old,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _make_binned_dataset(n: int, k: int, n_bins: int, seed: int) -> np.ndarray:
    """Build an int8 binned matrix of shape (n, k) in [0, n_bins) with one column copy
    of column 0 placed at index 1 so MI(col0, col1) is high."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, n_bins, size=(n, k), dtype=np.int8)
    # Force exact-copy at column 1 so we have a known maximum-MI pair.
    data[:, 1] = data[:, 0]
    return data


@pytest.fixture
def binned_data():
    """Binned data."""
    return _make_binned_dataset(n=500, k=10, n_bins=15, seed=42)


@pytest.fixture
def target_indices():
    """Target indices."""
    return np.array([0, 3], dtype=np.int64)


# ---------------------------------------------------------------------------
# Cross-implementation parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        grok_compute_mutual_information,
        chatgpt_compute_mutual_information,
        deepseek_compute_mutual_information,
    ],
)
def test_each_estimator_returns_expected_shape(fn, binned_data, target_indices):
    """Each estimator returns expected shape."""
    out = fn(binned_data, target_indices, n_bins=15)
    assert out.shape == (len(target_indices), binned_data.shape[1]), (
        f"{fn.__name__}: shape mismatch — got {out.shape}, expected {(len(target_indices), binned_data.shape[1])}"
    )
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out)), f"{fn.__name__}: produced non-finite MI values"


def test_self_mi_is_high_for_all_estimators(binned_data, target_indices):
    """MI(X, X) is the entropy of X. For all 3 estimators this must be substantially
    larger than MI(X, Y) for independent Y. Use uniform synthetic at n_bins=15:
    H(X) ~= log(15) ~ 2.7 nats; independent-Y MI ~ 0."""
    n_bins = 15
    for fn in (grok_compute_mutual_information, chatgpt_compute_mutual_information, deepseek_compute_mutual_information):
        out = fn(binned_data, target_indices, n_bins=n_bins)
        # target_indices=[0, 3]; col 1 is a perfect copy of col 0, so MI(0, 1) should be ~ H(X)
        self_mi_col0 = out[0, 0]
        copy_mi_col1 = out[0, 1]
        assert self_mi_col0 > 1.5, f"{fn.__name__}: self-MI(col0)={self_mi_col0:.3f}, expected > 1.5 nats"
        assert copy_mi_col1 > 1.5, f"{fn.__name__}: copy-MI(col0,col1)={copy_mi_col1:.3f}, expected > 1.5 nats"


def test_cross_implementation_parity_within_tolerance(binned_data, target_indices):
    """All three implementations must agree within 5% relative tolerance on the same
    integer input — this is THE highest-leverage MI test (per U14 in tests-expand audit).

    Silent divergence between estimators is the bug class we're locking down."""
    out_grok = grok_compute_mutual_information(binned_data, target_indices, n_bins=15)
    out_chatgpt = chatgpt_compute_mutual_information(binned_data, target_indices, n_bins=15)
    out_deepseek = deepseek_compute_mutual_information(binned_data, target_indices, n_bins=15)

    # Mathematically the three kernels compute the same quantity, only the summation
    # order differs (fastmath=True can introduce ~1e-14 relative drift; 5% gives ample headroom).
    np.testing.assert_allclose(
        out_grok,
        out_chatgpt,
        rtol=0.05,
        atol=1e-9,
        err_msg="grok vs chatgpt MI estimators diverged beyond 5% relative tolerance",
    )
    np.testing.assert_allclose(
        out_grok,
        out_deepseek,
        rtol=0.05,
        atol=1e-9,
        err_msg="grok vs deepseek MI estimators diverged beyond 5% relative tolerance",
    )
    np.testing.assert_allclose(
        out_chatgpt,
        out_deepseek,
        rtol=0.05,
        atol=1e-9,
        err_msg="chatgpt vs deepseek MI estimators diverged beyond 5% relative tolerance",
    )


def test_cross_implementation_tight_parity_on_self_mi(binned_data, target_indices):
    """Self-MI (column vs itself) is purely entropy-driven — no cross-term cancellation
    differences between the three formulas. Tight 1% tolerance here."""
    out_grok = grok_compute_mutual_information(binned_data, target_indices, n_bins=15)
    out_chatgpt = chatgpt_compute_mutual_information(binned_data, target_indices, n_bins=15)
    out_deepseek = deepseek_compute_mutual_information(binned_data, target_indices, n_bins=15)

    # target_indices=[0, 3]; self-MI is on the diagonal of the (n_targets, n_columns) block
    np.testing.assert_allclose(out_grok[0, 0], out_chatgpt[0, 0], rtol=0.01, atol=1e-9)
    np.testing.assert_allclose(out_grok[0, 0], out_deepseek[0, 0], rtol=0.01, atol=1e-9)
    np.testing.assert_allclose(out_grok[1, 3], out_chatgpt[1, 3], rtol=0.01, atol=1e-9)
    np.testing.assert_allclose(out_grok[1, 3], out_deepseek[1, 3], rtol=0.01, atol=1e-9)


# ---------------------------------------------------------------------------
# Single-pair / single-target kernels
# ---------------------------------------------------------------------------


def test_grok_compute_joint_hist_counts():
    """joint hist must count exactly the (a[i], b[i]) pairs once each."""
    a = np.array([0, 1, 2, 0, 1], dtype=np.int8)
    b = np.array([0, 1, 2, 1, 0], dtype=np.int8)
    h = grok_compute_joint_hist(a, b, n_bins=3, dtype=np.int64)
    assert h.shape == (3, 3)
    assert int(h.sum()) == 5, "joint hist row-sum must equal len(a)"
    assert h[0, 0] == 1
    assert h[1, 1] == 1
    assert h[2, 2] == 1
    assert h[0, 1] == 1
    assert h[1, 0] == 1


def test_grok_mutual_information_pair_matches_matrix_form():
    """Single-pair ``grok_mutual_information(col_a, col_b, ...)`` must equal the
    corresponding cell of the matrix-form ``grok_compute_mutual_information``."""
    data = _make_binned_dataset(n=300, k=4, n_bins=10, seed=7)
    n = data.shape[0]
    inv_n = 1.0 / n
    log_n = np.log(n)
    pair_mi = grok_mutual_information(data[:, 0], data[:, 2], inv_n_samples=inv_n, log_n_samples=log_n, n_bins=10)
    matrix_mi = grok_compute_mutual_information(data, np.array([0], dtype=np.int64), n_bins=10)
    np.testing.assert_allclose(pair_mi, matrix_mi[0, 2], rtol=1e-9, atol=1e-12, err_msg="pair-form grok MI must equal matrix-form cell exactly")


def test_grok_mutual_information_old_consistent_with_new():
    """The deprecated ``grok_mutual_information_old`` must still numerically agree with
    the current ``grok_mutual_information`` (used as a historical-reference correctness probe)."""
    data = _make_binned_dataset(n=300, k=4, n_bins=10, seed=11)
    n = data.shape[0]
    inv_n = 1.0 / n
    log_n = np.log(n)
    new = grok_mutual_information(data[:, 0], data[:, 1], inv_n_samples=inv_n, log_n_samples=log_n, n_bins=10)
    old = grok_mutual_information_old(data[:, 0], data[:, 1], n_bins=10)
    np.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-12, err_msg="grok_mutual_information vs grok_mutual_information_old diverged")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        grok_compute_mutual_information,
        chatgpt_compute_mutual_information,
        deepseek_compute_mutual_information,
    ],
)
def test_empty_data_returns_zero_matrix(fn):
    """n_samples == 0 must return a zero MI matrix without dividing by zero."""
    empty = np.zeros((0, 5), dtype=np.int8)
    target_indices = np.array([0, 1], dtype=np.int64)
    out = fn(empty, target_indices, n_bins=15)
    assert out.shape == (2, 5)
    assert np.all(out == 0.0), f"{fn.__name__}: empty input must yield zero-MI matrix; got {out!r}"


@pytest.mark.parametrize(
    "fn",
    [
        grok_compute_mutual_information,
        chatgpt_compute_mutual_information,
        deepseek_compute_mutual_information,
    ],
)
def test_constant_feature_zero_mi(fn):
    """A constant feature carries zero information about any target — MI must be ~0."""
    rng = np.random.default_rng(99)
    data = rng.integers(0, 15, size=(500, 5), dtype=np.int8)
    data[:, 2] = 3  # constant column at index 2
    target_indices = np.array([0], dtype=np.int64)
    out = fn(data, target_indices, n_bins=15)
    # MI(target_random, constant_feature) ~ 0
    assert abs(out[0, 2]) < 1e-9, f"{fn.__name__}: MI(random_target, constant_feature) should be ~0; got {out[0, 2]!r}"


@pytest.mark.parametrize(
    "fn",
    [
        grok_compute_mutual_information,
        chatgpt_compute_mutual_information,
        deepseek_compute_mutual_information,
    ],
)
def test_perfect_copy_gives_max_mi(fn):
    """When feature is a perfect copy of target, MI must equal feature entropy (>> independent baseline)."""
    rng = np.random.default_rng(123)
    data = rng.integers(0, 15, size=(2000, 4), dtype=np.int8)
    data[:, 1] = data[:, 0]  # column 1 is a perfect copy of target (column 0)
    target_indices = np.array([0], dtype=np.int64)
    out = fn(data, target_indices, n_bins=15)
    # MI(col0, col1) ~= H(col0) ~= log(15) ~ 2.7 nats; independent baseline ~ 0.01-0.02 nats at n=2000
    assert out[0, 1] > 2.0, f"{fn.__name__}: MI(target, perfect_copy) must be high; got {out[0, 1]!r}"
    # Independent col index 2 or 3
    assert out[0, 2] < 0.5, f"{fn.__name__}: MI(target, independent) must be small; got {out[0, 2]!r}"
    assert out[0, 1] > out[0, 2] * 4, f"{fn.__name__}: perfect-copy MI must dominate independent baseline; got copy={out[0, 1]:.3f}, indep={out[0, 2]:.3f}"


def test_chatgpt_rejects_out_of_range_bins():
    """Wave 40 guard: input bin codes outside [0, 127] must raise ValueError (not silently
    wrap to negative). Verifies the explicit min/max check at module-level line 195-202."""
    data = np.random.randint(0, 200, size=(100, 5), dtype=np.int32)  # values > 127
    target_indices = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match=r"bin codes must be in \[0, 127\]"):
        chatgpt_compute_mutual_information(data, target_indices, n_bins=15)
