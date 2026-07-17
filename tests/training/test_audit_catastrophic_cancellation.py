"""Wave-25 sensor: catastrophic-cancellation + Shannon-entropy formula fixes.

#1 P1 models/ensembling.py:_per_member_mae_std_njit
   Pre-fix computed std of |diff| via the naive `E[X^2] - E[X]^2`
   formula. When the regression-scale MAE was large (1e3+) and the
   spread of |d| was small, ``_s_sq/N`` ≈ ``mae*mae`` and the
   subtraction lost precision (catastrophic cancellation). The ``if
   _var < 0: _var = 0.0`` clamp was a band-aid that proved the
   author saw the problem. ``fastmath=True`` compounded the drift.
   Post-fix: two-pass form (compute mean, then sum deviation^2
   directly). No cancellation. ``fastmath=False`` to keep reductions
   associative-stable.

#2 P1 same shape at the 3-D (K, N, C) branch (line 78).

#3 BONUS feature_engineering/numerical.py:cont_entropy
   Audit flagged this outside the assigned class: pre-fix computed
   ``-(hist * log(hist + eps)).sum()`` where ``hist`` was raw integer
   COUNTS, not normalised probabilities. That formula is NOT Shannon
   entropy; it scales with bin count and total sample size, returning
   values like 12_345 nats instead of the (0, log(n_bins)) range.
   The commented hint ``# np.histogram(arr, bins=bins, density=True)``
   in the pre-fix source shows the author spotted the missing
   normalisation but didn't apply it.
   Post-fix: explicit ``p = counts / counts.sum()`` + ``-sum(p log p)``
   over nonzero-p bins.

Per `feedback_richness_first`: addressed the bonus finding too.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---- #1: variance precision on large-mean small-spread input ------------


def test_per_member_std_precision_on_large_mean_small_spread():
    """Construct a (K=2, N=1M) array where MAE is ~1000 and spread is ~0.001
    -- the regime where the pre-fix `E[X^2]-E[X]^2` formula loses precision.
    Assert post-fix std matches a numpy ddof=0 ground truth within 1e-10
    relative (pre-fix would be ~1e-3)."""
    from mlframe.models.ensembling import (
        _per_member_mae_std_njit,
        _HAS_NUMBA_PER_MEMBER,
    )

    if not _HAS_NUMBA_PER_MEMBER:
        pytest.skip("numba not available; per-member-mae-std njit path unused")

    rng = np.random.default_rng(42)
    median_preds = np.zeros(1_000_000, dtype=np.float64)
    arr = (1000.0 + rng.normal(0, 0.001, size=(2, 1_000_000))).astype(np.float64)
    out_mae, out_std = _per_member_mae_std_njit(arr, median_preds)
    expected_std = np.std(np.abs(arr - median_preds), axis=1, ddof=0)
    rel_err = np.abs(out_std - expected_std) / expected_std
    assert rel_err.max() < 1e-10, (
        f"Wave 25 P1 regression: std precision degraded to {rel_err.max():.2e} "
        f"-- catastrophic-cancellation `E[X^2]-E[X]^2` formula reverted. "
        f"Post-fix two-pass form gives ~1e-13 relative error; pre-fix "
        f"got ~1e-3 on regression-scale inputs."
    )


def test_per_member_std_3d_branch_precision():
    """Same check for the 3-D (K, N, C) multi-class branch (line 78)."""
    from mlframe.models.ensembling import (
        _per_member_mae_std_njit,
        _HAS_NUMBA_PER_MEMBER,
    )

    assert _HAS_NUMBA_PER_MEMBER, "numba is a hard dependency per pyproject.toml; install is broken if False"

    rng = np.random.default_rng(7)
    K, N, C = 2, 100_000, 3
    median_preds = np.zeros((N, C), dtype=np.float64)
    arr = (1000.0 + rng.normal(0, 0.001, size=(K, N, C))).astype(np.float64)
    out_mae, out_std = _per_member_mae_std_njit(arr, median_preds)
    # Ground truth matches the kernel + the _numpy_3d reference: per-COLUMN std over the N axis (anchored at each
    # column's own mean), then averaged across C. A pooled (N*C)-flattened std folds in the between-column mean scatter
    # and is a DIFFERENT statistic (the ~5e-6 finite-sample gap the kernel comment documents), not a precision loss.
    diffs = np.abs(arr - median_preds[None, :, :])
    expected_std = np.std(diffs, axis=1, ddof=0).mean(axis=1)
    rel_err = np.abs(out_std - expected_std) / expected_std
    assert rel_err.max() < 1e-10, f"Wave 25 P1 regression (3-D branch): std precision degraded to {rel_err.max():.2e}."


def test_per_member_std_constant_input_is_exactly_zero_not_negative():
    """The two-pass form yields a non-negative variance by construction (no clamp band-aid).
    On a constant |diff| input the naive ``E[X^2]-E[X]^2`` form could go slightly negative;
    the post-fix kernel must return exactly 0.0 std with no NaN/negative."""
    from mlframe.models.ensembling.base import _per_member_mae_std_njit

    median = np.zeros(100_000, dtype=np.float64)
    arr = np.full((2, 100_000), 1000.0, dtype=np.float64)
    _mae, out_std = _per_member_mae_std_njit(arr, median)
    assert np.all(out_std >= 0.0), "two-pass variance must never be negative"
    assert np.allclose(out_std, 0.0, atol=0.0), "constant |diff| must give exactly 0 std"


# ---- #3 BONUS: Shannon entropy formula -----------------------------------


def test_cont_entropy_is_actually_shannon_not_count_scaled():
    """Pre-fix ``cont_entropy`` returned raw-count-scaled values
    (~10_000 nats on 10k samples). Post-fix returns proper Shannon
    entropy in nats, bounded by log(n_bins) for any reasonable bin
    choice."""
    from mlframe.feature_engineering.numerical import cont_entropy

    rng = np.random.default_rng(123)
    arr_uniform = rng.uniform(0, 1, size=10000)
    ent = cont_entropy(arr_uniform)
    # Scott's rule picks ~30-60 bins for uniform [0,1] @ n=10k; uniform
    # over those bins has entropy log(n_bins) <= log(100) ≈ 4.6 nats.
    # Pre-fix this value was orders of magnitude larger.
    assert 0.5 < ent < 6.0, (
        f"Wave 25 BONUS regression: cont_entropy(uniform) returned "
        f"{ent:.3f} nats, outside the expected Shannon range. Pre-fix "
        f"returned raw-count-scaled values (~10000 nats); post-fix should "
        f"return entropy in (0, log(n_bins))."
    )


def test_cont_entropy_concentrated_lower_than_log_n_bins():
    """For a very-concentrated distribution most bins are empty -> the
    nonzero-prob mass is concentrated in few bins -> entropy is bounded
    by log(N_nonzero). Just assert finiteness + non-negativity here;
    the precise value depends on Scott's-rule bin-count which itself
    depends on input std."""
    from mlframe.feature_engineering.numerical import cont_entropy

    rng = np.random.default_rng(123)
    arr_concentrated = rng.normal(0.5, 0.01, size=10000)
    ent = cont_entropy(arr_concentrated)
    assert np.isfinite(ent), "cont_entropy should return finite value on well-formed input"
    assert ent >= 0.0, "Shannon entropy is non-negative"


def test_cont_entropy_empty_returns_nan():
    """Empty input -> NaN (preserves pre-fix behaviour for edge case)."""
    from mlframe.feature_engineering.numerical import cont_entropy

    out = cont_entropy(np.array([]))
    assert np.isnan(out)


def test_cont_entropy_is_sample_size_invariant_not_count_scaled():
    """Proper Shannon entropy (on normalised probabilities) is ~invariant to sample size for the
    same distribution; the pre-fix count-scaled formula grew roughly linearly with n. Drawing the
    same uniform at 10x the sample size must NOT 10x the entropy."""
    from mlframe.feature_engineering.numerical import cont_entropy

    rng = np.random.default_rng(123)
    ent_small = cont_entropy(rng.uniform(0, 1, size=10_000))
    ent_large = cont_entropy(rng.uniform(0, 1, size=100_000))
    assert abs(ent_large - ent_small) < 1.0, (
        f"cont_entropy must be ~sample-size invariant (Shannon on probabilities); "
        f"got {ent_small:.3f} vs {ent_large:.3f} -- count-scaled regression scales with n"
    )
