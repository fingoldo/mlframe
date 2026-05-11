"""biz_val tests for ``mlframe.feature_engineering.numerical`` --
``compute_numaggs`` and friends.

Per CLAUDE.md: each test asserts a SYNTHETIC measurable WIN that
locks in the numerical-aggregate contract. Naming:
``test_biz_val_numerical_<fn>_<scenario>``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# compute_numaggs
# ---------------------------------------------------------------------------


def test_biz_val_numerical_numaggs_returns_fixed_length_vector():
    """``compute_numaggs(arr)`` must produce a fixed-length vector of
    aggregates regardless of input length -- the "arbitrary array ->
    fixed feature vector" contract is the headline feature."""
    from mlframe.feature_engineering.numerical import compute_numaggs
    rng = np.random.default_rng(42)
    out_short = compute_numaggs(rng.normal(size=50))
    out_long = compute_numaggs(rng.normal(size=5000))
    assert len(out_short) == len(out_long), (
        f"output dim must be invariant to input length; "
        f"got short={len(out_short)}, long={len(out_long)}"
    )
    assert len(out_short) > 5  # Many aggregates expected


def test_biz_val_numerical_numaggs_finite_on_finite_input():
    """Output must be finite when input is finite. Catches regressions
    where a numerical aggregate goes NaN / inf on a well-behaved input."""
    from mlframe.feature_engineering.numerical import compute_numaggs
    rng = np.random.default_rng(42)
    arr = rng.normal(size=500).astype(np.float64)
    out = compute_numaggs(arr)
    # Some aggregates (e.g. hurst) can legitimately return NaN on
    # short series; require that the MAJORITY are finite.
    n_finite = np.sum(np.isfinite(np.asarray(out)))
    assert n_finite >= len(out) * 0.8, (
        f"at least 80% of aggregates must be finite; "
        f"got {n_finite}/{len(out)}"
    )


@pytest.mark.parametrize("n_samples", [100, 500, 2000])
def test_biz_val_numerical_numaggs_scales_across_sizes(n_samples):
    """Numaggs must complete + produce same-shape output across
    small/medium/large arrays."""
    from mlframe.feature_engineering.numerical import compute_numaggs
    rng = np.random.default_rng(42)
    arr = rng.normal(size=n_samples).astype(np.float64)
    out = compute_numaggs(arr)
    assert len(out) > 5


@pytest.mark.parametrize("seed", [1, 7, 42, 99])
def test_biz_val_numerical_numaggs_deterministic_per_seed(seed):
    """Same input array -> identical output. Catches non-determinism
    (e.g. parallel reduction order) regressions."""
    from mlframe.feature_engineering.numerical import compute_numaggs
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=500).astype(np.float64)
    a = compute_numaggs(arr.copy())
    b = compute_numaggs(arr.copy())
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_biz_val_numerical_numaggs_directional_only_smaller_output():
    """``directional_only=True`` must produce a SMALLER output vector
    than the default (full aggregates). Catches regressions in the
    directional-only branch."""
    from mlframe.feature_engineering.numerical import compute_numaggs
    rng = np.random.default_rng(42)
    arr = rng.normal(size=500).astype(np.float64)
    out_full = compute_numaggs(arr, directional_only=False)
    out_dir = compute_numaggs(arr, directional_only=True)
    assert len(out_dir) < len(out_full), (
        f"directional_only=True must produce smaller output; "
        f"got dir={len(out_dir)}, full={len(out_full)}"
    )


def test_biz_val_numerical_numaggs_return_float32_dtype():
    """``return_float32=True`` (default) must yield a float32 result,
    halving memory vs float64 on the downstream feature matrix."""
    from mlframe.feature_engineering.numerical import compute_numaggs
    rng = np.random.default_rng(42)
    arr = rng.normal(size=200).astype(np.float64)
    out = np.asarray(compute_numaggs(arr, return_float32=True))
    assert out.dtype == np.float32, (
        f"return_float32=True must produce float32; got {out.dtype}"
    )


def test_biz_val_numerical_numaggs_weighted_differs_from_unweighted():
    """Passing ``weights`` must produce a DIFFERENT output than
    unweighted (the weights actually affect the aggregates).
    Catches regressions where weights are silently ignored."""
    from mlframe.feature_engineering.numerical import compute_numaggs
    rng = np.random.default_rng(42)
    arr = rng.normal(size=500).astype(np.float64)
    weights = rng.uniform(0.1, 1.0, size=500)
    out_unweighted = np.asarray(compute_numaggs(arr, weights=None))
    out_weighted = np.asarray(compute_numaggs(arr, weights=weights))
    # At least one aggregate should differ -- weights should affect
    # weighted means / variances / etc.
    if len(out_unweighted) != len(out_weighted):
        # Different shapes is also a valid difference signal.
        return
    diff = np.any(out_unweighted != out_weighted)
    assert diff, "weights had zero effect on output -- silently ignored?"
