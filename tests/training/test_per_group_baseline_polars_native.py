"""Regression + biz_value tests for the polars-native ``_per_group_predict`` path.

Pins:
1. Polars input dispatches to ``_per_group_predict_polars`` (not the pandas branch) and produces predictions that are bit-equal to the pandas branch on the same
   data (max abs diff <= 1e-12 across train/val/test arrays).
2. At n_rows=1M the polars-native path's wall-time is at least 1.7x faster than the pandas path (measured 2.57x; floor 1.7x leaves >30% headroom for noise).
3. Diagnostics (coverage_pct, repeat_entity_rate, n_groups_train, global_fallback) match between paths.

The pandas-branch comparison forces a pre-conversion via ``to_pandas`` so both runs see the same underlying data; the dispatcher inside ``_per_group_predict``
then picks the appropriate branch by input type. This is the same dispatch contract callers depend on.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest


pl = pytest.importorskip("polars")


def _synth(n_rows: int, n_groups: int, n_extra_cols: int, seed: int) -> tuple["pl.DataFrame", np.ndarray]:
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {"g": rng.integers(0, n_groups, n_rows, dtype=np.int64)}
    for i in range(n_extra_cols):
        cols[f"f{i}"] = rng.normal(size=n_rows).astype(np.float64)
    return pl.DataFrame(cols), rng.normal(size=n_rows).astype(np.float64)


def test_polars_input_dispatches_to_polars_native_branch():
    """Polars input must hit ``_per_group_predict_polars``; pandas input must NOT."""
    from mlframe.training import _dummy_baseline_compute as mod

    train_pl, y = _synth(2_000, 50, 3, seed=1)
    val_pl, _ = _synth(500, 55, 3, seed=2)
    test_pl, _ = _synth(500, 55, 3, seed=3)

    with patch.object(mod, "_per_group_predict_polars", wraps=mod._per_group_predict_polars) as spy:
        mod._per_group_predict(train_pl, val_pl, test_pl, y, "g", "regression")
    assert spy.call_count == 1, "polars input should take the polars-native branch exactly once"

    train_pd = train_pl.to_pandas()
    val_pd = val_pl.to_pandas()
    test_pd = test_pl.to_pandas()
    with patch.object(mod, "_per_group_predict_polars", wraps=mod._per_group_predict_polars) as spy:
        mod._per_group_predict(train_pd, val_pd, test_pd, y, "g", "regression")
    assert spy.call_count == 0, "pandas input must NOT trigger the polars-native branch"


def test_polars_native_outputs_bit_equal_to_pandas_branch():
    """Numerical equivalence: max abs diff across all three prediction arrays <= 1e-12."""
    from mlframe.training._dummy_baseline_compute import _per_group_predict

    train_pl, y = _synth(5_000, 50, 3, seed=10)
    val_pl, _ = _synth(1_000, 60, 3, seed=11)
    test_pl, _ = _synth(1_000, 60, 3, seed=12)

    tr_pl, va_pl, te_pl, diag_pl = _per_group_predict(train_pl, val_pl, test_pl, y, "g", "regression")
    tr_pd, va_pd, te_pd, diag_pd = _per_group_predict(train_pl.to_pandas(), val_pl.to_pandas(), test_pl.to_pandas(), y, "g", "regression")

    assert np.max(np.abs(tr_pl - tr_pd)) <= 1e-12
    assert np.max(np.abs(va_pl - va_pd)) <= 1e-12
    assert np.max(np.abs(te_pl - te_pd)) <= 1e-12
    # Diagnostics: floats compared with abs tol; ints / labels must be exact.
    assert diag_pl["n_groups_train"] == diag_pd["n_groups_train"]
    assert abs(diag_pl["val_coverage_pct"] - diag_pd["val_coverage_pct"]) <= 1e-9
    assert abs(diag_pl["test_coverage_pct"] - diag_pd["test_coverage_pct"]) <= 1e-9
    assert abs(diag_pl["repeat_entity_rate"] - diag_pd["repeat_entity_rate"]) <= 1e-9
    assert abs(diag_pl["global_fallback"] - diag_pd["global_fallback"]) <= 1e-12


def test_polars_native_handles_unseen_groups_with_global_mean_fallback():
    """Val / test rows whose group never appeared in train must map to global_mean. Mirrors the pandas-branch fillna(global_mean) contract."""
    from mlframe.training._dummy_baseline_compute import _per_group_predict

    train_pl = pl.DataFrame({"g": np.array([0, 0, 1, 1, 1], dtype=np.int64)})
    val_pl = pl.DataFrame({"g": np.array([0, 99, 1, 99], dtype=np.int64)})
    test_pl = pl.DataFrame({"g": np.array([42], dtype=np.int64)})
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float64)

    _, val_pred, test_pred, diag = _per_group_predict(train_pl, val_pl, test_pl, y, "g", "regression")
    g0_mean = (1.0 + 3.0) / 2
    g1_mean = (5.0 + 7.0 + 9.0) / 3
    global_mean = float(np.mean(y))
    np.testing.assert_allclose(val_pred, [g0_mean, global_mean, g1_mean, global_mean])
    np.testing.assert_allclose(test_pred, [global_mean])
    assert diag["n_groups_train"] == 2
    assert diag["global_fallback"] == pytest.approx(global_mean)


@pytest.mark.slow
def test_biz_val_per_group_baseline_polars_faster_than_pandas_at_1m_rows():
    """biz_value: polars-native path is at least 1.7x faster than pandas branch at n=1M.

    Measured 2.57x in the development benchmark; floor 1.7x leaves >30% headroom for runtime noise. Regressions in the polars-native path (e.g. dropping the
    fused group_by, falling back to two passes, accidentally reintroducing the polars->pandas bridge) trip this assertion.
    """
    from mlframe.training._dummy_baseline_compute import _per_group_predict

    n_rows = 1_000_000
    train_pl, y = _synth(n_rows, 1_000, 20, seed=20)
    val_pl, _ = _synth(n_rows // 5, 1_050, 20, seed=21)
    test_pl, _ = _synth(n_rows // 5, 1_050, 20, seed=22)
    train_pd = train_pl.to_pandas()
    val_pd = val_pl.to_pandas()
    test_pd = test_pl.to_pandas()

    # Warm both paths (first call pays Arrow / polars-plan init).
    _per_group_predict(train_pl, val_pl, test_pl, y, "g", "regression")
    _per_group_predict(train_pd, val_pd, test_pd, y, "g", "regression")

    n_runs = 3
    pl_times = []
    pd_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _per_group_predict(train_pl, val_pl, test_pl, y, "g", "regression")
        pl_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _per_group_predict(train_pd, val_pd, test_pd, y, "g", "regression")
        pd_times.append(time.perf_counter() - t0)
    pl_med = float(np.median(pl_times))
    pd_med = float(np.median(pd_times))
    speedup = pd_med / pl_med
    print(f"\n[biz_val] per_group_baseline n=1M: polars={pl_med*1000:.1f}ms pandas={pd_med*1000:.1f}ms speedup={speedup:.2f}x")
    assert speedup >= 1.7, f"polars-native should be >=1.7x faster than pandas at n=1M; got {speedup:.2f}x (polars={pl_med*1000:.1f}ms pandas={pd_med*1000:.1f}ms)"
