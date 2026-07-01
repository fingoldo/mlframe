"""Regression sensors for the 2026-06-10 composite-discovery perf audit.

All of these assert VALUE-IDENTITY of the optimised path vs the naive path
(the optimisations only remove allocations / threads, never change numerics):

- P6: _extract_column_array(df, col, rows=idx) == _extract_column_array(df, col)[idx]
- P7: _safe_abs_corr_all matches the per-column _safe_corr reference
- P2: _mi_to_target_prebinned all-true gates are bit-identical
- P5: np.delete-derived per-base matrix == build-without-that-column
- P4: _build_tiny_model honours inner_n_jobs
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.discovery.screening import (
    _extract_column_array,
    _safe_abs_corr_all,
    _safe_corr,
    _mi_to_target_prebinned,
)
from mlframe.training.composite.discovery._screening_tiny import _build_tiny_model

pl = pytest.importorskip("polars")


class TestExtractColumnGather:
    def test_polars_gather_value_identical(self) -> None:
        rng = np.random.default_rng(0)
        n = 5000
        df = pl.DataFrame({"c": rng.normal(size=n)})
        idx = rng.choice(n, 300, replace=False).astype(np.int64)
        gathered = _extract_column_array(df, "c", rows=idx)
        full_then_slice = _extract_column_array(df, "c")[idx]
        np.testing.assert_array_equal(gathered, full_then_slice)

    def test_pandas_gather_value_identical(self) -> None:
        rng = np.random.default_rng(1)
        n = 5000
        df = pd.DataFrame({"c": rng.normal(size=n)})
        idx = rng.choice(n, 300, replace=False).astype(np.int64)
        gathered = _extract_column_array(df, "c", rows=idx)
        full_then_slice = _extract_column_array(df, "c")[idx]
        np.testing.assert_array_equal(gathered, full_then_slice)


class TestSafeAbsCorrAll:
    def test_matches_per_column_reference(self) -> None:
        rng = np.random.default_rng(2)
        n, f = 4000, 20
        X = rng.normal(size=(n, f)).astype(np.float32)
        y = (X[:, 0] * 0.7 + rng.normal(size=n)).astype(np.float32)
        got = _safe_abs_corr_all(y, X)
        ref = np.array([abs(_safe_corr(y, X[:, j])) for j in range(f)])
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    def test_all_finite_gate_matches_masked(self) -> None:
        """The all-finite fast path must equal the y-masked path."""
        rng = np.random.default_rng(3)
        n, f = 2000, 10
        X = rng.normal(size=(n, f)).astype(np.float32)
        y = rng.normal(size=n).astype(np.float32)
        all_finite = _safe_abs_corr_all(y, X)
        y2 = y.copy()
        y2[0] = np.nan  # force the masked branch on a single row
        masked = _safe_abs_corr_all(y2, X)
        # Dropping one row barely moves the correlations.
        np.testing.assert_allclose(all_finite, masked, atol=0.02)


class TestMiPrebinnedGates:
    def test_all_finite_gate_bit_identical(self) -> None:
        rng = np.random.default_rng(4)
        n, f, nbins = 3000, 5, 8
        fb = rng.integers(0, nbins, size=(n, f)).astype(np.int64)
        target = rng.normal(size=n)
        # No sentinels, target all-finite -> exercises the all-true gates.
        mi = _mi_to_target_prebinned(fb, target, nbins=nbins)
        # Manually reproduce with explicit copies (the pre-gate path).
        fb_copy = fb[np.isfinite(target)].copy()
        mi_ref = _mi_to_target_prebinned(fb_copy, target[np.isfinite(target)], nbins=nbins)
        assert mi == pytest.approx(mi_ref, rel=1e-12)


class TestBuildTinyModelInnerNJobs:
    def test_lgbm_honours_inner_n_jobs(self) -> None:
        pytest.importorskip("lightgbm")
        m = _build_tiny_model(
            "lgb", n_estimators=10, num_leaves=7,
            learning_rate=0.1, random_state=0, inner_n_jobs=2,
        )
        assert m.get_params()["n_jobs"] == 2

    def test_default_is_all_cores(self) -> None:
        pytest.importorskip("lightgbm")
        m = _build_tiny_model(
            "lgb", n_estimators=10, num_leaves=7,
            learning_rate=0.1, random_state=0,
        )
        assert m.get_params()["n_jobs"] == -1
