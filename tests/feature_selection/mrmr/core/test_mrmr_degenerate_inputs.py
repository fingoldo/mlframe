"""Degenerate-input edge cases for MRMR.fit (mrmr_audit_2026-07-20 edge_cases.md #115-135, #191-203).
Covers n=1 sample fits, an all-NaN column, bit-identical duplicate columns, constant y, and the
sample_weight validation guards (all-zero / negative / NaN / Inf / dominant-weight resampling) --
each previously exercised only incidentally (if at all) inside larger adversarial suites."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR


def _kw(**overrides):
    """Fast-fitting default MRMR constructor kwargs, overridable per test."""
    base = dict(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, fe_max_steps=0, skip_retraining_on_same_content=False)
    base.update(overrides)
    return base


class TestSingleRowFit:
    """n=1 is a concrete division-by-near-zero risk (Miller-Madow bias term uses 2*n_samples)."""

    def test_classification_single_row_does_not_crash(self):
        """A single-row classification fit must either succeed or raise a clear, documented error --
        never a bare divide-by-zero deep inside MI/binning machinery."""
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        y = pd.Series([0])
        m = MRMR(**_kw())
        try:
            m.fit(X, y)
        except (ValueError, RuntimeError):
            pass  # a clear, documented exception is an acceptable outcome for n=1

    def test_regression_single_row_does_not_crash(self):
        """Same contract as the classification case, for a single-row regression target."""
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        y = pd.Series([3.14])
        m = MRMR(**_kw())
        try:
            m.fit(X, y)
        except (ValueError, RuntimeError):
            pass


class TestAllNaNColumn:
    """An all-NaN column must be excluded from support_, never silently poison another column's MI."""

    def test_all_nan_column_excluded_from_support(self):
        """A fully-NaN column must never end up in support_ alongside a genuinely predictive column."""
        rng = np.random.default_rng(0)
        n = 200
        good = rng.standard_normal(n)
        y = (good > 0).astype(int)
        X = pd.DataFrame({"good": good, "allnan": np.full(n, np.nan)})
        m = MRMR(**_kw())
        m.fit(X, y)
        support = list(getattr(m, "support_", []))
        cols = list(X.columns)
        selected_names = {cols[i] for i, s in enumerate(support) if s} if support and isinstance(support[0], (bool, np.bool_)) else set(cols[i] for i in support)
        assert "allnan" not in selected_names


class TestDuplicateColumns:
    """Bit-identical duplicate columns: at most one twin should survive the redundancy gate."""

    def test_duplicate_column_redundancy_dropped(self):
        """Two bit-identical signal-carrying columns must not both survive into support_."""
        rng = np.random.default_rng(1)
        n = 300
        a = rng.standard_normal(n)
        y = (a > 0).astype(int)
        X = pd.DataFrame({"a": a, "a_dup": a.copy(), "noise": rng.standard_normal(n)})
        m = MRMR(**_kw())
        m.fit(X, y)
        support = list(getattr(m, "support_", []))
        cols = list(X.columns)
        selected_names = {cols[i] for i, s in enumerate(support) if s} if support and isinstance(support[0], (bool, np.bool_)) else set(cols[i] for i in support)
        assert not ({"a", "a_dup"} <= selected_names), f"both duplicate twins survived: {selected_names}"


class TestConstantY:
    """A zero-variance target: H(y)=0 so every MI(X_j, y)=0 by construction. MRMR.fit's own input
    validation raises a clear, documented ValueError rather than crashing deeper in the MI/gate
    machinery or silently returning an empty/degenerate support_."""

    def test_constant_y_raises_clear_value_error(self):
        """A constant target must raise the documented 'only 1 unique value' ValueError at fit-time."""
        rng = np.random.default_rng(2)
        n = 200
        X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(4)})
        y = pd.Series(np.full(n, 7))
        m = MRMR(**_kw(min_features_fallback=2))
        with pytest.raises(ValueError, match="only 1 unique value"):
            m.fit(X, y)


class TestSampleWeightValidation:
    """The _maybe_resample_for_sample_weight guard: all-zero / negative / NaN / Inf weights must all
    raise the SAME documented ValueError, never silently poison the resampling probabilities."""

    def _helper(self):
        """A plain MRMR instance for calling _maybe_resample_for_sample_weight directly (no fit needed)."""
        return MRMR(**_kw())

    def test_all_zero_weights_raises(self):
        """All-zero weights must raise, not silently fall through to a uniform-weight no-op."""
        m = self._helper()
        X = np.zeros((10, 2))
        y = np.zeros(10)
        with pytest.raises(ValueError, match="sums to zero"):
            m._maybe_resample_for_sample_weight(X, y, np.zeros(10))

    @pytest.mark.parametrize(
        "bad_weight",
        [
            pytest.param(-0.001, id="single_negative"),
            pytest.param(np.inf, id="inf"),
            pytest.param(np.nan, id="nan"),
        ],
    )
    def test_bad_single_weight_raises_finite_nonneg_error(self, bad_weight):
        """A single negative/inf/nan weight must raise the same finite-and-non-negative ValueError,
        never silently poisoning every row's draw probability (the NaN case in particular)."""
        m = self._helper()
        X = np.zeros((10, 2))
        y = np.zeros(10)
        sw = np.ones(10)
        sw[0] = bad_weight
        with pytest.raises(ValueError, match="finite and non-negative"):
            m._maybe_resample_for_sample_weight(X, y, sw)

    def test_dominant_weight_concentrates_resampled_rows_reproducibly(self):
        """One massively dominant weight should draw a majority of rows from that row, and the
        resample must be reproducible across two independent calls with the same random_seed."""
        n = 1000
        X = pd.DataFrame({"a": np.arange(n, dtype=np.float64)})
        y = pd.Series(np.arange(n, dtype=np.float64))
        sw = np.ones(n)
        sw[0] = 1e6

        m1 = MRMR(**_kw())
        X1, y1 = m1._maybe_resample_for_sample_weight(X, y, sw)
        m2 = MRMR(**_kw())
        X2, y2 = m2._maybe_resample_for_sample_weight(X, y, sw)

        frac_row0 = float((X1["a"] == 0.0).mean())
        assert frac_row0 > 0.9, f"dominant weight should draw row 0 in >90% of samples, got {frac_row0:.3f}"
        pd.testing.assert_frame_equal(X1.reset_index(drop=True), X2.reset_index(drop=True))
        pd.testing.assert_series_equal(pd.Series(y1).reset_index(drop=True), pd.Series(y2).reset_index(drop=True))


class TestGroupsWithoutGroupAwareMi:
    """strict_groups=True (default) must RAISE when groups= is passed without group_aware_mi=True;
    strict_groups=False must instead warn and fall back to the legacy group-naive path."""

    def _dataset(self, n=200, seed=3):
        """A trivial classification frame plus a groups= array for the strict_groups tests."""
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
        y = pd.Series((X["a"] > 0).astype(int))
        groups = rng.integers(0, 5, n)
        return X, y, groups

    def test_strict_groups_default_raises(self):
        """groups= without group_aware_mi=True and the default strict_groups=True must raise, not warn."""
        X, y, groups = self._dataset()
        m = MRMR(**_kw(group_aware_mi=False))
        assert getattr(m, "strict_groups", True) is True
        with pytest.raises(NotImplementedError):
            m.fit(X, y, groups=groups)

    def test_strict_groups_false_warns_instead_of_raising(self):
        """strict_groups=False opts back into the legacy warn-only group-naive fallback."""
        X, y, groups = self._dataset(seed=4)
        m = MRMR(**_kw(group_aware_mi=False, strict_groups=False))
        with pytest.warns(UserWarning):
            m.fit(X, y, groups=groups)


class TestCloneBeforeAndAfterFit:
    """clone() must never leak fitted state, and must agree on get_params() regardless of fit status."""

    def test_clone_never_carries_fitted_attributes(self):
        """clone() before and after fit must both lack support_, and both must agree on get_params()
        with the pre-fit original -- clone never carries over fitted state or its lazy resolution."""
        from sklearn.base import clone

        rng = np.random.default_rng(5)
        n = 200
        X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
        y = pd.Series((X["a"] > 0).astype(int))

        m_before = MRMR(**_kw())
        clone_before = clone(m_before)
        assert not hasattr(clone_before, "support_")

        m_after = MRMR(**_kw())
        m_after.fit(X, y)
        assert hasattr(m_after, "support_")
        clone_after = clone(m_after)
        assert not hasattr(clone_after, "support_")

        assert clone_before.get_params() == clone_after.get_params() == m_before.get_params()
