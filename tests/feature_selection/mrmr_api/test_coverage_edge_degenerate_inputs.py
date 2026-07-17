"""Edge-case / degenerate / adversarial coverage for ``MRMR.fit``.

These exercise the nasty inputs a real user eventually feeds the selector and pin
the GRACEFUL contract MRMR already provides: either a clear ``ValueError`` at the
validation boundary, or a sensible, deterministic ``support_`` with no NaN / garbage
feature code leaking into the selection. Each test asserts the OBSERVED defensible
behaviour (verified against the current code), so a future regression that turns a
clean raise into a deep crash -- or that starts emitting garbage on a pathological
column -- trips here.

Scope is deliberately the column-/target-side degeneracies that the existing
``test_biz_value_mrmr_hard_cases.py`` (signal under collinearity, XOR synergy, rare
1% class) does NOT cover: all-NaN / all-inf / mixed columns, constant / zero-variance
columns, single-class & constant targets, tiny n, float-id high cardinality, exact
duplicates & perfect collinearity, informative missingness, very wide p>n, mixed
dtypes, and the all-noise empty-after-filtering fallback.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _q():
    """Silence the (expected, voluminous) MRMR warnings on degenerate frames."""
    return warnings.catch_warnings()


def _no_garbage_names(names) -> bool:
    """No selected feature name is empty / NaN-coded / None."""
    for n in names:
        if n is None:
            return False
        s = str(n)
        if s == "" or s.lower() == "nan":
            return False
    return True


# ---------------------------------------------------------------------------
# Non-finite columns
# ---------------------------------------------------------------------------
class TestNonFiniteColumns:
    """Groups tests covering TestNonFiniteColumns."""
    def test_all_nan_column_is_dropped_and_diagnosed(self):
        """An all-NaN column must not crash, must NOT be selected, and must be
        recorded in ``degenerate_columns_`` as ``all_nan``."""
        rng = np.random.default_rng(0)
        n = 300
        a = rng.normal(size=n)
        X = pd.DataFrame({"a": a, "allnan": np.full(n, np.nan)})
        y = pd.Series((a > 0).astype(int))
        with _q():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "allnan" not in names, f"all-NaN column was selected: {names}"
        assert _no_garbage_names(names)
        assert sel.degenerate_columns_.get("allnan") == "all_nan"

    def test_all_inf_column_raises_clear_valueerror(self):
        """An all-+inf column must raise the explicit inf-guard ValueError, not
        crash deep in discretization."""
        rng = np.random.default_rng(0)
        n = 300
        a = rng.normal(size=n)
        X = pd.DataFrame({"a": a, "inf": np.full(n, np.inf)})
        y = pd.Series((a > 0).astype(int))
        with pytest.raises(ValueError, match="inf"):
            MRMR(verbose=0, fe_max_steps=0).fit(X, y)

    def test_mixed_nan_inf_raises_on_the_inf(self):
        """A column carrying both NaN and inf must be rejected for the inf
        (NaN alone is allowed, inf never is)."""
        rng = np.random.default_rng(0)
        n = 300
        a = rng.normal(size=n)
        a[0] = np.nan
        a[1] = np.inf
        X = pd.DataFrame({"a": a, "b": rng.normal(size=n)})
        y = pd.Series((X["b"] > 0).astype(int))
        with pytest.raises(ValueError, match="inf"):
            MRMR(verbose=0, fe_max_steps=0).fit(X, y)

    def test_partial_nan_column_is_tolerated(self):
        """A column with SOME (not all) NaN is allowed -- routed through
        nan_strategy. No crash, deterministic support, no garbage."""
        rng = np.random.default_rng(3)
        n = 500
        a = rng.normal(size=n)
        a[rng.random(n) < 0.3] = np.nan
        X = pd.DataFrame({"a": a, "b": rng.normal(size=n)})
        y = pd.Series((rng.normal(size=n) > 0).astype(int))
        with _q():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, fe_max_steps=0).fit(X, y)
        assert _no_garbage_names(list(sel.get_feature_names_out()))


# ---------------------------------------------------------------------------
# Constant / zero-variance columns and targets
# ---------------------------------------------------------------------------
class TestConstantInputs:
    """Groups tests covering TestConstantInputs."""
    def test_constant_column_dropped_and_diagnosed(self):
        """Constant column dropped and diagnosed."""
        rng = np.random.default_rng(1)
        n = 300
        a = rng.normal(size=n)
        X = pd.DataFrame({"a": a, "const": np.ones(n)})
        y = pd.Series((a > 0).astype(int))
        with _q():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "const" not in names, f"constant column selected: {names}"
        assert sel.degenerate_columns_.get("const") == "constant"

    def test_single_class_y_raises(self):
        """Classification with one y value -> H(y)=0 -> clear ValueError."""
        rng = np.random.default_rng(0)
        n = 300
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        y = pd.Series(np.ones(n, dtype=int))
        with pytest.raises(ValueError, match="1 unique value"):
            MRMR(verbose=0).fit(X, y)

    def test_constant_y_regression_raises(self):
        """Constant y regression raises."""
        rng = np.random.default_rng(0)
        n = 300
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        y = pd.Series(np.full(n, 3.14))
        with pytest.raises(ValueError, match="1 unique value"):
            MRMR(verbose=0).fit(X, y)


# ---------------------------------------------------------------------------
# Tiny n
# ---------------------------------------------------------------------------
class TestTinyN:
    """Groups tests covering TestTinyN."""
    def test_n_zero_raises(self):
        """N zero raises."""
        X = pd.DataFrame({"a": pd.Series([], dtype=float)})
        y = pd.Series([], dtype=int)
        with pytest.raises(ValueError, match="empty input"):
            MRMR(verbose=0).fit(X, y)

    def test_n_one_raises(self):
        """N one raises."""
        X = pd.DataFrame({"a": [1.0], "b": [3.0]})
        y = pd.Series([0])
        with pytest.raises(ValueError, match="single row"):
            MRMR(verbose=0).fit(X, y)

    def test_n_two_raises_min_rows_floor(self):
        """n=2 is below the 10-row factor floor -> clear ValueError, not a crash."""
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        y = pd.Series([0, 1])
        with pytest.raises(ValueError, match="at least 10 rows"):
            MRMR(verbose=0).fit(X, y)

    def test_n_below_bins_floor_raises(self):
        """n=8 (< 10-row floor, < default bins) -> clear ValueError."""
        rng = np.random.default_rng(0)
        n = 8
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        y = pd.Series((X["a"] > 0).astype(int))
        with pytest.raises(ValueError, match="at least 10 rows"):
            MRMR(verbose=0, fe_max_steps=0).fit(X, y)


# ---------------------------------------------------------------------------
# Cardinality extremes
# ---------------------------------------------------------------------------
class TestCardinality:
    """Groups tests covering TestCardinality."""
    def test_float_id_column_does_not_crash(self):
        """A near-unique float 'id' column (n_unique ~ n) must not crash binning;
        it should NOT outrank the genuine signal."""
        rng = np.random.default_rng(0)
        n = 400
        a = rng.normal(size=n)
        ident = np.arange(n, dtype=float) + rng.normal(scale=1e-6, size=n)
        X = pd.DataFrame({"id": ident, "a": a})
        y = pd.Series((a > 0).astype(int))
        with _q():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert _no_garbage_names(names)
        assert any("a" == n0 or n0.startswith("a") for n0 in names), f"genuine signal 'a' lost behind float-id column: {names}"


# ---------------------------------------------------------------------------
# Duplicates / perfect collinearity
# ---------------------------------------------------------------------------
class TestDuplicatesCollinearity:
    """Groups tests covering TestDuplicatesCollinearity."""
    def test_exact_duplicate_column_diagnosed_and_not_both_selected(self):
        """Exact duplicate column diagnosed and not both selected."""
        rng = np.random.default_rng(0)
        n = 400
        a = rng.normal(size=n)
        X = pd.DataFrame({"a": a, "a_dup": a.copy(), "b": rng.normal(size=n)})
        y = pd.Series((a > 0).astype(int))
        with _q():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert sel.degenerate_columns_.get("a_dup") == "duplicate_of:a"
        assert not ("a" in names and "a_dup" in names), f"both exact-duplicate columns selected: {names}"

    def test_perfect_collinear_column_diagnosed(self):
        """y = 2*a + 3 is a perfect linear dependence -> ``collinear_with:a``."""
        rng = np.random.default_rng(0)
        n = 400
        a = rng.normal(size=n)
        X = pd.DataFrame({"a": a, "lin": 2.0 * a + 3.0, "b": rng.normal(size=n)})
        y = pd.Series((a > 0).astype(int))
        with _q():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, fe_max_steps=0).fit(X, y)
        assert sel.degenerate_columns_.get("lin") == "collinear_with:a"


# ---------------------------------------------------------------------------
# Informative missingness / all-noise fallback / wide / mixed dtypes
# ---------------------------------------------------------------------------
class TestMiscDegenerate:
    """Groups tests covering TestMiscDegenerate."""
    def test_informative_missingness_recovers_signal(self):
        """When the MISSINGNESS pattern (not the values) drives y, MRMR must still
        select the column whose NaN pattern carries the signal."""
        rng = np.random.default_rng(5)
        n = 700
        a = rng.normal(size=n)
        miss = rng.random(n) < 0.4
        a[miss] = np.nan
        X = pd.DataFrame({"a": a, "b": rng.normal(size=n)})
        y = pd.Series(miss.astype(int))
        with _q():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "a" in names, f"informative-missingness signal lost: {names}"

    def test_all_noise_falls_back_not_garbage(self):
        """y independent of every column -> screening returns 0 -> the documented
        min_features_fallback keeps ONE real raw feature, never garbage / empty."""
        rng = np.random.default_rng(7)
        n = 500
        X = pd.DataFrame({f"n{i}": rng.normal(size=n) for i in range(6)})
        y = pd.Series(rng.integers(0, 2, n))
        with _q():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert _no_garbage_names(names)
        assert all(n0 in X.columns for n0 in names), f"fallback returned a non-raw / garbage name: {names}"
        assert getattr(sel, "fallback_used_", False) is True

    def test_wide_p_greater_than_n_does_not_crash(self):
        """Wide p greater than n does not crash."""
        rng = np.random.default_rng(0)
        n, p = 40, 120
        data = {f"x{i}": rng.normal(size=n) for i in range(p)}
        sig = data["x0"]
        X = pd.DataFrame(data)
        y = pd.Series((sig > 0).astype(int))
        with _q():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert _no_garbage_names(names)
        assert len(names) >= 1

    def test_mixed_dtypes_does_not_crash(self):
        """int / float / bool / string / categorical columns together must fit
        without dtype crashes and select the float signal."""
        rng = np.random.default_rng(0)
        n = 400
        f = rng.normal(size=n)
        X = pd.DataFrame(
            {
                "f": f,
                "i": rng.integers(0, 5, n),
                "bl": rng.integers(0, 2, n).astype(bool),
                "s": rng.choice(["x", "y", "z"], n),
                "c": pd.Categorical(rng.choice(["p", "q"], n)),
            }
        )
        y = pd.Series((f > 0).astype(int))
        with _q():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert _no_garbage_names(names)
        assert any("f" == n0 or n0.startswith("f") for n0 in names), f"float signal lost in mixed-dtype frame: {names}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    """Groups tests covering TestDeterminism."""
    def test_same_seed_same_support_on_degenerate_frame(self):
        """Two fits with the same seed on a frame full of degeneracies (NaN col,
        constant col, duplicate col) must produce IDENTICAL support."""
        rng = np.random.default_rng(11)
        n = 400
        a = rng.normal(size=n)
        X = pd.DataFrame(
            {
                "a": a,
                "a_dup": a.copy(),
                "const": np.zeros(n),
                "allnan": np.full(n, np.nan),
                "b": rng.normal(size=n),
            }
        )
        y = pd.Series((a > 0).astype(int))
        with _q():
            warnings.simplefilter("ignore")
            s1 = MRMR(verbose=0, random_seed=7, fe_max_steps=0).fit(X, y)
            s2 = MRMR(verbose=0, random_seed=7, fe_max_steps=0).fit(X, y)
        assert list(s1.get_feature_names_out()) == list(s2.get_feature_names_out())
