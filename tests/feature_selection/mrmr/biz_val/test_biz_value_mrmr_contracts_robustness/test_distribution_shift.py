"""Consolidated from test_biz_value_mrmr_layer19.py.

Layer 19 biz_value MRMR contracts: TRAIN/TEST DISTRIBUTION-SHIFT.

WHY THIS LAYER
--------------
MRMR is a two-phase API: ``fit`` chooses the support_ at train time, and
``transform`` applies that *fixed* selection to every future row. Every
production deployment hits the same hazard: the X passed to transform()
in prod is NOT the X seen at fit time. Distribution drift, schema drift,
new categorical levels, NaN corruption, column re-ordering by an
upstream join -- all routinely cause ``support_`` to silently desync
from the data and produce mis-projected feature matrices for downstream
inference.

Layers 1-18 stressed signal vs noise, dtype handling, leakage and
degeneracy at FIT time. None of them probed the TRANSFORM-side contracts
that determine whether a fitted MRMR survives life in production. This
layer pins those contracts.

THE CORE INVARIANT
------------------
``support_`` is a FIT-TIME decision. ``transform`` must honour it by
NAME for named-column frames (pandas / polars) and by POSITION for
unnamed arrays. A shifted distribution in X at transform time must NOT
change which columns are returned -- only their values. The wrong
behaviour (positional fallback on a renamed DataFrame, silent intersect
of missing columns, garbage from a mis-aligned ndarray) is exactly the
prod-only bug class this layer surfaces.

SIX SHIFT SCENARIOS
-------------------
A. MEAN SHIFT      -- train ~N(0,1), test shifted by +2 across all features
B. VARIANCE SHIFT  -- train ~N(0,1), test has 2x variance (scale by ~1.41 std)
C. PARTIAL SHIFT   -- only x_signal_1 shifts; others stay clean
D. UNSEEN CATEGORY -- train region in {A,B,C}, test region in {B,C,D}
E. ALL-NaN TEST    -- extreme corruption (every numeric cell NaN)
F. REORDERED COLS  -- DataFrame (realign by name) and ndarray (raise on shape)

CONTRACTS PINNED
----------------
1. ``support_`` and ``feature_names_in_`` are STABLE across shifts; the
   fit-time choice does not depend on transform-time data.
2. Mean/variance/partial shift: transform succeeds, returns a frame with
   the FIT-TIME selected column names (DataFrame) or correct width
   (ndarray) -- no silent re-selection, no shape mutation.
3. Unseen categorical level: transform either returns a frame containing
   the new level un-encoded (MRMR is a selector, not an encoder) OR
   raises an actionable error. A silent crash with a numba/numpy
   traceback is the failure mode.
4. All-NaN test frame: transform completes (MRMR passes through NaN; the
   selected columns just happen to be all-NaN) -- it is downstream's
   job to impute, not the selector's.
5. DataFrame with shuffled column order: transform realigns by column
   NAME and produces the same values it would on the unshuffled frame.
   A positional bug here silently picks wrong columns.
6. ndarray with shape mismatch: raises ``ValueError`` referencing the
   expected feature count (sklearn-canonical ``_check_n_features``
   contract).
7. ndarray with SAME shape but reordered content: MRMR cannot detect
   the reorder (no column names) and applies positional selection --
   this is the contract for naked arrays. We pin the POSITIONAL
   behaviour explicitly so a future refactor that "helpfully" tries to
   re-detect column order doesn't silently break ndarray callers.
"""

from __future__ import annotations

import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TRAIN = 1_500
N_TEST = 800
SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_numeric_train(seed: int):
    """Numeric-only train frame with 2 real signals + 4 noise cols.

    y = 2.0 * x_signal_1 + 1.0 * x_signal_2 + 0.3 * noise. The signals
    carry ~88% of var(y), so any sane MRMR run selects at least one.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(N_TRAIN)
    x2 = rng.standard_normal(N_TRAIN)
    y = 2.0 * x1 + 1.0 * x2 + 0.3 * rng.standard_normal(N_TRAIN)
    cols = {"x_signal_1": x1, "x_signal_2": x2}
    for k in range(4):
        cols[f"noise_{k}"] = rng.standard_normal(N_TRAIN)
    X = pd.DataFrame(cols)
    return X, pd.Series(y, name="y_reg")


def _build_numeric_test(seed: int, n: int = N_TEST):
    """Clean (un-shifted) test frame matching _build_numeric_train schema."""
    rng = np.random.default_rng(seed + 10_000)
    cols = {
        "x_signal_1": rng.standard_normal(n),
        "x_signal_2": rng.standard_normal(n),
    }
    for k in range(4):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    return pd.DataFrame(cols)


def _build_cat_train(seed: int):
    """Train frame with a categorical column carrying the bulk of signal.

    region is the dominant predictor -- a 3-level cat with effect
    coefficients (-3, 0, +3). Ensures MRMR will SELECT the cat column,
    so the unseen-level transform path actually exercises a categorical
    branch (not the trivial "cat not selected" no-op).
    """
    rng = np.random.default_rng(seed)
    region = rng.choice(["A", "B", "C"], size=N_TRAIN)
    region_effect = {"A": -3.0, "B": 0.0, "C": 3.0}
    y_arr = np.array([region_effect[r] for r in region]) + 0.3 * rng.standard_normal(N_TRAIN)
    X = pd.DataFrame(
        {
            "noise_a": rng.standard_normal(N_TRAIN),
            "region": pd.Categorical(region, categories=["A", "B", "C"]),
            "noise_b": rng.standard_normal(N_TRAIN),
        }
    )
    return X, pd.Series(y_arr, name="y_reg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mrmr(**overrides):
    """Default-config MRMR with wall-time pins; Wave 9 production surface."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    kwargs = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _fit_quiet(sel, X, y):
    """Fit ``sel`` on ``(X, y)`` with warnings silenced; return the fitted estimator."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sel.fit(X, y)


def _transform_quiet(sel, X):
    """Transform ``X`` through ``sel`` with warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sel.transform(X)


@cache
def _numeric_train_fit(seed: int):
    """Cached ``(X_train, y_train, sel)`` for the default-config fit on the
    clean numeric train frame, shared across 8 tests (support-stability,
    mean/variance/partial shift, all-NaN, reordered DataFrame, combined
    shift). Nothing downstream mutates X_train/y_train/sel in place --
    every consumer only calls transform() on a freshly-built X_test or
    inspects sel's fit-time attributes.
    """
    X_train, y_train = _build_numeric_train(seed)
    sel = _make_mrmr(random_seed=seed)
    _fit_quiet(sel, X_train, y_train)
    return X_train, y_train, sel


@cache
def _cat_train_fit(seed: int):
    """Cached ``(X_train, y_train, sel)`` for the default-config fit on the
    categorical train frame, shared across TestUnseenCategoricalLevel's 2
    tests. Nothing downstream mutates X_train/y_train/sel in place.
    """
    X_train, y_train = _build_cat_train(seed)
    sel = _make_mrmr(random_seed=seed)
    _fit_quiet(sel, X_train, y_train)
    return X_train, y_train, sel


@cache
def _numeric_train_ndarray_fit(seed: int):
    """Cached ``(X_arr, y_train, sel)`` for the default-config fit on the
    naked-ndarray view of the numeric train frame, shared across
    TestNdarrayShapeMismatch's 2 tests and
    test_reordered_ndarray_is_positional. Nothing downstream mutates
    X_arr/y_train/sel in place.
    """
    X_train, y_train = _build_numeric_train(seed)
    X_arr = X_train.to_numpy()
    sel = _make_mrmr(random_seed=seed)
    _fit_quiet(sel, X_arr, np.asarray(y_train))
    return X_arr, y_train, sel


# ---------------------------------------------------------------------------
# Contract 1: support_ stable across distribution shifts at transform time
# ---------------------------------------------------------------------------


class TestSupportIsFitTimeDecision:
    """``support_`` and ``feature_names_in_`` snapshot the fit-time
    column set; calling transform on shifted data must NOT mutate them.
    A side-effect here would break any audit script that introspects
    the selector AFTER inference (the canonical prod usage).
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_support_unchanged_after_transform(self, seed):
        """transform() must not mutate support_/feature_names_in_/n_features_in_."""
        _X_train, _y_train, sel = _numeric_train_fit(seed)
        support_before = np.array(sel.support_, copy=True)
        names_before = list(sel.feature_names_in_)
        n_in_before = sel.n_features_in_

        X_test = _build_numeric_test(seed) + 2.0  # mean shift
        _transform_quiet(sel, X_test)

        assert np.array_equal(np.array(sel.support_), support_before), (
            f"support_ mutated by transform on shifted data; seed={seed}. fit-time decision must be frozen across transform calls."
        )
        assert list(sel.feature_names_in_) == names_before, f"feature_names_in_ mutated by transform on shifted data; seed={seed}."
        assert sel.n_features_in_ == n_in_before, f"n_features_in_ mutated by transform on shifted data; seed={seed}."


# ---------------------------------------------------------------------------
# Contract 2: A - mean shift -> transform succeeds, correct columns
# ---------------------------------------------------------------------------


class TestMeanShiftTransform:
    """Train ~N(0,1), test ~N(+2,1). Transform must succeed and return
    a frame whose column NAMES match the fit-time selection -- the
    values in those columns are whatever the shifted X carried.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mean_shift_returns_selected_columns(self, seed):
        """Mean-shifted test frame still transforms to the fit-time selected columns."""
        _X_train, _y_train, sel = _numeric_train_fit(seed)
        selected = list(sel.get_feature_names_out())
        assert len(selected) >= 1, f"empty support_ on clean train; seed={seed}"

        X_test = _build_numeric_test(seed) + 2.0
        out = _transform_quiet(sel, X_test)
        assert isinstance(out, pd.DataFrame), f"transform should return a DataFrame for DataFrame input; seed={seed}, got {type(out).__name__}"
        assert list(out.columns) == selected, (
            f"mean-shift transform returned different columns than get_feature_names_out; seed={seed}. got={list(out.columns)}, expected={selected}"
        )
        assert out.shape[0] == X_test.shape[0], f"transform row count {out.shape[0]} != input {X_test.shape[0]}; seed={seed}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mean_shift_values_match_input_columns(self, seed):
        """The values in transform's output ARE the values in X_test at
        the selected columns -- no silent transformation."""
        _X_train, _y_train, sel = _numeric_train_fit(seed)
        selected = list(sel.get_feature_names_out())
        X_test = _build_numeric_test(seed) + 2.0
        out = _transform_quiet(sel, X_test)
        for col in selected:
            if col not in X_test.columns:
                # engineered recipe column, skip value comparison
                continue
            assert np.array_equal(out[col].to_numpy(), X_test[col].to_numpy()), (
                f"mean-shift transform mutated values of column {col!r}; seed={seed}. transform should be a pure column selector."
            )


# ---------------------------------------------------------------------------
# Contract 3: B - variance shift -> transform succeeds
# ---------------------------------------------------------------------------


class TestVarianceShiftTransform:
    """Test variance >> train variance. MRMR's selection is fit-time,
    so the variance shift must NOT change which columns come out.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_variance_shift_returns_correct_columns(self, seed):
        """Variance-shifted test frame still transforms to the fit-time selected columns."""
        _X_train, _y_train, sel = _numeric_train_fit(seed)
        selected = list(sel.get_feature_names_out())
        X_test = _build_numeric_test(seed) * 2.0  # 2x std => 4x variance
        out = _transform_quiet(sel, X_test)
        assert list(out.columns) == selected, f"variance-shift transform changed column set; seed={seed}. got={list(out.columns)}, expected={selected}"


# ---------------------------------------------------------------------------
# Contract 4: C - partial shift -> selection is still fit-time
# ---------------------------------------------------------------------------


class TestPartialShiftTransform:
    """Only x_signal_1 shifts in test; the rest are clean. This is the
    realistic prod scenario (one column's source drifts, others stay
    pinned). Selection must NOT depend on the shifted column's
    transform-time distribution.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_partial_shift_preserves_fit_time_selection(self, seed):
        """A single-column drift at transform time must not change the selected column set."""
        _X_train, _y_train, sel = _numeric_train_fit(seed)
        selected_before = list(sel.get_feature_names_out())

        X_test = _build_numeric_test(seed)
        X_test["x_signal_1"] = X_test["x_signal_1"] + 5.0  # only this drifts
        out = _transform_quiet(sel, X_test)

        assert list(out.columns) == selected_before, f"partial-shift transform changed column set; seed={seed}. selection must be frozen at fit time."


# ---------------------------------------------------------------------------
# Contract 5: D - unseen categorical level handled predictably
# ---------------------------------------------------------------------------


class TestUnseenCategoricalLevel:
    """Train has region in {A,B,C}; test has region in {B,C,D}. MRMR is
    a SELECTOR (not an encoder) so the canonical behaviour is to
    pass the column through to transform output unchanged -- D
    appears in the returned column as a level the downstream encoder /
    model must handle. A silent traceback from a numba kernel or a
    KeyError on the unseen level would be a real bug.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_unseen_level_does_not_crash(self, seed):
        """A test-time-only categorical level either passes through or raises an actionable error."""
        _X_train, _y_train, sel = _cat_train_fit(seed)
        selected = list(sel.get_feature_names_out())

        rng = np.random.default_rng(seed + 33_333)
        region_test = rng.choice(["B", "C", "D"], size=N_TEST)
        X_test = pd.DataFrame(
            {
                "noise_a": rng.standard_normal(N_TEST),
                "region": pd.Categorical(region_test, categories=["A", "B", "C", "D"]),
                "noise_b": rng.standard_normal(N_TEST),
            }
        )
        # Acceptable outcomes: (a) transform succeeds and returns a frame
        # whose columns match the fit-time selection, OR (b) raises a
        # clean ValueError / RuntimeError naming the offending level.
        # Anything else (numba/numpy traceback, silent shape mutation)
        # is a contract violation.
        try:
            out = _transform_quiet(sel, X_test)
        except (ValueError, RuntimeError):
            return  # acceptable actionable error
        except Exception as e:
            pytest.fail(
                f"unseen-level transform raised non-actionable "
                f"{type(e).__name__}: {e!r}. seed={seed}. "
                f"Expected either success (selector passes column "
                f"through) or ValueError / RuntimeError naming the "
                f"unseen level."
            )
        assert list(out.columns) == selected, f"unseen-level transform changed column set; seed={seed}. got={list(out.columns)}, expected={selected}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_unseen_level_passes_through_when_cat_selected(self, seed):
        """When ``region`` is in support_, the new level ``D`` must
        appear in the returned column -- the selector does NOT
        silently drop / re-encode rows.
        """
        _X_train, _y_train, sel = _cat_train_fit(seed)
        selected = list(sel.get_feature_names_out())
        if "region" not in selected:
            return  # cat not selected; passthrough contract trivially holds

        rng = np.random.default_rng(seed + 33_333)
        region_test = rng.choice(["B", "C", "D"], size=N_TEST)
        X_test = pd.DataFrame(
            {
                "noise_a": rng.standard_normal(N_TEST),
                "region": pd.Categorical(region_test, categories=["A", "B", "C", "D"]),
                "noise_b": rng.standard_normal(N_TEST),
            }
        )
        try:
            out = _transform_quiet(sel, X_test)
        except (ValueError, RuntimeError):
            return  # accepted alternative
        assert "region" in out.columns, f"region selected at fit but dropped at transform; seed={seed}"
        # Row count must be preserved (no silent filtering of unseen levels)
        assert out.shape[0] == X_test.shape[0], f"transform dropped rows containing unseen level D; seed={seed}. out={out.shape[0]}, in={X_test.shape[0]}"


# ---------------------------------------------------------------------------
# Contract 6: E - all-NaN test frame -> transform succeeds
# ---------------------------------------------------------------------------


class TestAllNaNTestTransform:
    """Worst-case corruption: every cell of X_test is NaN. MRMR is a
    selector, NaN passes through, downstream NaN-aware models
    (catboost / lightgbm / xgboost-hist) handle the missingness. The
    selector must NOT crash on the corrupted frame.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_all_nan_test_does_not_crash(self, seed):
        """An all-NaN test frame transforms without crashing; selector passes NaN through."""
        _X_train, _y_train, sel = _numeric_train_fit(seed)
        selected = list(sel.get_feature_names_out())

        # Build NaN test frame with the SAME columns as train -- column
        # set drift is covered by RuntimeError in transform()'s named-frame
        # path and is NOT what we're probing here.
        X_test = _build_numeric_test(seed).astype(float)
        X_test[:] = np.nan
        out = _transform_quiet(sel, X_test)
        assert list(out.columns) == selected, f"all-NaN transform returned different columns; seed={seed}"
        assert out.shape[0] == X_test.shape[0]
        # Selector passes through; output must still be all-NaN at the
        # selected columns.
        for col in selected:
            if col not in X_test.columns:
                continue  # engineered recipe -- may not be NaN
            assert out[col].isna().all(), f"selector mutated NaN values at col {col!r}; seed={seed}"


# ---------------------------------------------------------------------------
# Contract 7: F1 - DataFrame column reorder -> realign by name
# ---------------------------------------------------------------------------


class TestDataFrameColumnReorderRealignByName:
    """Upstream join / Parquet reader sometimes emits columns in a
    different order than fit-time. For a named-column DataFrame, MRMR
    must realign by NAME and produce the same values as if the
    DataFrame had been un-shuffled. A positional bug here would
    silently project the wrong columns.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_reordered_dataframe_returns_correct_values(self, seed):
        """A column-reordered DataFrame realigns by name and produces identical values."""
        _X_train, _y_train, sel = _numeric_train_fit(seed)
        selected = list(sel.get_feature_names_out())

        X_test = _build_numeric_test(seed)
        # Hard reorder: reverse the columns
        X_test_rev = X_test[list(reversed(X_test.columns))]
        out_orig = _transform_quiet(sel, X_test)
        out_rev = _transform_quiet(sel, X_test_rev)

        assert list(out_orig.columns) == selected
        assert list(out_rev.columns) == selected, (
            f"reordered DataFrame transform produced different column set than original; seed={seed}. reorder={list(out_rev.columns)}, orig={selected}"
        )
        for col in selected:
            if col not in X_test.columns:
                continue  # engineered
            assert np.array_equal(out_orig[col].to_numpy(), out_rev[col].to_numpy()), (
                f"reordered DataFrame produced different values at col {col!r} -- positional selection bug. seed={seed}"
            )


# ---------------------------------------------------------------------------
# Contract 8: F2 - ndarray shape mismatch -> ValueError
# ---------------------------------------------------------------------------


class TestNdarrayShapeMismatch:
    """sklearn-canonical: transform on a naked ndarray with the wrong
    number of columns must raise ``ValueError``. Pre-Wave-9.1 the
    positional path silently sliced support_ from whatever columns
    happened to land at those positions, returning garbage. This is
    the regression-pin for that fix.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_fewer_cols_raises_valueerror(self, seed):
        """A narrower ndarray at transform time raises ValueError naming the expected feature count."""
        X_arr, _y_train, sel = _numeric_train_ndarray_fit(seed)
        # Drop the last column
        X_test_bad = X_arr[:, :-1]
        with pytest.raises(ValueError) as exc_info:
            _transform_quiet(sel, X_test_bad)
        msg = str(exc_info.value).lower()
        assert "features" in msg, f"ValueError message missing 'features'; seed={seed}, got: {exc_info.value!r}"
        assert str(X_arr.shape[1]) in str(exc_info.value), (
            f"ValueError message should mention expected feature count ({X_arr.shape[1]}); seed={seed}, got: {exc_info.value!r}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_more_cols_raises_valueerror(self, seed):
        """A wider ndarray at transform time raises ValueError."""
        X_arr, _y_train, sel = _numeric_train_ndarray_fit(seed)
        # Add a junk column
        X_test_bad = np.column_stack([X_arr, np.zeros(X_arr.shape[0])])
        with pytest.raises(ValueError):
            _transform_quiet(sel, X_test_bad)


# ---------------------------------------------------------------------------
# Contract 9: F3 - ndarray same shape, reordered content -> positional
# ---------------------------------------------------------------------------


class TestNdarraySameShapeReorderedIsPositional:
    """A naked ndarray has no column-name information, so MRMR CANNOT
    detect content reorder (sklearn _check_n_features only validates
    shape). transform() returns columns at the FIT-TIME POSITIONS of
    support_, regardless of what got shuffled in.

    We pin this positional behaviour explicitly so a future "helpful"
    refactor that tries to re-detect column order doesn't silently
    break naked-ndarray callers who already pass columns in the right
    order.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_reordered_ndarray_is_positional(self, seed):
        """A naked ndarray with reordered content is transformed by fit-time POSITION, not name."""
        X_arr, _y_train, sel = _numeric_train_ndarray_fit(seed)
        support = np.asarray(sel.support_)
        if support.dtype == bool:
            support_idx = np.flatnonzero(support)
        else:
            support_idx = support.astype(np.intp)

        # Use a structurally identical test array (no shift), reorder cols.
        rng = np.random.default_rng(seed + 99_999)
        X_test = rng.standard_normal(X_arr.shape).astype(X_arr.dtype)
        # Reverse column order
        X_test_rev = X_test[:, ::-1]

        out_orig = _transform_quiet(sel, X_test)
        out_rev = _transform_quiet(sel, X_test_rev)
        assert out_orig.shape == out_rev.shape, f"transform shape changed under content reorder; seed={seed}"
        # Positional contract: out_orig[:, k] == X_test[:, support_idx[k]],
        # out_rev[:, k] == X_test_rev[:, support_idx[k]]
        for k, j in enumerate(support_idx):
            assert np.array_equal(out_orig[:, k], X_test[:, j]), f"unreordered ndarray transform did not select position {j} at output slot {k}; seed={seed}"
            assert np.array_equal(out_rev[:, k], X_test_rev[:, j]), (
                f"reordered ndarray transform did not select position {j} at output slot {k} -- positional contract broken. seed={seed}"
            )


# ---------------------------------------------------------------------------
# Contract 10: end-to-end -- fit, shift everything, transform succeeds
# ---------------------------------------------------------------------------


class TestEndToEndShiftedTransform:
    """Smoke-level e2e: fit on clean train, apply A+B+C simultaneously
    (mean + variance + per-column drift), transform must still produce
    a frame with the fit-time selected columns and right shape.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_combined_shifts_transform_succeeds(self, seed):
        """Mean + variance + per-column drift + NaN sprinkle combined: transform still succeeds."""
        _X_train, _y_train, sel = _numeric_train_fit(seed)
        selected = list(sel.get_feature_names_out())

        X_test = _build_numeric_test(seed)
        # Mean shift
        X_test = X_test + 1.5
        # Variance shift on signal_2
        X_test["x_signal_2"] = X_test["x_signal_2"] * 2.0
        # Sprinkle some NaNs into a noise column (10%)
        rng = np.random.default_rng(seed + 7777)
        nan_mask = rng.random(X_test.shape[0]) < 0.1
        X_test.loc[nan_mask, "noise_0"] = np.nan

        out = _transform_quiet(sel, X_test)
        assert list(out.columns) == selected
        assert out.shape == (X_test.shape[0], len(selected))
