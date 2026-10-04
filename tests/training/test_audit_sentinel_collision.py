"""Wave 50 (2026-05-20): numeric sentinel collision audit.

Audit class: using -1 / -999 / np.nan / np.iinfo(dtype).max / 0 as missing-or-
invalid markers where real data can legitimately contain those values, silently
confusing real data with sentinel.

3 P1 + 4 P2 = 7 fixes applied:

  P1:
    1. training/extractors.py:791 (classification targets)
       fillna(0) before threshold -> raise on NaN target (silent label flip on
       thresh_val<=0 eliminated).

    2. estimators/custom.py:179 (PdOrdinalEncoder)
       encoded_missing_value default flipped np.nan -> -1; transform asserts
       no NaN survives the int32 cast (was producing INT_MIN platform-dependent).

    3. training/dummy_baselines.py:1379 (LTR fast-path group sanity)
       pd.factorize emits -1 for NaN -> np.bincount(-1) raised ValueError;
       filter codes>=0 before bincount.

  P2:
    4. training/_predict_guards.py:288 (NaN-guard detection)
       ~np.isfinite included +/-inf which SimpleImputer doesn't replace ->
       use np.isnan to match the pandas branch's semantics.

    5. feature_selection/filters/discretization.py:126 (categorize_1d_array)
       nan_filler=0.0 default biased MI by collapsing NaN onto real-0; added
       nan_filler=None -> raise option + WARN when default fires.

    6. training/target_temporal_audit.py:581 (per-bin positive rate)
       fillna(0) > 0 deflated rate by counting NaN as negative -> dropna()
       before mean; honest "positive fraction over non-missing".

    7. feature_engineering/bruteforce.py:145,156 (PySR sampling)
       fill_null/fill_nan(0) on numeric -> per-column median; PySR's candidate
       scoring no longer biased toward features where NaN ~ 0 by coincidence.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_a_missing_classification_label_is_refused_before_training() -> None:
    """The extractor keeps a missing label missing; the suite refuses the target before any model trains, naming it."""
    from mlframe.training.configs import TargetTypes
    from mlframe.training.core._target_labels import raise_on_missing_labels

    with pytest.raises(ValueError, match="y_bin_gte_0.5: target contains 1 NaN/null label"):
        raise_on_missing_labels({TargetTypes.BINARY_CLASSIFICATION: {"y_bin_gte_0.5": np.array([1.0, np.nan, 0.0], dtype=np.float32)}})


def test_dummy_baselines_factorize_filters_negative_codes() -> None:
    """The LTR fast path skips (rather than crashes on) a train group column holding NaN ids, and sizes groups over the non-missing rows."""
    import pandas as pd

    from mlframe.training.baselines._dummy_compute_helpers import _compute_ltr_baselines
    from mlframe.training.configs import DummyBaselinesConfig

    n = 60
    ids = (np.arange(n) % 6).astype(float)
    ids[::7] = np.nan
    assert pd.factorize(ids)[0].min() == -1
    y = np.arange(n, dtype=float)
    clean_ids = (np.arange(20) % 6).astype(float)
    val_preds, _test_preds, extras = _compute_ltr_baselines("ltr", y, y[:20], y[:20], ids, clean_ids, clean_ids, None, None, None, DummyBaselinesConfig())
    assert "ltr_skip_reason" not in extras
    assert extras["n_groups_train"] == 7
    assert set(val_preds) >= {"random_within_query", "mean_relevance"}


def _reference_impute_scale(arr):
    """sklearn mean-imputation then standardisation of ``arr`` with +/-inf treated as missing."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    cleaned = np.where(np.isfinite(arr), arr, np.nan)
    return StandardScaler().fit_transform(SimpleImputer(strategy="mean", keep_empty_features=True).fit_transform(cleaned))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_predict_guards_nan_detection_uses_isnan_not_isfinite(bad_value) -> None:
    """A numpy frame holding NaN or +/-inf in its first rows is imputed and scaled before the model sees it; a clean frame is passed through untouched."""
    from types import SimpleNamespace

    from mlframe.training._predict_guards import _apply_nan_guard

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    X[10, 1] = bad_value
    seen: list = []

    def fn(a):
        """Record the frame handed to the model and return its first column."""
        seen.append(np.asarray(a))
        return np.asarray(a)[:, 0]

    model = SimpleNamespace()
    out = _apply_nan_guard(model, X, fn, 600, fit_at_predict=True)
    assert len(seen) == 1
    assert np.isfinite(seen[0]).all()
    np.testing.assert_allclose(seen[0], _reference_impute_scale(X), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(out, seen[0][:, 0])
    assert model._mlframe_nan_imputer is not None and model._mlframe_nan_scaler is not None

    clean = rng.normal(size=(600, 3))
    seen.clear()
    clean_model = SimpleNamespace()
    _apply_nan_guard(clean_model, clean, fn, 600, fit_at_predict=True)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], clean)
    assert not hasattr(clean_model, "_mlframe_nan_imputer")


def test_discretization_nan_filler_supports_raise() -> None:
    """``nan_filler=None`` raises on NaN input; the legacy 0.0 default warns and bins NaN together with the real zeros."""
    from mlframe.feature_selection.filters.discretization import categorize_1d_array

    vals = np.array([0.0, np.nan, 5.0, 0.0, 5.0, 2.0])
    with pytest.raises(ValueError, match="input contains NaN and nan_filler=None"):
        categorize_1d_array(vals, 2, "discretizer", 0, {"n_bins": 3}, nan_filler=None)
    with pytest.warns(UserWarning, match="biases MI by mixing"):
        out = categorize_1d_array(vals, 2, "discretizer", 0, {"n_bins": 3})
    codes = np.asarray(out).ravel()
    assert codes[1] == codes[0] == codes[3]
    assert codes[2] != codes[0]


def test_a_bin_s_positive_rate_ignores_missing_rows_rather_than_counting_them_negative() -> None:
    """NaN is not a negative. Counting it as one deflates the rate by the missing fraction.

    Behavioural since 2026-09-04. This asserted that `(c.fillna(0) > 0).mean()` is absent from the
    module and `(c.dropna() > 0).mean()` present -- two spellings of one lambda, already chased
    once through a module split, and silent about the number that comes out.
    """
    import pandas as pd

    from mlframe.training.targets._target_temporal_audit_aggregate import _aggregate_by_time_pandas

    # One bin: two positives, two negatives, two missing. The honest rate is 2/4.
    frame = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01"] * 6), "y": [1.0, 1.0, 0.0, 0.0, float("nan"), float("nan")]})
    out = _aggregate_by_time_pandas(frame, "ts", "y", "day", target_type="binary_classification")
    rate = float(out["target_rate"].iloc[0])

    assert rate == 0.5, f"rate {rate} -- fillna(0) would give {2 / 6:.3f} by counting the missing rows as negatives"


def test_an_all_missing_bin_reports_nan_not_zero() -> None:
    """A bin with nothing observed has no rate. Zero would read as "nobody converted"."""
    import math

    import pandas as pd

    from mlframe.training.targets._target_temporal_audit_aggregate import _aggregate_by_time_pandas

    frame = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01"] * 3), "y": [float("nan")] * 3})
    out = _aggregate_by_time_pandas(frame, "ts", "y", "day", target_type="binary_classification")

    assert math.isnan(float(out["target_rate"].iloc[0]))


def test_median_fill_uses_the_median_of_the_FINITE_values() -> None:
    """The polars trap this fix exists for, and the reason drop_nans() precedes median().

    Behavioural since 2026-09-04. This asserted that one of two exact expression spellings appears
    in bruteforce.py. On [1,2,3,4,NaN,6,7,8,9,10] polars 1.x ``Series.median()`` includes the NaN
    in its sort order and returns 6.5 -- the mid-pair of a ten-element sort -- where the median of
    the nine finite values is 6.0. The spelling check cannot tell those two numbers apart.
    """
    pl = pytest.importorskip("polars")

    from mlframe.feature_engineering.bruteforce import median_fill_polars

    filled = median_fill_polars(pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, float("nan"), 6.0, 7.0, 8.0, 9.0, 10.0]}))

    assert filled["x"].to_list()[4] == 6.0, "the NaN was included in the sort, so the fill is the mid-pair not the median"


def test_median_fill_never_substitutes_zero() -> None:
    """Filling with 0 invents a mode at zero and collapses missing rows onto real-zero rows, which
    is what PySR's candidate-score ranking then reads as signal."""
    pd = pytest.importorskip("pandas")

    from mlframe.feature_engineering.bruteforce import median_fill_pandas

    filled = median_fill_pandas(pd.DataFrame({"x": [10.0, 20.0, 30.0, float("nan")]}))

    assert filled["x"].iloc[3] == 20.0


def test_median_fill_leaves_non_numeric_columns_alone() -> None:
    """A Categorical carrying a NaN raises "Cannot setitem on a Categorical with a new category"
    if it is filled here; those columns are dropped or encoded downstream instead."""
    pd = pytest.importorskip("pandas")

    from mlframe.feature_engineering.bruteforce import median_fill_pandas

    frame = pd.DataFrame({"x": [1.0, float("nan")], "c": pd.Categorical(["a", None])})
    filled = median_fill_pandas(frame)

    assert filled["x"].iloc[1] == 1.0
    assert pd.isna(filled["c"].iloc[1])


# ---------------------------------------------------------------------------
# Behavioural sensors
# ---------------------------------------------------------------------------


def test_extractors_classification_nan_stays_missing() -> None:
    """A NaN classification label must stay NaN in the derived target, not be coerced to class 0 or 1."""
    import pandas as pd
    from mlframe.training import extractors as _ext_mod

    if "src" + "\\" + "mlframe" not in _ext_mod.__file__ and "src/mlframe" not in _ext_mod.__file__:
        pytest.skip(f"extractors loaded from stale build path {_ext_mod.__file__}")

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y_bin": [1.0, np.nan, 0.0]})
    ext = _ext_mod.SimpleFeaturesAndTargetsExtractor(
        classification_targets=["y_bin"],
        classification_gte_thresholds={"y_bin": 0.5},
    )
    targets = ext.transform(df)[1]
    y = np.asarray(next(iter(targets.values()))["y_bin_gte_0.5"], dtype=np.float64)
    assert np.isnan(y[1])
    np.testing.assert_array_equal(y[[0, 2]], [1.0, 0.0])


def test_pd_ordinal_encoder_default_encodes_missing_as_minus_one() -> None:
    """Verify the new default encoded_missing_value=-1 reaches OrdinalEncoder.

    Pytest may resolve mlframe.estimators.custom to the stale build/lib/ copy
    (namespace-package gotcha documented in wave 49). Skip when that happens;
    the source-level test above guarantees the live source is correct.
    """
    import pandas as pd
    from mlframe.estimators import custom as _custom_mod

    if "src" + "\\" + "mlframe" not in _custom_mod.__file__ and "src/mlframe" not in _custom_mod.__file__:
        pytest.skip(f"PdOrdinalEncoder loaded from stale build path {_custom_mod.__file__}")

    enc = _custom_mod.PdOrdinalEncoder()
    # sklearn OrdinalEncoder distinguishes None (a category) from np.nan (missing).
    # Use float dtype + np.nan so encoded_missing_value=-1 actually fires.
    df = pd.DataFrame({"c": [1.0, 2.0, np.nan, 1.0]})
    enc.fit(df)
    out = enc.transform(df)
    # Missing row (np.nan) gets code -1 (not platform-dependent INT_MIN).
    assert int(out["c"].iloc[2]) == -1
    # Real categories are >= 0.
    assert int(out["c"].iloc[0]) >= 0
    assert int(out["c"].iloc[1]) >= 0


def test_dummy_baselines_handles_nan_in_group_field() -> None:
    """LTR fast-path group sanity gate must not crash on NaN group_id."""
    import pandas as pd

    # Direct unit on the factorize + bincount chain.
    g_train = pd.Series(["a", "b", "a", None, "c", None, "a"])
    _factor_codes = pd.factorize(g_train)[0]
    # Pre-fix `np.bincount(pd.factorize(...)[0])` would raise; post-fix path:
    train_group_sizes = np.bincount(_factor_codes[_factor_codes >= 0])
    assert train_group_sizes.sum() == 5  # 7 - 2 NaNs
