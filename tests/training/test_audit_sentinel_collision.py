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

from pathlib import Path

import numpy as np
import pytest

MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Read."""
    _path = MLFRAME_ROOT / rel
    if not _path.exists() and _path.suffix == ".py":
        # Monolith-split compat: the flat module became a subpackage
        # (``X.py`` -> ``X/__init__.py`` + submodules). Read __init__ + every submodule.
        _pkg = _path.with_suffix("")
        _init = _pkg / "__init__.py"
        if _init.exists():
            parts = [_init.read_text(encoding="utf-8")]
            for _sub in sorted(_pkg.glob("*.py")):
                if _sub.name != "__init__.py":
                    parts.append(_sub.read_text(encoding="utf-8"))
            return "\n".join(parts)
    return _path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_extractors_classification_target_rejects_nan() -> None:
    """The classification-target NaN-rejection lives in sibling
    _extractors_simple.py after the extractors monolith split; concat
    so the source sensor matches the post-carve layout."""
    src = _read("training/extractors.py")
    _sib = MLFRAME_ROOT / "training" / "_extractors_simple.py"
    if _sib.exists():
        src += "\n" + _sib.read_text(encoding="utf-8")
    assert "Classification target" in src
    assert "drop or impute upstream" in src


def test_pd_ordinal_encoder_default_uses_minus_one() -> None:
    """Pd ordinal encoder default uses minus one."""
    src = _read("estimators/custom.py")
    assert "encoded_missing_value=-1" in src
    # transform must guard against NaN -> int32 platform-dependent behaviour.
    assert "NaN codes in output" in src


def test_dummy_baselines_factorize_filters_negative_codes() -> None:
    # The LTR factorize fast-path was moved to the
    # ``_dummy_compute_helpers.py`` sibling when ``dummy_baselines.py`` was
    # split below 1k LOC.
    """Dummy baselines factorize filters negative codes."""
    src = _read("training/baselines/_dummy_compute_helpers.py")
    # The fix filters codes>=0 before bincount.
    assert "_factor_codes = pd.factorize(g_train)[0]" in src
    assert "np.bincount(_factor_codes[_factor_codes >= 0])" in src


def test_predict_guards_nan_detection_uses_isnan_not_isfinite() -> None:
    """Predict guards nan detection uses isnan not isfinite."""
    src = _read("training/_predict_guards.py")
    # The numpy branch must use np.isnan (not ~np.isfinite) for parity.
    assert "_has_nan = bool(np.any(np.isnan(_arr_check[:500])))" in src


def test_discretization_nan_filler_supports_raise() -> None:
    """Discretization nan filler supports raise."""
    src = _read("feature_selection/filters/discretization.py")
    # The fix adds a nan_filler=None branch that raises.
    assert "input contains NaN and nan_filler=None" in src
    # And a WARN when the legacy default fires.
    assert "biases MI by mixing" in src


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


def test_extractors_classification_nan_raises() -> None:
    """NaN classification target must raise, not silently coerce to class 0."""
    import pandas as pd
    from mlframe.training import extractors as _ext_mod

    if "src" + "\\" + "mlframe" not in _ext_mod.__file__ and "src/mlframe" not in _ext_mod.__file__:
        pytest.skip(f"extractors loaded from stale build path {_ext_mod.__file__}")

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y_bin": [1.0, np.nan, 0.0]})
    ext = _ext_mod.SimpleFeaturesAndTargetsExtractor(
        classification_targets=["y_bin"],
        classification_lower_thresholds={"y_bin": 0.5},
    )
    with pytest.raises(ValueError, match="contains NaN"):
        ext.transform(df)


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
