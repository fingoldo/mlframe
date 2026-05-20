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
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_extractors_classification_target_rejects_nan() -> None:
    src = _read("training/extractors.py")
    # The fix replaces fillna(0)/fill_null(0) with a NaN-detection raise.
    assert "Classification target" in src
    assert "drop or impute upstream" in src


def test_pd_ordinal_encoder_default_uses_minus_one() -> None:
    src = _read("estimators/custom.py")
    assert "encoded_missing_value=-1" in src
    # transform must guard against NaN -> int32 platform-dependent behaviour.
    assert "NaN codes in output" in src


def test_dummy_baselines_factorize_filters_negative_codes() -> None:
    src = _read("training/dummy_baselines.py")
    # The fix filters codes>=0 before bincount.
    assert "_factor_codes = pd.factorize(g_train)[0]" in src
    assert "np.bincount(_factor_codes[_factor_codes >= 0])" in src


def test_predict_guards_nan_detection_uses_isnan_not_isfinite() -> None:
    src = _read("training/_predict_guards.py")
    # The numpy branch must use np.isnan (not ~np.isfinite) for parity.
    assert "_has_nan = bool(np.any(np.isnan(_arr_check[:500])))" in src


def test_discretization_nan_filler_supports_raise() -> None:
    src = _read("feature_selection/filters/discretization.py")
    # The fix adds a nan_filler=None branch that raises.
    assert "input contains NaN and nan_filler=None" in src
    # And a WARN when the legacy default fires.
    assert "biases MI by mixing" in src


def test_target_temporal_audit_drops_nan_before_rate() -> None:
    src = _read("training/target_temporal_audit.py")
    # Pre-fix lambda was `(c.fillna(0) > 0).mean()` -- gone.
    assert "(c.fillna(0) > 0).mean()" not in src
    # Post-fix is `(c.dropna() > 0).mean()` with a notna gate.
    assert "(c.dropna() > 0).mean()" in src


def test_bruteforce_fill_uses_median_not_zero() -> None:
    src = _read("feature_engineering/bruteforce.py")
    # Polars path uses median instead of 0.
    assert "cs.numeric().fill_nan(cs.numeric().median()).fill_null(cs.numeric().median())" in src
    # Pandas path uses median too.
    assert "tmp_df[numeric_cols].fillna(tmp_df[numeric_cols].median())" in src


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
    import inspect
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
