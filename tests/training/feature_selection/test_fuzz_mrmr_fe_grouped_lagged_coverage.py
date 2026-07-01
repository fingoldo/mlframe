"""Coverage sensor: the grouped_delta / lagged_diff MRMR-FE kinds actually RUN in the fuzz sweep.

Previously these two ``mrmr_fe_ratio_delta_diff_cfg`` kinds were canonicalised to "off" because the fuzz frame builder
supplied no group / time column, so the sweep never exercised them. The frame builder now emits a group key
(``mrmr_fe_group``) for grouped_delta and a monotone order column (``mrmr_fe_order``) for lagged_diff, plus a numeric
source column engineered so the FE transform recovers a target-predictive signal that clears the MRMR Tier-1 local-MI
floor. This sensor builds the frame + MRMR kwargs exactly as the suite does and asserts the kinds emit surviving
engineered columns end-to-end -- a future regression that re-collapses the kinds (or breaks the wiring) fails here.
"""
from __future__ import annotations

import inspect

import pytest

from tests.training._fuzz_combo import AXES, _build_combo, build_frame_for_combo, build_mrmr_kwargs

MRMR = pytest.importorskip("mlframe.feature_selection.filters.mrmr").MRMR


def _combo(kind):
    axes = {name: values[0] for name, values in AXES.items()}
    axes.update(
        use_mrmr_fs=True,
        n_rows=2000,
        cat_feature_count=0,
        target_type="binary_classification",
        input_type="pandas",
        mrmr_fe_ratio_delta_diff_cfg=kind,
    )
    return _build_combo(models=("cb",), axes=axes, seed=42)


@pytest.mark.parametrize(
    "kind, features_attr",
    [
        ("grouped_delta", "grouped_delta_features_"),
        ("lagged_diff", "lagged_diff_features_"),
    ],
)
def test_mrmr_fe_kind_emits_surviving_engineered_columns(kind, features_attr):
    combo = _combo(kind)
    # The canon must NOT collapse the kind to "off" -- its canonical key differs from the off combo.
    assert combo.canonical_key() != _combo("off").canonical_key(), f"{kind} was canonicalised away"

    df, target_col, _cats = build_frame_for_combo(combo)
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    kwargs = build_mrmr_kwargs(combo)
    sig = set(inspect.signature(MRMR.__init__).parameters)
    mrmr = MRMR(**{k: v for k, v in kwargs.items() if k in sig})
    mrmr.fit(X, y)

    emitted = list(getattr(mrmr, features_attr, []) or [])
    assert emitted, (
        f"{kind} produced ZERO engineered columns: the kind either did not run or every column was gated out. "
        f"The frame builder must emit a source column whose {kind} transform recovers a signal above the local-MI floor."
    )


def test_off_kind_emits_no_grouped_or_lagged_columns():
    """Control: the 'off' kind must not emit grouped_delta / lagged_diff columns (no false-positive coverage)."""
    combo = _combo("off")
    df, target_col, _cats = build_frame_for_combo(combo)
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()
    kwargs = build_mrmr_kwargs(combo)
    sig = set(inspect.signature(MRMR.__init__).parameters)
    mrmr = MRMR(**{k: v for k, v in kwargs.items() if k in sig})
    mrmr.fit(X, y)
    assert not (getattr(mrmr, "grouped_delta_features_", []) or [])
    assert not (getattr(mrmr, "lagged_diff_features_", []) or [])
