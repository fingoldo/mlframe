"""Wave 96 (2026-05-21): split feature_engineering/timeseries.py
(1132 lines) into timeseries.py (now 893 lines) + new
_timeseries_emit.py (280 lines).

Moved to the sibling file: the 11 ``_emit_*`` per-transform helpers
(groupby block, categorical counts, raw numaggs, differences, ratios,
wavelets, weighted, ewma, rolling, nonlinear, robust, counts_regexp).

Original re-exports them so existing
``from mlframe.feature_engineering.timeseries import _emit_robust``
imports continue to work.
"""
from __future__ import annotations

from pathlib import Path


def test_emit_symbols_still_importable_from_facade() -> None:
    from mlframe.feature_engineering.timeseries import (
        _emit_groupby_block,
        _emit_categorical_counts,
        _emit_raw_numaggs,
        _emit_differences,
        _emit_ratios,
        _emit_wavelets,
        _emit_weighted,
        _emit_ewma,
        _emit_rolling,
        _emit_nonlinear,
        _emit_robust,
        _emit_counts_regexp,
    )
    for fn in (
        _emit_groupby_block,
        _emit_categorical_counts,
        _emit_raw_numaggs,
        _emit_differences,
        _emit_ratios,
        _emit_wavelets,
        _emit_weighted,
        _emit_ewma,
        _emit_rolling,
        _emit_nonlinear,
        _emit_robust,
        _emit_counts_regexp,
    ):
        assert callable(fn), fn


def test_public_ts_api_still_importable() -> None:
    from mlframe.feature_engineering.timeseries import (
        get_numaggs_metadata,
        create_aggregated_features,
        create_windowed_features,
        create_ts_features_parallel,
        compute_corr,
        general_acf,
    )
    for fn in (
        get_numaggs_metadata,
        create_aggregated_features,
        create_windowed_features,
        create_ts_features_parallel,
        compute_corr,
        general_acf,
    ):
        assert callable(fn), fn


def test_facade_below_1k_line_threshold() -> None:
    root = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe" / "feature_engineering"
    facade = root / "timeseries.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"timeseries.py is {n} lines, still over the 1k threshold"


def test_emit_module_owns_the_moved_symbols() -> None:
    """Identity: facade and sibling module expose the SAME object."""
    from mlframe.feature_engineering import timeseries as ts, _timeseries_emit as em
    for name in (
        "_emit_groupby_block",
        "_emit_categorical_counts",
        "_emit_raw_numaggs",
        "_emit_differences",
        "_emit_ratios",
        "_emit_wavelets",
        "_emit_weighted",
        "_emit_ewma",
        "_emit_rolling",
        "_emit_nonlinear",
        "_emit_robust",
        "_emit_counts_regexp",
    ):
        assert getattr(ts, name) is getattr(em, name), name


def test_emit_differences_round_trip() -> None:
    """Functional smoke: feed a tiny array through _emit_differences and verify
    row_features / features_names get extended."""
    import numpy as np
    from mlframe.feature_engineering.timeseries import _emit_differences

    row_features: list = []
    features_names: list = []
    _emit_differences(
        var="x",
        raw_vals=np.array([1.0, 2.0, 4.0, 7.0]),
        numaggs_kwds={},
        dataset_name="ds",
        captions_vars_sep="_",
        row_features=row_features,
        features_names=features_names,
        create_features_names=True,
    )
    # Must have appended at least one numagg.
    assert len(row_features) > 0
    assert len(features_names) == len(row_features)
    # The first name should follow the ds_x_dif_<feat> pattern.
    assert features_names[0].startswith("ds_x_dif_")
