"""Regression tests for the F4 (phase fixes) audit wave.

Covers:
    - SKEW-COL-ORDER (``_validate_input_columns_against_metadata`` prefers raw_input_columns).
    - SKEW-RECURRENT (``_apply_recurrent_to_ensemble`` idempotent helper extracted).
    - DSET-REUSE-NO-PP-KEY (cache key folds pp_name so two pre_pipelines don't collide).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# SKEW-COL-ORDER
# ---------------------------------------------------------------------------


def test_validate_input_columns_prefers_raw_input_columns():
    """Validate input columns prefers raw input columns."""
    pl = pytest.importorskip("polars")
    from mlframe.training.core._misc_helpers import _validate_input_columns_against_metadata

    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    metadata = {
        "raw_input_columns": ["a", "b"],
        # Stale legacy keys must NOT win: this is the bug -- pre-fix the fallback was order-sensitive
        # and a post-pipeline ``columns`` key would silently override the raw input check.
        "input_columns": ["wrong"],
        "columns": ["wrong"],
    }
    out = _validate_input_columns_against_metadata(df, metadata, verbose=False)
    assert list(out.columns) == ["a", "b"]


def test_validate_input_columns_falls_back_to_legacy_keys():
    """Validate input columns falls back to legacy keys."""
    pl = pytest.importorskip("polars")
    from mlframe.training.core._misc_helpers import _validate_input_columns_against_metadata

    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    # No raw_input_columns -> input_columns wins.
    out = _validate_input_columns_against_metadata(df, {"input_columns": ["a", "b"]}, verbose=False)
    assert list(out.columns) == ["a", "b"]
    # No raw_input_columns AND no input_columns -> columns wins.
    out = _validate_input_columns_against_metadata(df, {"columns": ["a", "b"]}, verbose=False)
    assert list(out.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# SKEW-RECURRENT
# ---------------------------------------------------------------------------


def test_apply_recurrent_to_ensemble_is_importable_and_idempotent_on_empty():
    """The shared helper must be importable from _phase_recurrent and degrade to the prior dict
    when the member list is too short to support an ensemble. Idempotency = calling twice
    returns the same shape."""
    from mlframe.training.core._phase_recurrent import _apply_recurrent_to_ensemble

    ctx = SimpleNamespace(
        models={},
        ensembles={},
        train_idx=np.arange(3),
        val_idx=None,
        test_idx=None,
        verbose=0,
        model_name="m",
    )
    target_values = np.arange(3)
    prior = {"prior_method": "result"}
    res1 = _apply_recurrent_to_ensemble(
        ctx=ctx,
        ensemble_dict=prior,
        target_type="regression",
        target_name="y",
        target_values=target_values,
    )
    res2 = _apply_recurrent_to_ensemble(
        ctx=ctx,
        ensemble_dict=prior,
        target_type="regression",
        target_name="y",
        target_values=target_values,
    )
    # Idempotent: both calls return the same (unchanged) prior dict because there are <2 members.
    assert res1 is prior
    assert res2 is prior


# ---------------------------------------------------------------------------
# DSET-REUSE-NO-PP-KEY
# ---------------------------------------------------------------------------


def test_dataset_reuse_cache_key_distinguishes_pp_names():
    """Dataset reuse cache key distinguishes pp names."""
    from mlframe.training.core._phase_train_one_target import _dataset_reuse_cache_key

    k_a = _dataset_reuse_cache_key("xgb", "MRMR")
    k_b = _dataset_reuse_cache_key("xgb", "ordinary")
    assert k_a != k_b
    # Same pp_name -> same key.
    assert _dataset_reuse_cache_key("xgb", "MRMR") == k_a


def test_dataset_reuse_capture_and_restore_round_trip():
    """Dataset reuse capture and restore round trip."""
    from mlframe.training.core._phase_train_one_target import (
        _capture_dataset_reuse_cache,
        _restore_dataset_reuse_cache,
    )

    ctx = SimpleNamespace(artifacts={})

    class _Tpl:
        """Groups tests covering tpl."""
        _cached_train_dmatrix = object()

    tpl_a = _Tpl()
    tpl_b = _Tpl()
    _capture_dataset_reuse_cache(ctx, "xgb", tpl_a, pp_name="MRMR")
    _capture_dataset_reuse_cache(ctx, "xgb", tpl_b, pp_name="ordinary")
    # Restore to a fresh template for pp_name=MRMR: must see tpl_a's object, NOT tpl_b's.
    fresh = SimpleNamespace()
    _restore_dataset_reuse_cache(ctx, "xgb", fresh, pp_name="MRMR")
    assert getattr(fresh, "_cached_train_dmatrix") is tpl_a._cached_train_dmatrix
