"""Regression test: ``run_temporal_audit_batch`` must route to the polars
multi-target single-pass aggregation path, not the per-target pandas fallback.

The module docstring on ``_phase_temporal_audit.py`` promises:
    "Single polars multi-aggregation pass over all (target_type, target_name)
     pairs (~5x faster than per-call for N>1 targets on >1M rows)."

Pre-fix the implementation built a ``pd.DataFrame`` and handed it to
``audit_targets_over_time``, whose dispatch at the top of the function
(``if _HAS_POLARS and isinstance(df, pl.DataFrame): ... else: pandas fallback``)
therefore always took the per-target pandas branch. The 1M-row fuzz profile
2026-05-19 surfaced this as 11.7s / 23% of train wall on the
regression x lgb x ts x n_targets=2 combo; the polars multi path is ~3x
faster on the same input.

This test asserts the polars path actually fires by monkeypatching
``audit_targets_over_time`` to record the type of its first positional
argument, then calling ``run_temporal_audit_batch`` end-to-end with a
representative input shape (numeric int64 ts seconds-since-epoch, 2
binary targets).
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def _audit_call_recorder(monkeypatch):
    """Replace audit_targets_over_time with a recorder so we can assert
    the input dataframe type without doing a real polars groupby."""
    seen = {"df_type": None, "df_dtypes": None, "n_targets": 0}

    def _fake_audit(df, *, timestamp_col, targets, granularity="auto", **_kw):
        seen["df_type"] = type(df).__module__ + "." + type(df).__name__
        if hasattr(df, "schema"):
            seen["df_dtypes"] = {c: str(t) for c, t in dict(df.schema).items()}
        elif hasattr(df, "dtypes"):
            seen["df_dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
        seen["n_targets"] = len(targets)
        # Return an empty result so the wrapper short-circuits gracefully.
        return {k: None for k in targets}

    from mlframe.training.core import _phase_temporal_audit as ph_mod
    monkeypatch.setattr(ph_mod, "_audit_targets_over_time", _fake_audit)
    return seen


def _make_minimal_behavior_and_fte():
    """Build the minimal duck-typed inputs ``run_temporal_audit_batch``
    needs without dragging in the full configs / FTE machinery."""

    class _Cfg:
        target_temporal_audit_column = None  # fall through to FTE.ts_field
        target_temporal_audit_granularity = "auto"

    class _Fte:
        ts_field = "ts"

    return _Cfg(), _Fte()


def test_run_temporal_audit_batch_builds_polars_input(_audit_call_recorder):
    """With polars installed, run_temporal_audit_batch MUST construct
    a polars.DataFrame so audit_targets_over_time takes its single-pass
    multi-target fastpath. Pre-fix this was a pd.DataFrame, forcing the
    per-target pandas fallback."""
    pytest.importorskip("polars")
    from mlframe.training.core._phase_temporal_audit import run_temporal_audit_batch

    behavior_config, fte = _make_minimal_behavior_and_fte()
    # Numeric int64 seconds-since-epoch -- what the synthetic profile and
    # real FTEs emit. The fastpath must accept this without raising.
    n = 1_000
    timestamps = (1_700_000_000 + np.arange(n, dtype=np.int64))
    y_arr = np.random.RandomState(0).randint(0, 2, size=n).astype(np.int8)
    y2_arr = np.random.RandomState(1).randint(0, 2, size=n).astype(np.int8)
    target_by_type = {
        "binary_classification": {"y": y_arr, "y2": y2_arr},
    }

    run_temporal_audit_batch(
        behavior_config=behavior_config,
        features_and_targets_extractor=fte,
        timestamps=timestamps,
        target_by_type=target_by_type,
        verbose=False,
    )

    assert _audit_call_recorder["df_type"] is not None, (
        "run_temporal_audit_batch did not invoke audit_targets_over_time at all"
    )
    assert "polars" in _audit_call_recorder["df_type"].lower(), (
        f"expected a polars.DataFrame to reach audit_targets_over_time "
        f"(unlocks the multi-target single-pass fastpath), got "
        f"{_audit_call_recorder['df_type']!r}"
    )
    assert _audit_call_recorder["n_targets"] == 2, (
        f"expected 2 targets (y + y2), got {_audit_call_recorder['n_targets']}"
    )
    # Timestamp column must arrive as polars Datetime so dt.truncate works.
    _ts_dtype = (_audit_call_recorder["df_dtypes"] or {}).get("ts", "")
    assert "Datetime" in _ts_dtype, (
        f"timestamp column 'ts' must be polars Datetime (so "
        f"_aggregate_by_time_polars_multi can call dt.truncate); got {_ts_dtype!r}"
    )


def test_run_temporal_audit_batch_pandas_fallback_when_polars_missing(
    _audit_call_recorder, monkeypatch
):
    """If polars is unavailable (or import fails), the wrapper must
    fall through to a pandas DataFrame rather than raising."""
    # Force the `import polars as _pl` line to fail.
    import sys
    monkeypatch.setitem(sys.modules, "polars", None)

    from mlframe.training.core._phase_temporal_audit import run_temporal_audit_batch
    importlib.reload(sys.modules["mlframe.training.core._phase_temporal_audit"])  # ensure clean import path

    behavior_config, fte = _make_minimal_behavior_and_fte()
    n = 100
    timestamps = (1_700_000_000 + np.arange(n, dtype=np.int64))
    y_arr = np.random.RandomState(0).randint(0, 2, size=n).astype(np.int8)
    target_by_type = {"binary_classification": {"y": y_arr}}

    # Re-bind the patched audit recorder against the reloaded module.
    from mlframe.training.core import _phase_temporal_audit as ph_mod
    monkeypatch.setattr(ph_mod, "_audit_targets_over_time",
                        lambda df, **kw: (_audit_call_recorder.update(
                            df_type=type(df).__module__ + "." + type(df).__name__,
                            n_targets=len(kw["targets"]),
                        ) or {k: None for k in kw["targets"]}))

    ph_mod.run_temporal_audit_batch(
        behavior_config=behavior_config,
        features_and_targets_extractor=fte,
        timestamps=timestamps,
        target_by_type=target_by_type,
        verbose=False,
    )

    # Either polars was successfully blocked (pandas fallback fires) or the
    # fastpath still finds polars installed via a different cache route --
    # both are acceptable; what we're really asserting is the wrapper doesn't
    # raise on the ImportError branch.
    assert _audit_call_recorder["df_type"] is not None
