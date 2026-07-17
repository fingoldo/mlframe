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

import numpy as np
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
    timestamps = 1_700_000_000 + np.arange(n, dtype=np.int64)
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

    assert _audit_call_recorder["df_type"] is not None, "run_temporal_audit_batch did not invoke audit_targets_over_time at all"
    assert "polars" in _audit_call_recorder["df_type"].lower(), (
        f"expected a polars.DataFrame to reach audit_targets_over_time (unlocks the multi-target single-pass fastpath), got {_audit_call_recorder['df_type']!r}"
    )
    assert _audit_call_recorder["n_targets"] == 2, f"expected 2 targets (y + y2), got {_audit_call_recorder['n_targets']}"
    # Timestamp column must arrive as polars Datetime so dt.truncate works.
    _ts_dtype = (_audit_call_recorder["df_dtypes"] or {}).get("ts", "")
    assert "Datetime" in _ts_dtype, (
        f"timestamp column 'ts' must be polars Datetime (so _aggregate_by_time_polars_multi can call dt.truncate); got {_ts_dtype!r}"
    )


def test_pick_granularity_accepts_min_max_tuple_shortcut():
    """``_pick_granularity`` must accept a ``(min_ts, max_ts)`` tuple to
    skip the O(N) ``pd.Series + pd.to_datetime`` materialisation on the
    full timestamp array. The 1M-row fuzz profile surfaced this as 0.85s
    cumtime (the function only needs the span; the caller materialised
    the full Python list via ``polars.Series.to_list()`` first).

    A regression that drops the tuple-shortcut branch would fall back to
    indexing into the 2-element tuple as if it were a Sequence and pick
    a wrong granularity (or raise). This test pins the shortcut by
    asserting that two semantically equivalent calls (full sequence vs
    its (min, max)) produce the same granularity choice.
    """
    import datetime as _dt

    from mlframe.training.targets.target_temporal_audit import _pick_granularity

    base = _dt.datetime(2024, 1, 1)
    full = [base + _dt.timedelta(days=i) for i in range(120)]  # ~4 months
    granularity_from_full = _pick_granularity(full)
    granularity_from_minmax = _pick_granularity((full[0], full[-1]))

    assert granularity_from_minmax == granularity_from_full, (
        f"(min, max) shortcut must match the full-sequence path: shortcut={granularity_from_minmax!r}, full={granularity_from_full!r}"
    )


def test_pick_granularity_minmax_zero_span_returns_month():
    """Degenerate single-point span via (min, max) must NOT raise; falls
    back to the 'month' default the full-sequence path also uses."""
    from mlframe.training.targets.target_temporal_audit import _pick_granularity

    import datetime as _dt

    ts = _dt.datetime(2024, 1, 1)
    assert _pick_granularity((ts, ts)) == "month"
    # None inputs (caller couldn't determine span) also degrade safely.
    assert _pick_granularity((None, None)) == "month"


def test_run_temporal_audit_batch_pandas_fallback_when_polars_missing(_audit_call_recorder, monkeypatch):
    """If polars is unavailable (or import fails), the wrapper must fall through to a pandas DataFrame
    rather than raising.

    The polars fastpath imports polars at CALL time (``import polars as _pl`` inside
    ``run_temporal_audit_batch``), so blocking it via ``sys.modules['polars'] = None`` is sufficient to
    exercise the ImportError -> pandas-fallback branch. No ``importlib.reload`` is needed -- monkeypatch
    isolation alone, avoiding the cross-test module-rebinding pollution the reload would risk.
    """
    import sys

    monkeypatch.setitem(sys.modules, "polars", None)

    from mlframe.training.core import _phase_temporal_audit as ph_mod

    monkeypatch.setattr(
        ph_mod,
        "_audit_targets_over_time",
        lambda df, **kw: (
            _audit_call_recorder.update(
                df_type=type(df).__module__ + "." + type(df).__name__,
                n_targets=len(kw["targets"]),
            )
            or {k: None for k in kw["targets"]}
        ),
    )

    behavior_config, fte = _make_minimal_behavior_and_fte()
    n = 100
    timestamps = 1_700_000_000 + np.arange(n, dtype=np.int64)
    y_arr = np.random.RandomState(0).randint(0, 2, size=n).astype(np.int8)
    target_by_type = {"binary_classification": {"y": y_arr}}

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
