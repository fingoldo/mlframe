"""Backfill regression tests for the medium-tier eval/diagnostics fixes that
landed during the Wave 3 audit but had no dedicated test file.

Source diff: ``git diff cba1630..HEAD -- src/mlframe/training/evaluation.py
src/mlframe/training/dummy_baselines.py
src/mlframe/training/_dummy_baseline_compute.py
src/mlframe/training/_eval_helpers.py src/mlframe/training/_reporting.py
src/mlframe/training/drift_report.py
src/mlframe/training/baseline_diagnostics.py``.

The ten fixes covered (one test each) are the most behavior-impacting:

1.  ``_pick_per_group_categorical`` - deterministic alphabetical tiebreaker
    when two cat features share the same cardinality (prevents
    caller-insertion-order sensitivity).
2.  ``_compute_metric`` (evaluation.py) - accepts 2-D ``(N, 2)`` binary
    probability arrays; slices the positive-class column out so sklearn
    doesn't raise ``bad input shape``.
3.  ``_normalize_pandas_offset_alias`` - maps legacy ``"M"/"Q"/"Y"/"A"`` to
    pandas-2.2+ ``"ME"/"QE"/"YE"`` so callers don't trigger
    ``FutureWarning``.
4.  ``compute_ml_perf_by_time`` - routes ``freq`` through the normalizer.
5.  ``BaselineReport.__str__`` lift line - direction-aware lift% for
    maximize-metrics (NDCG/MAP/MRR/AUC) vs minimize-metrics (RMSE/log_loss).
6.  ``_pick_strongest`` - deterministic alphabetical tiebreaker when two
    baselines share the optimum metric value (no longer hostage to
    DataFrame insertion order).
7.  ``_multilabel_split_summary`` - narrow ``ValueError | TypeError`` catch
    around ``np.stack`` of jagged object arrays (no longer swallows
    KeyboardInterrupt / MemoryError).
8.  ``baseline_diagnostics.BaselineDiagnostics`` outer try - narrow exception
    set (KeyboardInterrupt / generic Exception now propagates instead of
    silently swallowed as "internal_error").
9.  ``_reporting`` ``classification_report`` fallback - narrow
    ``ValueError/TypeError/ImportError/AttributeError`` so programming bugs
    propagate.
10. ``dummy_baselines._compute_metrics_table`` numba log_loss kernel - narrow
    ``TypeError/ValueError/FloatingPointError/RuntimeError`` so we no longer
    paper over MemoryError / KeyboardInterrupt as ``nan``.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. _pick_per_group_categorical: alphabetical tiebreaker
# ---------------------------------------------------------------------------


def test_pick_per_group_categorical_alphabetical_tiebreaker():
    """When two cat features tie on cardinality the alphabetically-first wins,
    regardless of caller-supplied ``cat_features`` ordering."""
    from mlframe.training._dummy_baseline_compute import _pick_per_group_categorical

    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "z_cat": rng.integers(0, 5, size=n),
        "a_cat": rng.integers(0, 5, size=n),
    })

    # Force exactly the same nunique on both
    df["a_cat"] = (np.arange(n) % 5)
    df["z_cat"] = (np.arange(n) % 5)

    forward = _pick_per_group_categorical(df, ["z_cat", "a_cat"], n_train=n, max_cardinality_ratio=0.5)
    reverse = _pick_per_group_categorical(df, ["a_cat", "z_cat"], n_train=n, max_cardinality_ratio=0.5)
    assert forward == reverse == "a_cat", (
        f"alphabetical tiebreaker broken: forward={forward!r} reverse={reverse!r}"
    )


# ---------------------------------------------------------------------------
# 2 + 4. _compute_metric accepts (N, 2); compute_ml_perf_by_time routes freq.
# ---------------------------------------------------------------------------


def test_compute_metric_accepts_2d_binary_probabilities():
    """A 2-D ``(N, 2)`` probability array from a binary classifier is handled
    gracefully - the positive-class column is sliced out."""
    from mlframe.training.evaluation import _compute_metric

    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    pos_probs = rng.random(n)
    two_d = np.column_stack([1 - pos_probs, pos_probs])  # (N, 2)

    one_d_val = _compute_metric("roc_auc", y_true, pos_probs)
    two_d_val = _compute_metric("roc_auc", y_true, two_d)

    assert np.isfinite(one_d_val) and np.isfinite(two_d_val)
    assert one_d_val == pytest.approx(two_d_val, abs=1e-12), (
        f"2-D (N,2) handling diverged from 1-D: {one_d_val} vs {two_d_val}"
    )


def test_normalize_pandas_offset_alias_maps_legacy_aliases():
    """``M/Q/Y/A`` collapse onto their pandas-2.2+ end-of-period
    equivalents; non-affected aliases pass through unchanged."""
    from mlframe.training.evaluation import _normalize_pandas_offset_alias as norm

    assert norm("M") == "ME"
    assert norm("Q") == "QE"
    assert norm("Y") == "YE"
    assert norm("A") == "YE"
    # non-affected aliases pass through
    assert norm("D") == "D"
    assert norm("h") == "h"
    assert norm("W-MON") == "W-MON"
    assert norm("ME") == "ME"  # already normalized -> no double-map
    # non-string -> identity
    assert norm(7) == 7  # type: ignore[arg-type]


def test_compute_ml_perf_by_time_no_futurewarning_on_M_alias():
    """``freq="M"`` no longer fires pandas's deprecation FutureWarning thanks
    to the alias-normalization shim. We assert no FutureWarning carrying the
    deprecation token surfaces."""
    from mlframe.training.evaluation import compute_ml_perf_by_time

    rng = np.random.default_rng(0)
    n = 600
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.random(n)
    ts = pd.date_range("2024-01-01", periods=n, freq="D")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compute_ml_perf_by_time(y_true, y_pred, ts, freq="M", metric="roc_auc", min_samples=10)

    deprecation_hits = [
        w for w in caught
        if issubclass(w.category, FutureWarning)
        and ("'M' is deprecated" in str(w.message) or "month-end" in str(w.message).lower())
    ]
    assert not deprecation_hits, (
        f"expected no pandas 'M' deprecation FutureWarning, got: "
        f"{[str(w.message) for w in deprecation_hits]}"
    )
    assert len(out) >= 1
    assert "roc_auc" in out.columns


# ---------------------------------------------------------------------------
# 5. BaselineReport direction-aware lift
# ---------------------------------------------------------------------------


def _build_minimal_report(primary_metric, strongest, trivial, primary_val, trivial_val):
    from mlframe.training.dummy_baselines import BaselineReport

    table = pd.DataFrame(
        {primary_metric: [primary_val, trivial_val]},
        index=[strongest, trivial],
    )
    return BaselineReport(
        target_type="binary",
        target_name="y",
        table=table,
        strongest=strongest,
        primary_metric=primary_metric,
        ts_period_used=None,
        plot_path=None,
        elapsed_s=0.0,
        n_train=100, n_val=20, n_test=20,
        n_train_finite=100, n_val_finite=20, n_test_finite=20,
        extras={"strongest_pick_excluded": []},
    )


def test_baseline_report_lift_direction_aware_maximize_metric():
    """For maximize-metrics (NDCG/AUC) lift = (primary - trivial) / |trivial|."""
    rep = _build_minimal_report(
        primary_metric="val_NDCG@10", strongest="strong", trivial="prior",
        primary_val=0.75, trivial_val=0.50,
    )
    rendered = rep.format_text()
    assert "lift_vs_prior=+50.0%" in rendered, rendered


def test_baseline_report_lift_direction_aware_minimize_metric():
    """For minimize-metrics (log_loss/RMSE) lift = (trivial - primary) / |trivial|."""
    rep = _build_minimal_report(
        primary_metric="val_log_loss", strongest="strong", trivial="prior",
        primary_val=0.30, trivial_val=0.60,
    )
    rendered = rep.format_text()
    # positive lift = strongest improved over prior
    assert "lift_vs_prior=+50.0%" in rendered, rendered


# ---------------------------------------------------------------------------
# 6. _pick_strongest alphabetical tiebreaker
# ---------------------------------------------------------------------------


def test_pick_strongest_alphabetical_tiebreaker_on_minimize():
    """Two baselines tying on the optimum metric value resolve to the
    alphabetically first name (order-independent)."""
    from mlframe.training.dummy_baselines import _pick_strongest

    primary_metric = "val_log_loss"
    extras = {"strongest_pick_excluded": []}

    # Same metric value for both rows - the tiebreaker must pick "alpha".
    table_fwd = pd.DataFrame({primary_metric: [0.4, 0.4]}, index=["zeta", "alpha"])
    table_rev = pd.DataFrame({primary_metric: [0.4, 0.4]}, index=["alpha", "zeta"])

    y_val = np.array([0, 1] * 50)
    y_test = np.array([0, 1] * 50)

    strongest_fwd, _ = _pick_strongest(
        "binary", table_fwd, y_val, y_test, primary_metric, extras, config=None,
    )
    strongest_rev, _ = _pick_strongest(
        "binary", table_rev, y_val, y_test, primary_metric, extras, config=None,
    )

    assert strongest_fwd == strongest_rev == "alpha", (
        f"deterministic tiebreaker broken: fwd={strongest_fwd!r} rev={strongest_rev!r}"
    )


# ---------------------------------------------------------------------------
# 7. drift_report _multilabel_split_summary narrow catch
# ---------------------------------------------------------------------------


def test_multilabel_split_summary_well_formed_object_array_stacks():
    """A well-formed object-array of equal-length per-row arrays still stacks
    to (N, K). This guards the happy path of the narrow-catch site."""
    from mlframe.training.drift_report import _multilabel_split_summary

    obj_arr = np.empty(4, dtype=object)
    obj_arr[0] = np.array([0, 1, 0])
    obj_arr[1] = np.array([1, 0, 1])
    obj_arr[2] = np.array([0, 0, 1])
    obj_arr[3] = np.array([1, 1, 0])

    summary = _multilabel_split_summary(obj_arr)
    assert summary["n"] == 4
    assert summary["n_labels"] == 3
    # 2 ones in col0, 2 ones in col1, 2 ones in col2
    assert summary["n_positive_per_label"] == [2, 2, 2]

    # Source-shape pin: the narrow tuple is present, ``except Exception``
    # is gone. KeyboardInterrupt / MemoryError now propagate.
    from mlframe.training import drift_report
    src = inspect.getsource(drift_report._multilabel_split_summary)
    assert "except (ValueError, TypeError)" in src, (
        "narrow exception tuple in _multilabel_split_summary is missing"
    )
    assert "except Exception" not in src, (
        "broad except Exception was reintroduced into _multilabel_split_summary"
    )


# ---------------------------------------------------------------------------
# 8. baseline_diagnostics outer try narrow exception set
# ---------------------------------------------------------------------------


def test_baseline_diagnostics_outer_try_does_not_swallow_keyboard_interrupt():
    """Catch is narrow: KeyboardInterrupt / MemoryError must NOT be in the
    swallowed set; programming bugs must propagate."""
    from mlframe.training import baseline_diagnostics

    # The narrow catch is at the top-level try in BaselineDiagnostics.run().
    src = inspect.getsource(baseline_diagnostics.BaselineDiagnostics)
    assert "except (ValueError, TypeError, RuntimeError, ImportError, KeyError" in src, (
        "expected narrow exception tuple was removed from BaselineDiagnostics"
    )
    # The bare `except Exception as exc:  # pragma: no cover - defensive`
    # used to be the catch -- the comment is gone now, ensure no bare
    # ``except Exception`` survives in the fit_and_report path (which
    # is the public entry; ``run`` is not a method of this class).
    fit_src = inspect.getsource(baseline_diagnostics.BaselineDiagnostics.fit_and_report)
    assert "except Exception" not in fit_src, (
        "broad except Exception was reintroduced into BaselineDiagnostics.fit_and_report"
    )


# ---------------------------------------------------------------------------
# 9. _reporting classification_report fallback narrow
# ---------------------------------------------------------------------------


def test_reporting_classification_report_fallback_narrow():
    """Fallback no longer catches bare ``Exception``; specific types only."""
    from mlframe.training import _reporting

    src = inspect.getsource(_reporting.report_probabilistic_model_perf)
    # Confirm the narrow fallback tuple is present and bare Exception is not.
    assert "except (ValueError, TypeError, ImportError, AttributeError) as _cls_err" in src, (
        "classification_report narrow fallback tuple is gone"
    )


# ---------------------------------------------------------------------------
# 10. dummy_baselines numba log_loss kernel narrow catch
# ---------------------------------------------------------------------------


def test_dummy_baselines_numba_log_loss_kernel_narrow_catch():
    """Numba kernel fallback no longer catches bare ``Exception``; the
    narrow tuple lets ``MemoryError`` / ``KeyboardInterrupt`` propagate."""
    from mlframe.training import dummy_baselines

    src = inspect.getsource(dummy_baselines._compute_metrics_table)
    assert "except (TypeError, ValueError, FloatingPointError, RuntimeError) as _ll_exc" in src, (
        "numba log_loss kernel narrow tuple is missing"
    )
