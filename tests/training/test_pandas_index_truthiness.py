"""Regression: pandas Index passed as ``columns=`` must not crash report_model_perf / cross-target chart emitter.

Trace from the production log (2026-05-17 run, target=TVT):
    [dummy-baselines] report_model_perf for dummy failed:
        The truth value of a Index is ambiguous.
    [CompositeCrossTargetEnsemble] target='TVT' could not emit scatter / log charts:
        The truth value of a Index is ambiguous.

Both paths used ``if columns`` / ``columns or []`` which raises ``ValueError`` when ``columns`` is a ``pd.Index`` (non-empty Index has ambiguous truthiness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.reporting._reporting import report_regression_model_perf

_Y = np.array([0.5, 1.5, 1.0, 2.5])
_PREDS = np.array([0.4, 1.4, 1.1, 2.6])


def _report(columns) -> dict:
    """Run the reporter on the fixed targets/preds with ``columns`` and return its metrics, after checking the returned predictions."""
    metrics: dict = {}
    out_preds, out_proba = report_regression_model_perf(
        targets=_Y,
        columns=columns,
        model_name="dummy_mean",
        model=None,
        preds=_PREDS,
        print_report=False,
        show_perf_chart=False,
        metrics=metrics,
    )
    np.testing.assert_array_equal(out_preds, _PREDS)
    assert out_proba is None
    return metrics


def _assert_reference_metrics(metrics: dict) -> None:
    """The reported headline metrics equal the closed-form values for ``_Y`` / ``_PREDS``."""
    err = _Y - _PREDS
    assert metrics["MAE"] == pytest.approx(float(np.mean(np.abs(err))), abs=1e-12)
    assert metrics["RMSE"] == pytest.approx(float(np.sqrt(np.mean(err**2))), abs=1e-12)
    assert metrics["MaxError"] == pytest.approx(float(np.max(np.abs(err))), abs=1e-12)
    assert metrics["R2"] == pytest.approx(1.0 - float(np.sum(err**2) / np.sum((_Y - _Y.mean()) ** 2)), abs=1e-12)


class TestPandasIndexColumns:
    """``columns=df.columns`` (pd.Index) used to crash; now should run."""

    def test_pd_index_does_not_crash_reporter(self) -> None:
        """A non-empty pd.Index as ``columns`` (``if columns`` raised ``The truth value of a Index is ambiguous``) yields the reference metrics."""
        df = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 2.0]})
        _assert_reference_metrics(_report(df.columns))

    def test_empty_pd_index_does_not_crash_reporter(self) -> None:
        """An empty pd.Index as ``columns`` yields the reference metrics."""
        _assert_reference_metrics(_report(pd.DataFrame().columns))

    def test_none_columns_does_not_crash_reporter(self) -> None:
        """``columns=None`` yields the reference metrics."""
        _assert_reference_metrics(_report(None))

    @pytest.mark.parametrize("cols", [["a", "b"], ("a", "b"), pd.Index(["a", "b"])])
    def test_list_tuple_index_all_work(self, cols) -> None:
        """A list, a tuple and a pd.Index of the same names give identical metrics, equal to the reference."""
        metrics = _report(cols)
        _assert_reference_metrics(metrics)
        reference = _report(["a", "b"])
        for key in ("MAE", "RMSE", "MaxError", "R2", "Pearson"):
            assert metrics[key] == reference[key]
