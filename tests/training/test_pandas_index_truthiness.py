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

from mlframe.training._reporting import report_regression_model_perf


class TestPandasIndexColumns:
    """``columns=df.columns`` (pd.Index) used to crash; now should run."""

    def test_pd_index_does_not_crash_reporter(self) -> None:
        df = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 2.0]})
        y = np.array([0.5, 1.5, 1.0, 2.5])
        preds = np.array([0.4, 1.4, 1.1, 2.6])
        # Pre-fix: ``if columns`` on a pd.Index raised ``The truth value of a Index is ambiguous``.
        report_regression_model_perf(
            targets=y,
            columns=df.columns,
            model_name="dummy_mean",
            model=None,
            preds=preds,
            print_report=False,
            show_perf_chart=False,
        )

    def test_empty_pd_index_does_not_crash_reporter(self) -> None:
        y = np.array([0.5, 1.5, 1.0, 2.5])
        preds = np.array([0.4, 1.4, 1.1, 2.6])
        report_regression_model_perf(
            targets=y,
            columns=pd.DataFrame().columns,
            model_name="dummy_mean",
            model=None,
            preds=preds,
            print_report=False,
            show_perf_chart=False,
        )

    def test_none_columns_does_not_crash_reporter(self) -> None:
        y = np.array([0.5, 1.5, 1.0, 2.5])
        preds = np.array([0.4, 1.4, 1.1, 2.6])
        report_regression_model_perf(
            targets=y,
            columns=None,
            model_name="dummy_mean",
            model=None,
            preds=preds,
            print_report=False,
            show_perf_chart=False,
        )

    @pytest.mark.parametrize("cols", [["a", "b"], ("a", "b"), pd.Index(["a", "b"])])
    def test_list_tuple_index_all_work(self, cols) -> None:
        y = np.array([0.5, 1.5, 1.0, 2.5])
        preds = np.array([0.4, 1.4, 1.1, 2.6])
        report_regression_model_perf(
            targets=y,
            columns=cols,
            model_name="dummy_mean",
            model=None,
            preds=preds,
            print_report=False,
            show_perf_chart=False,
        )
