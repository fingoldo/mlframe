"""Regression tests for the chunk-13 audit fixes in the FE helper modules (packer, prerank, retention, matrix io, MI contract, anchors)."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fresh_throttles():
    """Forget the per-key log counts so each test sees its own warning."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def test_quantize_mi_tiebreak_leaves_infinity_unquantised():
    """round() on an infinite MI used to raise OverflowError."""
    from mlframe.feature_selection.filters._fe_mi_contract import quantize_mi_tiebreak

    assert quantize_mi_tiebreak(float("inf")) == float("inf")
    assert quantize_mi_tiebreak(float("-inf")) == float("-inf")


def test_numpy_feature_matrix_round_trip_keeps_infinity():
    """An infinity is a value, not missingness: the numpy round trip must not turn it into NaN."""
    from mlframe.feature_selection.filters._fe_matrix_io import from_feature_matrix, to_feature_matrix

    arr = np.array([[1.0, np.inf], [2.0, -np.inf], [np.nan, 3.0]], dtype=np.float32)
    out = from_feature_matrix(to_feature_matrix(arr))
    assert np.isposinf(out[0, 1]) and np.isneginf(out[1, 1])
    assert np.isnan(out[2, 0])


def test_packer_warns_when_ortools_missing(monkeypatch, caplog):
    """The CP-SAT to greedy fallback must be logged at WARNING, not debug."""
    from mlframe.feature_selection.filters._fe_gpu_batch import _packer

    monkeypatch.setitem(sys.modules, "ortools", None)
    monkeypatch.setitem(sys.modules, "ortools.sat", None)
    monkeypatch.setitem(sys.modules, "ortools.sat.python", None)
    with caplog.at_level(logging.WARNING):
        res = _packer.pack_blocks_to_devices([3, 3, 2, 2, 2], [1.0, 1.0])
    assert len(res) == 5
    assert any("greedy" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_gbm_propensity_failure_is_warned(monkeypatch, caplog):
    """A failing booster silently zeroed the fused propensity ingredient at debug level."""
    lgb = pytest.importorskip("lightgbm")
    from mlframe.feature_selection.filters._fe_interaction_prerank import gbm_split_propensity

    def _boom(*a, **k):
        """Stand-in for lgb.train that always fails."""
        raise RuntimeError("boom")

    monkeypatch.setattr(lgb, "train", _boom)
    rng = np.random.default_rng(0)
    with caplog.at_level(logging.WARNING):
        out = gbm_split_propensity(rng.standard_normal((50, 3)), (rng.random(50) < 0.5).astype(int))
    assert not out.any()
    assert any(r.levelno == logging.WARNING and "gbm" in r.getMessage() for r in caplog.records)


def test_linear_explainability_probe_failure_is_warned(monkeypatch, caplog):
    """A failed probe falls back to the slow nonlinear passes; that must be visible at WARNING."""
    import mlframe.feature_selection.filters._fe_accuracy_gate as gate
    from mlframe.feature_selection.filters._fe_linear_explainability import raws_linearly_explain_y

    def _boom(*a, **k):
        """Stand-in for infer_classification that always fails."""
        raise RuntimeError("boom")

    monkeypatch.setattr(gate, "infer_classification", _boom)
    X = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0)[::-1]})
    with caplog.at_level(logging.WARNING):
        assert raws_linearly_explain_y(X, np.arange(40.0)) is False
    assert any(r.levelno == logging.WARNING and "probe failed" in r.getMessage() for r in caplog.records)


def test_raws_gate_does_not_skip_on_overfit_wide_noise():
    """With nearly as many columns as rows an in-sample R^2 of pure noise clears 0.92 and silently skipped the retention pass."""
    from mlframe.feature_selection.filters._fe_pure_form_retention import _raws_already_explain_y

    rng = np.random.default_rng(0)
    n, p = 300, 290
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"c{i}" for i in range(p)])
    y = rng.standard_normal(n)
    assert _raws_already_explain_y(X, y, list(X.columns), False, 0) is False


def test_raws_gate_still_skips_on_linear_target():
    """The held-out gate must keep skipping a genuinely linear target."""
    from mlframe.feature_selection.filters._fe_pure_form_retention import _raws_already_explain_y

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((400, 4)), columns=list("abcd"))
    y = X.to_numpy() @ np.array([1.0, 2.0, -1.0, 0.5]) + 0.01 * rng.standard_normal(400)
    assert _raws_already_explain_y(X, y, list(X.columns), False, 0) is True


def test_base_trim_select_ignores_all_nan_column_in_std_ranking():
    """An all-NaN column has a NaN std that poisoned the sort key and let the column into the trimmed base."""
    from mlframe.feature_selection.filters._fe_pure_form_retention import _base_trim_select

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"b": np.full(50, np.nan), "a": 2.0 * rng.standard_normal(50), "c": rng.standard_normal(50)})
    out = _base_trim_select(SimpleNamespace(), X, ["b", "a", "c"], 2)
    assert out == ["a", "c"]


def test_group_leak_probe_judges_full_rows_when_probe_is_subsampled(monkeypatch):
    """The group segments index the full frame; a probe on a row subsample used to return False and re-admit a between-group leak."""
    import mlframe.feature_selection.filters.info_theory.shared as shared
    from mlframe.feature_selection.filters._fe_pure_form_retention import _is_group_leak_raw

    n_full, n_sub = 100, 30
    seen = {}

    def _verdict(codes_x, codes_y, *a, **k):
        """Record the length the verdict was formed on and flag a leak."""
        seen["n"] = len(codes_x)
        return None, True

    monkeypatch.setattr(shared, "get_group_mi", lambda: (np.arange(n_full), np.array([0, n_full]), 1, False))
    monkeypatch.setattr(shared, "group_leak_verdict", _verdict)
    rng = np.random.default_rng(0)
    X_full = pd.DataFrame({"a": rng.standard_normal(n_full)})
    y_full = rng.standard_normal(n_full)
    assert _is_group_leak_raw(X_full.iloc[:n_sub], "a", y_full[:n_sub], X_full=X_full, y_full=y_full) is True
    assert seen["n"] == n_full


def test_pure_form_backend_choice_failure_is_warned(monkeypatch, caplog):
    """A failing kernel-tuning choose() changes the backend; it must be logged at WARNING."""
    from mlframe.feature_selection.filters import _fe_pure_form_retention_gpu_resident as mod

    class _Spec:
        """Spec whose choose() always fails."""

        def choose(self, **k):
            """Raise as a broken registry would."""
            raise RuntimeError("boom")

    monkeypatch.setattr(mod, "_SPEC", _Spec())
    with caplog.at_level(logging.WARNING):
        assert mod.pure_form_device_chosen(1000) is False
    assert any(r.levelno == logging.WARNING and "choose" in r.getMessage() for r in caplog.records)


def test_raw_redundancy_resident_binning_failure_is_warned(monkeypatch, caplog):
    """The resident to host binning fallback in the anchor builder was debug-only."""
    import mlframe.feature_selection.filters._mi_greedy_cmi_fe as cmi
    from mlframe.feature_selection.filters._fe_raw_redundancy_anchors import _build_raw_redundancy_step1_def_dev_cont

    def _boom(*a, **k):
        """Stand-in for the resident binner that always fails."""
        raise RuntimeError("boom")

    monkeypatch.setattr(cmi, "_quantile_bin_gpu_resident", _boom)
    fn = _build_raw_redundancy_step1_def_dev_cont(True)
    with caplog.at_level(logging.WARNING):
        assert fn(np.arange(20, dtype=float), 4) is None
    assert any(r.levelno == logging.WARNING and "quantile-bin" in r.getMessage() for r in caplog.records)
