"""Regressions for defects found reading the ShapProxiedFS fit helpers, clustering and resolver modules."""

from __future__ import annotations

import contextlib
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_cluster_su as su_mod
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit_steps import ShapProxiedFitStepsMixin
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_resolvers import _resolve_brute_force_work_budget
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster import cluster_correlated_features
from mlframe.utils.log_throttle import reset_throttle_counts


def test_refit_replays_the_seeded_rng_stream():
    """A second fit must start from the seeded generator, not continue the stream __init__ (and the first fit) advanced."""
    est = ShapProxiedFS(random_state=3)
    est._rng.random(5)
    X = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)[::-1]})
    est._fit_ingest(X, np.arange(10) % 2)
    assert est._rng.random() == np.random.default_rng(3).random()


def test_pearson_clustering_does_not_merge_independent_columns_under_a_large_offset():
    """Independent columns at 1e10 + N(0, 50) collapsed to one float32 value whose mean error read as a non-zero std, so both became +-1 constants and were merged."""
    rng = np.random.default_rng(0)
    X = np.column_stack([1e10 + 50.0 * rng.normal(size=4000), 1e10 + 50.0 * rng.normal(size=4000), 1e10 + 50.0 * rng.normal(size=4000)])
    labels = cluster_correlated_features(X, threshold=0.7, use_gpu=False)
    assert len(set(labels.tolist())) == 3


def test_pearson_clustering_does_not_turn_nan_cells_into_offset_outliers():
    """A NaN in a column centred at 1e6 must be imputed at the column mean, not at 0 (a 1e6 outlier that flattens the column)."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=3000)
    a = 1e6 + z
    a[:5] = np.nan
    X = np.column_stack([a, 1e6 + z + 0.2 * rng.normal(size=3000)])
    labels = cluster_correlated_features(X, threshold=0.7, use_gpu=False)
    assert labels[0] == labels[1]


def test_serial_su_marginal_accepts_a_negative_missing_code():
    """The serial scan crashed in np.bincount on a -1 code the packed-kernel path already shifts to bin 0."""
    cls, freqs = su_mod._column_marginal(np.array([-1, 0, 1, 1, -1]))
    assert freqs.sum() == pytest.approx(1.0)
    assert cls.min() == 0


def _force_gpu_gate(monkeypatch, free_bytes):
    """Make every gate before the memory check pass on a CPU-only host."""
    monkeypatch.setattr(su_mod, "cluster_su_gpu_available", lambda: True)
    monkeypatch.setattr(su_mod, "su_gpu_pays_off", lambda f, n: True)
    monkeypatch.setattr(su_mod, "_gpu_free_memory_bytes", lambda: free_bytes)
    reset_throttle_counts()


def test_su_gpu_memory_decline_is_logged(monkeypatch, caplog):
    """Declining the GPU kernel for lack of memory sends the scan down the slow CPU path and must say so."""
    _force_gpu_gate(monkeypatch, 1000)
    with caplog.at_level(logging.WARNING, logger=su_mod.logger.name):
        assert su_mod._should_route_su_gpu(500, 10_000, 10) is False
    assert "declined" in caplog.text and "free" in caplog.text


def test_su_gpu_row_limit_declines_and_logs(monkeypatch, caplog):
    """float32 joint counts are inexact from 2**24 rows; the GPU kernel must not run there and the decline is logged."""
    _force_gpu_gate(monkeypatch, 1 << 60)
    with caplog.at_level(logging.WARNING, logger=su_mod.logger.name):
        assert su_mod._should_route_su_gpu(500, 1 << 24, 2) is False
    assert "exact-count" in caplog.text
    _force_gpu_gate(monkeypatch, 1 << 60)
    assert su_mod._should_route_su_gpu(500, 10_000, 2) is True


def test_banzhaf_ranking_overridden_by_residual_blend_is_reported_and_logged(caplog):
    """prescreen_ranking='banzhaf' is silently dropped when a residual blend supplies the importance; the report must name the ranking really used."""
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(50, 6))
    stub = SimpleNamespace(prescreen_ranking="banzhaf")
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING):
        report: dict = {}
        _, _, kept, _ = ShapProxiedFitStepsMixin._fit_step3_prescreen_top_none(
            stub, 3, 6, lambda name: contextlib.nullcontext(), np.arange(6.0), phi, np.zeros(50), np.zeros(50), [], [], None, None, report, np.arange(6))
    assert report["prescreen"]["ranking"] == "residual_blend"
    assert list(kept[:3]) is not None and "not applied" in caplog.text


def test_target_with_nan_is_rejected_before_the_booster_sees_it():
    """A NaN in y reached the booster / numba proxy-loss kernels as a float64 label; it must raise at fit entry."""
    est = ShapProxiedFS(classification=False)
    with pytest.raises(ValueError, match="NaN/inf"):
        est._coerce_target(np.array([0.1, np.nan, 0.3]))


def test_unparseable_work_budget_env_is_logged_not_swallowed(monkeypatch, caplog):
    """A mistyped MLFRAME_SHAP_BRUTE_FORCE_WORK_BUDGET silently fell back to the default."""
    monkeypatch.setenv("MLFRAME_SHAP_BRUTE_FORCE_WORK_BUDGET", "4e9x")
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING):
        assert _resolve_brute_force_work_budget() == 4_000_000_000
    assert "not a number" in caplog.text
