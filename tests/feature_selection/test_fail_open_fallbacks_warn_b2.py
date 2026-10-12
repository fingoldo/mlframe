"""Batch 2: fallbacks in the adaptive-bin and dynamic-cluster paths log at WARNING.

Each test forces the failure on the path the fail-open gate flags and asserts a WARNING record. The two dynamic-cluster
swap fallbacks change the member relevances they return, so their warnings say the result is degraded. At DEBUG the same
records are dropped by default logging, so these tests fail on the old code.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import numpy as np


def _warnings_mentioning(caplog, needle: str) -> list[logging.LogRecord]:
    """Return WARNING-or-higher records whose message contains needle."""
    return [r for r in caplog.records if r.levelno >= logging.WARNING and needle in r.getMessage()]


def _x(n: int = 200, p: int = 3) -> np.ndarray:
    """Return a deterministic float matrix for the edge computations."""
    return np.random.default_rng(0).normal(size=(n, p))


def test_per_feature_edges_warns_when_cache_setup_fails(monkeypatch, caplog, tmp_path):
    """A failing cache setup disables the cache and warns."""
    from mlframe.feature_selection.filters import _adaptive_nbins as mod
    from mlframe.utils import disk_cache

    def _boom(*_a, **_k):
        """Raise to force the fallback."""
        raise RuntimeError("hash broken")

    monkeypatch.setattr(disk_cache, "hash_object", _boom)
    with caplog.at_level(logging.DEBUG):
        edges = mod.per_feature_edges(_x(), None, method="freedman_diaconis", cache_dir=str(tmp_path), n_jobs=1)
    assert len(edges) == 3
    assert _warnings_mentioning(caplog, "cache disabled")


def test_per_feature_edges_warns_when_cache_get_fails(monkeypatch, caplog, tmp_path):
    """A failing per-column cache lookup computes that column and warns."""
    from mlframe.feature_selection.filters import _adaptive_nbins as mod
    from mlframe.utils import disk_cache

    def _boom(*_a, **_k):
        """Raise to force the fallback."""
        raise RuntimeError("cache read broken")

    monkeypatch.setattr(disk_cache.DiskCache, "get", _boom)
    with caplog.at_level(logging.DEBUG):
        edges = mod.per_feature_edges(_x(), None, method="freedman_diaconis", cache_dir=str(tmp_path), n_jobs=1)
    assert len(edges) == 3
    assert _warnings_mentioning(caplog, "cache get failed")


def test_per_feature_edges_warns_when_psutil_is_unavailable(monkeypatch, caplog):
    """With psutil missing, the thread count falls back to os.cpu_count and warns."""
    from mlframe.feature_selection.filters import _adaptive_nbins as mod

    monkeypatch.setitem(sys.modules, "psutil", None)  # makes `import psutil` raise ImportError
    with caplog.at_level(logging.DEBUG):
        edges = mod.per_feature_edges(_x(), None, method="freedman_diaconis", n_jobs=-1)
    assert len(edges) == 3
    assert _warnings_mentioning(caplog, "psutil.cpu_count(logical=False) failed")


def test_binarize_aggregate_warns_and_degrades_when_quantile_fails(monkeypatch, caplog):
    """A failing quantile binning falls back to min-max bins and says the result is degraded."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import _dcd_metrics as mod

    def _boom(*_a, **_k):
        """Raise to force the fallback."""
        raise RuntimeError("quantile broken")

    monkeypatch.setattr(np, "quantile", _boom)
    values = np.arange(20.0)
    with caplog.at_level(logging.DEBUG):
        codes = mod._binarize_aggregate(values, method="quantile", n_bins=4, dtype=np.int64)
    assert codes.min() == 0 and codes.max() == 3
    assert _warnings_mentioning(caplog, "quantile-edge binning failed, result degraded")


def _swap_state(data: np.ndarray) -> SimpleNamespace:
    """Return the minimal state the per-member relevance block reads."""
    return SimpleNamespace(factors_data=data, factors_nbins=np.array([3, 3, 3], dtype=np.int64))


def test_swap_per_member_relevance_warns_when_batched_path_fails(monkeypatch, caplog):
    """A failing batched member-relevance call falls back to the per-member path and warns."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import _dcd_swap as mod

    monkeypatch.setitem(sys.modules, "mlframe.feature_selection.filters._dynamic_cluster_discovery._dcd_member_rank_batch", None)
    data = np.random.default_rng(1).integers(0, 3, size=(200, 3)).astype(np.int32)
    target = data[:, 2].astype(np.int64)
    with caplog.at_level(logging.DEBUG):
        best_idx, best_rel = mod._evaluate_swap_cand_its_own_per_member(
            _swap_state(data),
            [0, 1],
            target,
            [],
            {},
            float("-inf"),
            {},
            -1,
        )
    assert best_idx in (0, 1)
    assert np.isfinite(best_rel)
    assert _warnings_mentioning(caplog, "member-relevance computation failed, result degraded")


def test_swap_per_member_relevance_warns_and_records_zero_when_member_fails(monkeypatch, caplog):
    """A member whose relevance cannot be computed is recorded as 0.0 and the degradation is logged."""
    from mlframe.feature_selection.filters import info_theory
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import _dcd_swap as mod

    def _boom(*_a, **_k):
        """Raise to force the fallback."""
        raise RuntimeError("mi broken")

    monkeypatch.setitem(sys.modules, "mlframe.feature_selection.filters._dynamic_cluster_discovery._dcd_member_rank_batch", None)
    monkeypatch.setattr(info_theory, "mi", _boom)
    data = np.random.default_rng(2).integers(0, 3, size=(200, 3)).astype(np.int32)
    target = data[:, 2].astype(np.int64)
    with caplog.at_level(logging.DEBUG):
        best_idx, best_rel = mod._evaluate_swap_cand_its_own_per_member(
            _swap_state(data),
            [0, 1],
            target,
            [],
            {},
            float("-inf"),
            {},
            -1,
        )
    # Every member scores 0.0, so the first member wins the tie and the degraded value is what is returned.
    assert (best_idx, best_rel) == (0, 0.0)
    assert len(_warnings_mentioning(caplog, "recording 0.0")) == 2
