"""Batch 4: dynamic-cluster, permutation-null and stability fallbacks log at WARNING.

Each test forces the failure on the path the fail-open gate flags and asserts a WARNING record. The stability
handlers skip a bootstrap or pair and the permutation handler changes the subsample cap, so their messages say the result
is degraded. At DEBUG the same records are dropped by default logging, so these tests fail on the old code.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import numpy as np


def _warnings_mentioning(caplog, needle: str) -> list[logging.LogRecord]:
    """Return WARNING-or-higher records whose message contains needle."""
    return [r for r in caplog.records if r.levelno >= logging.WARNING and needle in r.getMessage()]


class _Unindexable:
    """A factors_data stand-in whose indexing always raises."""

    def __getitem__(self, _key):
        """Raise to force the target-column read to fail."""
        raise RuntimeError("factors_data unreadable")


def test_select_swap_method_auto_warns_when_target_column_unreadable(caplog):
    """An unreadable target column with no explicit target falls back to pca_pc1 and warns that the result is degraded."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import _dcd_swap as mod

    state = SimpleNamespace(target_indices=np.array([0]), factors_data=_Unindexable())
    Z = np.random.default_rng(0).normal(size=(50, 3))
    with caplog.at_level(logging.DEBUG):
        method, _params = mod._select_swap_method_auto(state=state, Z=Z, target_y=None, member_names=("a", "b", "c"))
    assert method == "pca_pc1"
    assert _warnings_mentioning(caplog, "result degraded to the pca_pc1 method")


def test_calibrate_tau_auto_warns_when_batched_su_fails(monkeypatch, caplog):
    """A failing batched SU call falls back to the per-pair loop and the warning is logged."""
    from mlframe.feature_selection.filters import _dcd_pair_su_batch as batch_mod
    from mlframe.feature_selection.filters import _dcd_tau_auto as mod

    def _boom(*_a, **_k):
        """Raise to force the fallback."""
        raise RuntimeError("batch SU broken")

    monkeypatch.setattr(batch_mod, "pair_su_batch", _boom)
    data = np.random.default_rng(3).integers(0, 3, size=(300, 8)).astype(np.int32)
    with caplog.at_level(logging.DEBUG):
        tau, diagnostics = mod._calibrate_tau_auto(
            factors_data=data,
            factors_nbins=np.full(8, 3, dtype=np.int64),
            distance="su",
            n_pairs=30,
            seed=0,
        )
    assert np.isfinite(tau)
    assert diagnostics["n_pairs_finite"] > 0
    assert _warnings_mentioning(caplog, "pair_su_batch failed, falling back to the per-pair loop")


def test_pair_su_batch_warns_when_batched_joint_entropy_fails(monkeypatch, caplog):
    """A failing batched joint-entropy validation falls back to serial joints and logs a warning."""
    from mlframe.feature_selection.filters import _dcd_pair_su_batch as mod
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import DCDState

    def _boom(*_a, **_k):
        """Raise to force the fallback."""
        raise RuntimeError("validation broken")

    monkeypatch.setattr(mod, "_validate_batch_joint_entropy_pairs_inputs", _boom)
    data = np.random.default_rng(4).integers(0, 3, size=(300, 6)).astype(np.int32)
    state = DCDState(
        pool_pruned_mask=np.zeros(6, dtype=bool),
        factors_data=data,
        factors_nbins=np.full(6, 3, dtype=np.int64),
        cols=[],
        nbins=np.full(6, 3, dtype=np.int64),
        distance="su",
    )
    pairs = [(0, 1), (1, 2), (2, 3)]
    with caplog.at_level(logging.DEBUG):
        scores = mod.pair_su_batch(state, pairs)
    assert len(scores) == len(pairs)
    assert _warnings_mentioning(caplog, "batched joint-entropy computation failed")


def test_default_pair_maxt_max_rows_warns_when_feature_engineering_import_fails(monkeypatch, caplog):
    """A failing feature_engineering import uses the bootstrap cap of 30000 rows, halved, and warns that the result is degraded."""
    from mlframe.feature_selection.filters import _permutation_null as mod

    monkeypatch.setitem(sys.modules, "mlframe.feature_selection.filters.feature_engineering", None)
    with caplog.at_level(logging.DEBUG):
        cap = mod._default_pair_maxt_max_rows()
    assert cap == 15000
    assert _warnings_mentioning(caplog, "result degraded to the bootstrap cap of 30000 rows")


def test_cluster_stability_selection_warns_when_a_bootstrap_selector_raises(caplog):
    """A selector that raises on one bootstrap skips that bootstrap and logs a degraded-result warning."""
    from mlframe.feature_selection.filters._stability_cluster import cluster_stability_selection

    calls = {"n": 0}

    def _selector(X_sub, y_sub):
        """Raise on the first bootstrap, then select the first column."""
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("selector broken once")
        return np.array([0])

    X = np.random.default_rng(5).normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    with caplog.at_level(logging.DEBUG):
        cluster_stability_selection(X, y, _selector, n_bootstrap=5)
    assert _warnings_mentioning(caplog, "bootstrap skipped (result degraded)")


def test_complementary_pairs_stability_warns_when_a_pair_selector_raises(caplog):
    """A selector that raises on one pair skips that pair and logs a degraded-result warning."""
    from mlframe.feature_selection.filters._stability_cluster import complementary_pairs_stability

    calls = {"n": 0}

    def _selector(X_sub, y_sub):
        """Raise on the first pair, then select the first column."""
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("selector broken once")
        return np.array([0])

    X = np.random.default_rng(6).normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    with caplog.at_level(logging.DEBUG):
        complementary_pairs_stability(X, y, _selector, n_pairs=4)
    assert _warnings_mentioning(caplog, "pair skipped (result degraded)")
