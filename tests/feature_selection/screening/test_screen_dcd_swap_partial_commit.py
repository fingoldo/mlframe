"""DCD discover/swap: a commit that fails after it extended the matrix still hands the extended refs back, and the failure is logged through the throttle."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.feature_selection.filters import _dynamic_cluster_discovery as dcd
from mlframe.feature_selection.filters._screen_dcd_swap import screen_dcd_discover_and_swap
from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fresh_throttle():
    """Forget throttle counts so the warning is emitted in each test."""
    reset_throttle_counts()


def test_failed_commit_after_matrix_extension_still_rebinds_refs(monkeypatch, caplog):
    """commit_swap raising after it wrote the extended matrix into data_ref: the returned refs and ctx are the extended ones, and a WARNING is logged."""
    extended = np.zeros((20, 4))
    nbins = np.array([3, 3, 3, 3])
    monkeypatch.setattr(dcd, "discover_cluster_members", lambda *a, **k: None)
    monkeypatch.setattr(dcd, "evaluate_swap_candidate", lambda *a, **k: SimpleNamespace(accept=True))

    def _commit(state, var, decision, *, selected_vars, data_ref, engineered_recipes, predictors_log):
        """Write the extended matrix, then fail."""
        data_ref["data"], data_ref["nbins"], data_ref["cols"] = extended, nbins, ["a", "b", "c", "agg"]
        raise RuntimeError("late failure")

    monkeypatch.setattr(dcd, "commit_swap", _commit)
    state = SimpleNamespace(pool_pruned_mask=np.zeros(3, dtype=bool), cluster_anchors={0: {1, 2}}, cluster_size_threshold=2)
    ctx = SimpleNamespace()
    with caplog.at_level(logging.WARNING):
        fd, fn, names, dc = screen_dcd_discover_and_swap(
            dcd_state=state,
            var=0,
            factors_data=np.zeros((20, 3)),
            factors_nbins=np.array([3, 3, 3]),
            factors_names=["a", "b", "c"],
            data_copy=np.zeros((20, 3)),
            selected_vars=[0],
            entropy_cache={},
            cached_MIs={},
            full_npermutations=3,
            y=[2],
            engineered_recipes={},
            predictors=[],
            ctx=ctx,
            verbose=False,
        )
    assert fd is extended and dc is extended and ctx.factors_data is extended and ctx.data_copy is extended
    assert list(names) == ["a", "b", "c", "agg"] and fn is nbins
    assert any("DCD discover/swap step failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
