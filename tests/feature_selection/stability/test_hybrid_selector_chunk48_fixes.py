"""Regression pins for the HybridSelector glue: blocked-clustering precision, augmented row labels, the vote-fallback log, the support-mask set reuse."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.hybrid_selector import HybridSelector, corr_clusters
from mlframe.utils.log_throttle import reset_throttle_counts


def test_blocked_corr_clusters_centres_in_float64_before_narrowing(monkeypatch):
    """A large-offset pair that is 0.995 correlated clusters on the blocked path exactly as on the dense float64 path."""
    monkeypatch.delenv("MLFRAME_CRIT_DTYPE_RELAXED", raising=False)
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    df = pd.DataFrame({"a": 1e7 + x, "b": 1e7 + x + 0.1 * rng.normal(size=4000), "c": rng.normal(size=4000)})
    dense = corr_clusters(df, 0.97, block_threshold=10_000)
    blocked = corr_clusters(df, 0.97, block_threshold=1)
    assert dense[1]["a"] == ["a", "b"]
    assert blocked == dense


def test_augment_keeps_the_callers_row_labels():
    """The augmented frame carries X's own index, like the unaugmented path, so it lines up with a y that keeps its labels."""
    h = HybridSelector()
    h._eng_names, h._eng_rename, h._mrmr_member = [], {}, None
    h._tree_prod_names_, h._tree_op_ = ["tmul_0"], {"tmul_0": ("a", "b", "mul")}
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=[30, 10, 20])
    out = h._augment(X)
    assert "tmul_0" in out.columns
    assert list(out.index) == [30, 10, 20]


def test_combine_union_fallback_warns(caplog):
    """When no cluster reaches ``vote`` the clusters with the most votes are kept (a tie keeps all of them), logged at WARNING (audit 24-F42)."""
    reset_throttle_counts()
    h = HybridSelector(vote=3)
    h.members_ = {"a": ["a"], "b": ["b"]}
    h.cluster_of_ = {"a": "a", "b": "b"}
    h.fi_ = {"a": 1.0, "b": 1.0}
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.hybrid_selector"):
        out = h._combine({"mrmr": ["a"], "shap": ["b"]}, ["a", "b"])
    assert sorted(out) == ["a", "b"]
    assert any("keeping the" in r.getMessage() and "most votes" in r.getMessage() for r in caplog.records)


def test_get_support_builds_the_selected_set_once():
    """The membership set behind the support mask is built once, not once per input column."""

    class _CountingList(list):
        """A list that counts how often it is iterated."""

        n_iter = 0

        def __iter__(self):
            """Count the iteration, then iterate normally."""
            type(self).n_iter += 1
            return super().__iter__()

    h = HybridSelector()
    h.feature_names_in_ = [f"f{i}" for i in range(50)]
    h.raw_selected_ = _CountingList(["f1", "f7"])
    mask = h.get_support()
    assert mask.sum() == 2
    assert _CountingList.n_iter == 1
