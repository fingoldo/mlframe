"""Cluster 41 regression: one NaN in y no longer disables the clustered-target check (audit 2026-11-20, finding 41-F11)."""

import numpy as np

from mlframe.training.targets._target_distribution_analyzer_classif import _group_ids_pathologies


def test_clustered_target_is_still_reported_with_one_nan_label():
    """Targets that are constant within each of 20 groups, with one NaN row: the pathology and the group-aware hint are recorded."""
    rng = np.random.default_rng(0)
    gids = np.repeat(np.arange(20), 25)
    y = gids * 1.0 + rng.normal(0, 0.01, gids.size)
    y[3] = np.nan
    finite = np.isfinite(y)
    pathologies, knobs, diagnostics = [], {}, {}
    _group_ids_pathologies(gids, y, finite, y.size, y[finite], diagnostics, pathologies, knobs, lambda *a, **k: None)
    assert any(p.startswith("clustered_target") for p in pathologies), pathologies
    assert knobs["split_config"]["prefer_group_aware"] is True
    assert "within_between_group_var_ratio" in diagnostics
