"""Unit tests for BorutaShap premerge_clusters (round-3 B3-1): collapse raw-corr clusters to one representative
before the shadow gate, then re-expand accepted reps to their members.

Validates: the option is a real constructor param (off by default); under premerge the public outputs (support_ /
selected_features_ / feature_names_in_) are aligned to the ORIGINAL input columns (not the collapsed reps); the
signal is recovered; and re-expansion actually re-includes redundant cluster members of an accepted representative.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from sklearn.ensemble import RandomForestClassifier


def _redundant_data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    y = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-(z @ np.array([1.5, -1.2, 1.0]))))).astype(int))
    cols = {f"inf_{i}": z[:, i] for i in range(3)}
    for j in range(4):  # 4 near-duplicate copies of inf_0 -> one tight cluster
        cols[f"red_0_{j}"] = z[:, 0] + 0.05 * rng.standard_normal(n)
    for k in range(6):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), y


def _fit(premerge):
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y = _redundant_data()
    b = BorutaShap(
        model=RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=0),
        importance_measure="gini",
        classification=True,
        n_trials=30,
        percentile=95,
        verbose=False,
        random_state=0,
        premerge_clusters=premerge,
    )
    b.fit(X, y)
    return b, X


def test_premerge_is_an_off_by_default_constructor_option():
    from mlframe.feature_selection.boruta_shap import BorutaShap

    p = inspect.signature(BorutaShap.__init__).parameters
    assert p["premerge_clusters"].default is False and "premerge_corr_thr" in p
    b = BorutaShap(premerge_clusters=True, premerge_corr_thr=0.9)
    assert b.premerge_clusters is True and b.premerge_corr_thr == 0.9


def test_premerge_outputs_aligned_to_original_columns_and_recovers_signal():
    b, X = _fit(premerge=True)
    # public outputs span the ORIGINAL columns, not the collapsed representatives
    assert len(b.support_) == X.shape[1]
    assert list(b.feature_names_in_) == list(X.columns)
    assert all(c in X.columns for c in b.selected_features_)
    # signal recovered (the 3 informative latents)
    assert sum(f"inf_{i}" in set(b.selected_features_) for i in range(3)) >= 2


def test_premerge_reexpands_accepted_cluster_members():
    """When the inf_0 representative is accepted, re-expansion re-includes its redundant copies, so the selected
    set contains MORE than one member of that cluster (the mechanism vs plain Boruta which would keep just one)."""
    b, X = _fit(premerge=True)
    inf0_cluster = {"inf_0", "red_0_0", "red_0_1", "red_0_2", "red_0_3"}
    sel = set(b.selected_features_)
    if inf0_cluster & sel:  # the cluster was accepted
        assert len(inf0_cluster & sel) >= 2, "accepted cluster should re-expand to multiple members"
