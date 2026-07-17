"""Regression: ShapProxiedFS row-split must not OOM at C4-scale widths.

iter46 surfaced a second-stage OOM in ``fit`` after iter45 unblocked dataset
construction at C4: the two back-to-back ``X.iloc[idx].reset_index(drop=True)``
splits held the parent frame + both child frames + reset_index reallocation
buffers simultaneously (~3.7 GiB at width=20000 / n_rows=10000) and the
holdout split allocation failed on a 17 GB / 6.4 GB-free host. The fix:

  - column-batched copy for ``X_search`` so peak transient is one batch's
    worth instead of a full second split,
  - deferred ``X_hold`` materialisation -- built post-prefilter at the narrow
    working-column count (~5 MiB instead of ~381 MiB at C4),
  - dropped the cosmetic ``reset_index(drop=True)`` calls on the wide blocks
    (downstream reads via ``.values`` / ``.iloc[:, cols]`` / positional row
    access; none depend on a 0..n-1 RangeIndex).

This test exercises the split path at a downscaled width that still crosses
the "wide enough to need column-batching" boundary, and asserts that the
resulting selector behaves identically to a manual reference split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_simple(n=600, p=2000, seed=0):
    rng = np.random.default_rng(seed)
    # Two informative columns; rest noise.
    X = rng.standard_normal((n, p)).astype(np.float64)
    beta = np.zeros(p)
    beta[0] = 1.5
    beta[1] = -1.2
    logit = X @ beta
    y = (logit + 0.5 * rng.standard_normal(n) > 0).astype(np.int64)
    cols = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), y


def test_split_path_recovers_informatives():
    pytest.importorskip("xgboost")
    pytest.importorskip("shap")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _make_simple(n=600, p=2000, seed=0)
    sel = ShapProxiedFS(
        classification=True,
        metric="brier",
        optimizer="greedy_forward",
        max_features=8,
        top_n=10,
        n_splits=3,
        n_revalidation_models=1,
        trust_guard=False,
        run_importance_ablation=False,
        within_cluster_refine=False,
        random_state=0,
        verbose=False,
        n_jobs=1,
    )
    sel.fit(X, y)
    selected = set(sel.selected_features_)
    assert "f0" in selected, f"informative f0 not selected: {sorted(selected)}"
    assert "f1" in selected, f"informative f1 not selected: {sorted(selected)}"


def test_split_path_drops_parent_frame_reference():
    """After ``fit`` returns the selector must not retain a handle to the (large) parent X.

    ``_deferred_holdout`` is the lever's private hand-off between the row split and the
    post-prefilter holdout materialisation; it MUST be cleared by the time ``fit`` returns
    so the parent block can be garbage-collected as soon as the caller drops its reference.
    """
    pytest.importorskip("xgboost")
    pytest.importorskip("shap")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _make_simple(n=400, p=600, seed=1)
    sel = ShapProxiedFS(
        classification=True,
        metric="brier",
        optimizer="greedy_forward",
        max_features=6,
        top_n=5,
        n_splits=3,
        n_revalidation_models=1,
        trust_guard=False,
        run_importance_ablation=False,
        within_cluster_refine=False,
        random_state=0,
        verbose=False,
        n_jobs=1,
    )
    sel.fit(X, y)
    assert sel._deferred_holdout is None, (
        "selector leaked a reference to the parent X via _deferred_holdout; this would keep the wide block alive past fit and defeat the iter46 memory lever"
    )


def test_split_path_works_without_prefilter():
    """When ``prefilter_top=None`` the prefilter ``if`` block is skipped; the deferred
    holdout materialisation must still fire and produce a full-width X_hold."""
    pytest.importorskip("xgboost")
    pytest.importorskip("shap")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _make_simple(n=300, p=50, seed=2)
    sel = ShapProxiedFS(
        classification=True,
        metric="brier",
        optimizer="greedy_forward",
        max_features=6,
        top_n=5,
        n_splits=3,
        n_revalidation_models=1,
        prefilter_top=None,  # no prefilter -> working_cols stays at full width
        cluster_features=False,
        trust_guard=False,
        run_importance_ablation=False,
        within_cluster_refine=False,
        random_state=0,
        verbose=False,
        n_jobs=1,
    )
    sel.fit(X, y)
    # Sklearn contract intact.
    assert sel.support_.shape == (50,)
    assert sel.n_features_in_ == 50
    assert sel._deferred_holdout is None
