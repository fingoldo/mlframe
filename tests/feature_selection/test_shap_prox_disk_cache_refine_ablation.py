"""iter81 cache wiring tests for within_cluster_refine + importance_topk_ablation.

iter80 wired the cross-process disk cache through ``proxy_trust_guard`` and ``revalidate_top_n``.
iter81 extends the same infrastructure to the remaining two honest-retrain stages:
``within_cluster_refine`` (cluster-collapse probes + per-round single-drop trials) and
``importance_topk_ablation`` (two full-template honest retrains: proxy subset vs SHAP-importance
top-k subset). Both stages call ``_honest_loss`` / ``_parallel_honest_losses`` repeatedly with the
same (X, y, cols, template, cap) tuple across fits on hyperparam sweeps; a warm-cache lookup
short-circuits the booster fit.

Coverage:
  1. Default ``disk_cache_dir=None`` -> refine + ablation behaviour bit-identical vs no-cache path.
  2. ``within_cluster_refine`` cache hit: second call with the same cache_dir returns the same
     refined subset AND populates ``honest_loss_*`` entries in the dir.
  3. ``importance_topk_ablation`` cache hit: bit-identical proxy + importance honest losses across
     cold/warm.
  4. ShapProxiedFS e2e with ``cluster_features=True``: chosen subset bit-identical across two
     fits sharing a ``cache_dir``.

These tests deliberately stay small (n_rows ~500, n_features ~12) so CI walls stay short; the C3
bench is responsible for the absolute speedup numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _make_clf_data(n=500, f=12, seed=0):
    """Small linear-signal classification frame; first 3 columns informative, rest noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float64)
    logits = X[:, 0] * 1.4 - X[:, 1] * 0.9 + X[:, 2] * 0.6
    y = (logits + rng.standard_normal(n) * 0.4 > 0).astype(np.int64)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(f)])
    return Xdf, y


# -------------------- 1. default disk_cache_dir=None conservation --------------------


def test_refine_ablation_default_no_cache_dir_identical():
    """``disk_cache_dir=None`` (default) -> two back-to-back calls produce bit-identical results
    for both ``within_cluster_refine`` and ``importance_topk_ablation``. Conservation invariant: the
    new kwarg must NOT perturb the legacy in-memory-only contract."""
    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection._shap_proxy_revalidate import (
        importance_topk_ablation, within_cluster_refine,
    )

    X, y = _make_clf_data(n=300, f=8)
    n_search = 220
    X_search, X_hold = X.iloc[:n_search].reset_index(drop=True), X.iloc[n_search:].reset_index(drop=True)
    y_search, y_hold = y[:n_search], y[n_search:]
    tpl = make_default_estimator(classification=True, random_state=0, n_estimators=30)
    # Two synthetic clusters so refine's stage-1 cluster-collapse path fires meaningfully.
    member_cols = [0, 1, 2, 3, 4, 5]
    member_groups = [[0, 1, 2], [3, 4, 5]]

    refined_a = within_cluster_refine(
        member_cols, tpl, X_search, y_search, X_hold, y_hold,
        classification=True, n_jobs=1, member_groups=member_groups, refine_n_estimators=20,
    )
    refined_b = within_cluster_refine(
        member_cols, tpl, X_search, y_search, X_hold, y_hold,
        classification=True, n_jobs=1, member_groups=member_groups, refine_n_estimators=20,
    )
    assert tuple(refined_a) == tuple(refined_b)

    # Synthesise a phi for the ablation API (it just consumes column-wise mean |phi|).
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((50, len(member_cols)))
    proxy_best_idx = (0, 1, 2)

    rep_a = importance_topk_ablation(
        phi, proxy_best_idx, tpl, X_search, y_search, X_hold, y_hold,
        classification=True,
    )
    rep_b = importance_topk_ablation(
        phi, proxy_best_idx, tpl, X_search, y_search, X_hold, y_hold,
        classification=True,
    )
    assert rep_a["proxy_honest_loss"] == rep_b["proxy_honest_loss"]
    assert rep_a["importance_honest_loss"] == rep_b["importance_honest_loss"]
    assert rep_a["proxy_features"] == rep_b["proxy_features"]
    assert rep_a["importance_features"] == rep_b["importance_features"]


# -------------------- 2. within_cluster_refine cache hit --------------------


def test_within_cluster_refine_cache_hit_identical_subset_and_populates_dir(tmp_path: Path):
    """Two ``within_cluster_refine`` calls with the same ``disk_cache_dir`` -> bit-identical refined
    subset and ``honest_loss_*`` entries populated by the first call. The cache-key contract
    (cols + seed + template + cap) means the second call's per-trial losses are served from disk."""
    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection._shap_proxy_revalidate import within_cluster_refine

    X, y = _make_clf_data(n=300, f=8)
    n_search = 220
    X_search, X_hold = X.iloc[:n_search].reset_index(drop=True), X.iloc[n_search:].reset_index(drop=True)
    y_search, y_hold = y[:n_search], y[n_search:]
    tpl = make_default_estimator(classification=True, random_state=0, n_estimators=30)
    member_cols = [0, 1, 2, 3, 4, 5]
    member_groups = [[0, 1, 2], [3, 4, 5]]
    cache_dir = tmp_path / "refine_cache"

    refined_cold = within_cluster_refine(
        member_cols, tpl, X_search, y_search, X_hold, y_hold,
        classification=True, n_jobs=1, member_groups=member_groups, refine_n_estimators=20,
        disk_cache_dir=cache_dir,
    )
    # First call (miss) MUST have populated the cache_dir with honest_loss_ entries.
    files = list(cache_dir.iterdir())
    assert len(files) >= 1, "refine cache_dir should have honest_loss_ entries after the cold call"
    assert all(f.name.startswith("honest_loss_") for f in files), (
        f"all refine entries should be in the honest_loss_ namespace, got {[f.name for f in files]}"
    )

    refined_warm = within_cluster_refine(
        member_cols, tpl, X_search, y_search, X_hold, y_hold,
        classification=True, n_jobs=1, member_groups=member_groups, refine_n_estimators=20,
        disk_cache_dir=cache_dir,
    )
    # Refined subset bit-identical (cache hits returned same per-trial floats).
    assert tuple(refined_cold) == tuple(refined_warm)


# -------------------- 3. importance_topk_ablation cache hit --------------------


def test_importance_topk_ablation_cache_hit_identical_losses(tmp_path: Path):
    """Two ``importance_topk_ablation`` calls with the same ``disk_cache_dir`` -> bit-identical
    proxy + importance honest losses, and cache directory populated with ``honest_loss_*`` entries."""
    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection._shap_proxy_revalidate import importance_topk_ablation

    X, y = _make_clf_data(n=300, f=8)
    n_search = 220
    X_search, X_hold = X.iloc[:n_search].reset_index(drop=True), X.iloc[n_search:].reset_index(drop=True)
    y_search, y_hold = y[:n_search], y[n_search:]
    tpl = make_default_estimator(classification=True, random_state=0, n_estimators=30)
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((50, 8))
    proxy_best_idx = (0, 1, 2)
    cache_dir = tmp_path / "ablation_cache"

    rep_a = importance_topk_ablation(
        phi, proxy_best_idx, tpl, X_search, y_search, X_hold, y_hold,
        classification=True, disk_cache_dir=cache_dir,
    )
    files = list(cache_dir.iterdir())
    assert len(files) >= 1, "ablation cache_dir should have honest_loss_ entries after cold call"
    assert all(f.name.startswith("honest_loss_") for f in files), (
        f"all ablation entries should be in the honest_loss_ namespace, got {[f.name for f in files]}"
    )

    rep_b = importance_topk_ablation(
        phi, proxy_best_idx, tpl, X_search, y_search, X_hold, y_hold,
        classification=True, disk_cache_dir=cache_dir,
    )
    # Bit-identical losses (cache hit served the same float).
    assert rep_a["proxy_honest_loss"] == rep_b["proxy_honest_loss"]
    assert rep_a["importance_honest_loss"] == rep_b["importance_honest_loss"]
    assert rep_a["proxy_features"] == rep_b["proxy_features"]
    assert rep_a["importance_features"] == rep_b["importance_features"]


# -------------------- 4. ShapProxiedFS e2e identical subset across two fits --------------------


def test_shapproxiedfs_two_fit_cache_dir_refine_path_identical_subset(tmp_path: Path):
    """End-to-end: ShapProxiedFS with ``cluster_features=True`` and ``cache_dir`` -> two back-to-back
    fits produce bit-identical chosen subset. Validates the refine + ablation cache wiring at the
    facade level (cluster_features forces ``within_cluster_refine`` to fire on the chosen units)."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _make_clf_data()
    cache_dir = tmp_path / "e2e_refine_cache"

    def _fit():
        sel = ShapProxiedFS(
            classification=True, metric="brier", optimizer="bruteforce",
            max_features=4, top_n=6, n_splits=3, n_revalidation_models=1,
            trust_guard=True, n_anchors=6, revalidate=True,
            cluster_features=True, run_importance_ablation=True,
            within_cluster_refine=True,
            random_state=0, verbose=False, n_jobs=1,
            cache_dir=str(cache_dir),
        )
        sel.fit(X, y)
        return sel

    sel_a = _fit()
    sel_b = _fit()
    np.testing.assert_array_equal(sel_a.support_, sel_b.support_)
    assert tuple(sel_a.selected_features_) == tuple(sel_b.selected_features_)
    # Cache should have honest_loss_ entries written by refine + ablation paths.
    all_files = [f.name for f in Path(cache_dir).iterdir()]
    assert any(n.startswith("honest_loss_") for n in all_files), (
        f"refine/ablation should have written honest_loss_ entries; got {all_files}"
    )
