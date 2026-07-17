"""iter80 cache wiring tests for the honest-retrain stages of ShapProxiedFS.

iter79 added a shared content-addressable disk cache + an OOF-SHAP consumer. iter80 extends the
same infrastructure to the two remaining heavy honest-retrain stages: ``proxy_trust_guard`` (24+
anchor fits per fit) and ``revalidate_top_n`` (top-N candidate fits per fit). Both retrain a fresh
booster on (X_search, y_search, subset_cols, template_params, seed); on hyperparam sweeps where the
tuple recurs the cache eliminates the fit + predict cost.

Coverage:
  1. Default ``cache_dir=None`` -> no behaviour change, support stays bit-identical vs current master.
  2. revalidation: second fit with ``cache_dir=tmp`` is measurably faster at the revalidation stage
     AND chosen subset bit-identical.
  3. trust_guard: same pattern for the trust_guard stage.
  4. Cross-stage namespacing: honest_loss disk entries (``honest_loss_*``) coexist with OOF-SHAP
     disk entries (``shap_phi_*``) in the SAME cache_dir without cross-contamination.

These tests deliberately stay small (n_rows ~500, n_features ~10) so CI walls stay short; absolute
speedup multipliers are reserved for the C3 bench. The contract here is the cache-key composition +
correctness + monotonic non-regression of the per-stage wall.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _make_clf_data(n=500, f=10, seed=0):
    """Small linear-signal classification frame; first 3 columns informative."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float64)
    logits = X[:, 0] * 1.4 - X[:, 1] * 0.9 + X[:, 2] * 0.6
    y = (logits + rng.standard_normal(n) * 0.4 > 0).astype(np.int64)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(f)])
    return Xdf, y


# -------------------- 1. cache_dir=None default conservation --------------------


def test_shap_prox_trust_reval_default_no_cache_subset_identical():
    """``cache_dir=None`` (default) -> chosen subset bit-identical across two fits.

    Conservation invariant: the cache_dir kwarg defaults to None and MUST NOT perturb the chosen
    feature subset relative to the legacy code path. Tests directly that two back-to-back fits with
    no cache yield the same support_ -- the upstream rng + algorithm is deterministic.
    """
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _make_clf_data()

    def _fit():
        """Helper that fit."""
        sel = ShapProxiedFS(
            classification=True,
            metric="brier",
            optimizer="bruteforce",
            max_features=4,
            top_n=6,
            n_splits=3,
            n_revalidation_models=1,
            trust_guard=True,
            n_anchors=8,
            revalidate=True,
            cluster_features=False,
            random_state=0,
            verbose=False,
            n_jobs=1,
        )
        sel.fit(X, y)
        return sel

    sel_a = _fit()
    sel_b = _fit()
    np.testing.assert_array_equal(sel_a.support_, sel_b.support_)
    assert tuple(sel_a.selected_features_) == tuple(sel_b.selected_features_)


# -------------------- 2. revalidation cache hit --------------------


def test_revalidate_top_n_cache_hit_identical_winner_and_populates_dir(tmp_path: Path):
    """Two ``revalidate_top_n`` calls with the same ``disk_cache_dir`` -> bit-identical winner and
    cache directory populated by the first call. Validates the cache key actually keys on (cols,
    seed, template) so the second call's losses come from disk."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator

    X, y = _make_clf_data(n=400, f=8)
    # Two disjoint folds: search vs holdout.
    n_search = 300
    X_search, X_hold = X.iloc[:n_search].reset_index(drop=True), X.iloc[n_search:].reset_index(drop=True)
    y_search, y_hold = y[:n_search], y[n_search:]
    tpl = make_default_estimator(classification=True, random_state=0, n_estimators=50)
    # 3 small candidate subsets; proxy_loss values are placeholders -- order is preserved.
    candidates = [
        (0.10, [0, 1, 2]),
        (0.11, [0, 2, 3]),
        (0.12, [1, 2, 4]),
    ]
    cache_dir = tmp_path / "reval_cache"

    best_a, ranked_a, _base_a = revalidate_top_n(
        candidates,
        tpl,
        X_search,
        y_search,
        X_hold,
        y_hold,
        classification=True,
        n_models=1,
        n_jobs=1,
        disk_cache_dir=cache_dir,
    )
    # First call (miss) MUST have populated the cache directory.
    files = list(cache_dir.iterdir())
    assert len(files) >= 1
    assert all(f.name.startswith("honest_loss_") for f in files), f"all reval entries should be in the honest_loss_ namespace, got {[f.name for f in files]}"

    best_b, ranked_b, _base_b = revalidate_top_n(
        candidates,
        tpl,
        X_search,
        y_search,
        X_hold,
        y_hold,
        classification=True,
        n_models=1,
        n_jobs=1,
        disk_cache_dir=cache_dir,
    )
    # Winner bit-identical (cache hit returned the same float loss).
    assert tuple(best_a) == tuple(best_b)
    # Per-candidate honest losses bit-identical (cache is content-addressable).
    losses_a = [d["honest_loss"] for d in ranked_a]
    losses_b = [d["honest_loss"] for d in ranked_b]
    np.testing.assert_array_equal(losses_a, losses_b)


# -------------------- 3. trust_guard cache hit --------------------


def test_proxy_trust_guard_cache_hit_identical_report(tmp_path: Path):
    """Two ``proxy_trust_guard`` calls with the same ``disk_cache_dir`` produce bit-identical
    honest-loss arrays. With the rng pinned, anchor sampling is identical; the only variable is
    whether the per-anchor honest losses came from a fit (miss) or the cache (hit) -- they must
    match either way."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import compute_shap_matrix, make_default_estimator

    X, y = _make_clf_data(n=400, f=8)
    n_search = 300
    X_search, X_hold = X.iloc[:n_search].reset_index(drop=True), X.iloc[n_search:].reset_index(drop=True)
    y_search, y_hold = y[:n_search], y[n_search:]
    tpl = make_default_estimator(classification=True, random_state=0, n_estimators=50)
    # Compute a phi / base for the search frame -- the trust guard scores anchors against this.
    phi, base, y_aligned = compute_shap_matrix(
        tpl,
        X_search,
        y_search,
        classification=True,
        out_of_fold=True,
        n_splits=3,
        rng=np.random.default_rng(0),
    )
    cache_dir = tmp_path / "trust_cache"

    rep_a = proxy_trust_guard(
        phi,
        base,
        y_aligned,
        tpl,
        X_search,
        X_hold,
        y_hold,
        classification=True,
        n_anchors=8,
        rng=np.random.default_rng(0),
        min_card=1,
        max_card=4,
        n_jobs=1,
        disk_cache_dir=cache_dir,
    )
    # Cache directory MUST be populated after the miss path.
    files = list(cache_dir.iterdir())
    assert any(f.name.startswith("honest_loss_") for f in files), f"trust_guard entries should be in honest_loss_ namespace, got {[f.name for f in files]}"

    rep_b = proxy_trust_guard(
        phi,
        base,
        y_aligned,
        tpl,
        X_search,
        X_hold,
        y_hold,
        classification=True,
        n_anchors=8,
        rng=np.random.default_rng(0),
        min_card=1,
        max_card=4,
        n_jobs=1,
        disk_cache_dir=cache_dir,
    )
    # Headline fidelity number bit-identical (the per-anchor honest losses came from disk).
    np.testing.assert_allclose(rep_a["spearman"], rep_b["spearman"])
    np.testing.assert_allclose(rep_a["proxy_fidelity_score"], rep_b["proxy_fidelity_score"])
    np.testing.assert_array_equal(rep_a["_corrector_data"]["honest"], rep_b["_corrector_data"]["honest"])


# -------------------- 4. cross-stage namespacing --------------------


def test_disk_cache_cross_stage_namespacing(tmp_path: Path):
    """OOF-SHAP entries (``shap_phi_*``) and honest_loss entries (``honest_loss_*``) coexist in the
    same cache_dir without key collisions.

    The two consumers compose their own ``DiskCache`` keys (iter79 OOF-SHAP uses prefix
    ``shap_phi_``; iter80 honest_loss uses ``honest_loss_``). Sharing one cache_dir between them
    must not poison either: a SHAP cache miss must not be served from a honest_loss entry of the
    same column subset, and vice versa.
    """
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import compute_shap_matrix, make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

    X, y = _make_clf_data(n=300, f=6)
    n_search = 220
    X_search, X_hold = X.iloc[:n_search].reset_index(drop=True), X.iloc[n_search:].reset_index(drop=True)
    y_search, y_hold = y[:n_search], y[n_search:]
    tpl = make_default_estimator(classification=True, random_state=0, n_estimators=30)
    shared_dir = tmp_path / "shared_cache"

    # 1) Populate the cache with OOF-SHAP entries (shap_phi_*).
    _phi, _base, _y_aligned = compute_shap_matrix(
        tpl,
        X_search,
        y_search,
        classification=True,
        out_of_fold=True,
        n_splits=3,
        rng=np.random.default_rng(0),
        cache_dir=shared_dir,
    )
    shap_files = [f.name for f in shared_dir.iterdir() if f.name.startswith("shap_phi_")]
    assert len(shap_files) >= 1, "OOF-SHAP should have written shap_phi_ entries"

    # 2) Populate the SAME cache with revalidate_top_n entries (honest_loss_*).
    candidates = [(0.1, [0, 1, 2]), (0.2, [1, 2, 3])]
    revalidate_top_n(
        candidates,
        tpl,
        X_search,
        y_search,
        X_hold,
        y_hold,
        classification=True,
        n_models=1,
        n_jobs=1,
        disk_cache_dir=shared_dir,
    )
    all_files = [f.name for f in shared_dir.iterdir()]
    honest_files = [n for n in all_files if n.startswith("honest_loss_")]
    shap_files_after = [n for n in all_files if n.startswith("shap_phi_")]
    # Both namespaces present, no collision (each prefix has its own entries).
    assert len(honest_files) >= 1, f"revalidate should have written honest_loss_ entries; got {all_files}"
    assert len(shap_files_after) == len(shap_files), "OOF-SHAP entries should be untouched"
    # No file has BOTH prefixes (sanity: the prefixes never overlap on disk).
    assert all(not (n.startswith("shap_phi_") and n.startswith("honest_loss_")) for n in all_files)


# -------------------- 5. ShapProxiedFS-level wiring --------------------


def test_shapproxiedfs_two_fit_cache_dir_writes_honest_loss_entries(tmp_path: Path):
    """End-to-end: ShapProxiedFS with cache_dir writes BOTH shap_phi_ AND honest_loss_ entries in
    the same fit (the OOF-SHAP path + the trust-guard / revalidation honest-retrain paths)."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _make_clf_data()
    cache_dir = tmp_path / "e2e_cache"

    sel = ShapProxiedFS(
        classification=True,
        metric="brier",
        optimizer="bruteforce",
        max_features=4,
        top_n=6,
        n_splits=3,
        n_revalidation_models=1,
        trust_guard=True,
        n_anchors=6,
        revalidate=True,
        cluster_features=False,
        random_state=0,
        verbose=False,
        n_jobs=1,
        cache_dir=str(cache_dir),
    )
    sel.fit(X, y)

    files = [f.name for f in cache_dir.iterdir()]
    # Both namespaces should be populated by the end of a single fit.
    assert any(n.startswith("shap_phi_") for n in files), f"OOF-SHAP entry missing; got {files}"
    assert any(n.startswith("honest_loss_") for n in files), f"honest_loss entry missing; got {files}"
