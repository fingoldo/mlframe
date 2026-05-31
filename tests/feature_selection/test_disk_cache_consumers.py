"""Consumer-side tests for the iter79 disk cache wiring.

Two consumers share the same cache infrastructure (``mlframe.utils.disk_cache``):

* ``compute_shap_matrix`` (ShapProxiedFS) caches the OOF-SHAP (phi, base, ...) tuple.
* ``per_feature_edges`` (MRMR) caches per-column adaptive bin edges.

Both wire ``cache_dir=None`` as the default -> zero behaviour change. With a dir set,
a second call on the same inputs must return BYTE-IDENTICAL results AND be measurably
cheaper. These tests use small data so wall-clock measurements stay quick on CI; the
speedup multiplier is loose (the focus is correctness + non-zero hit-rate, not perf
calibration -- that belongs to the bench script).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# -------------------- ShapProxiedFS / compute_shap_matrix --------------------


def _make_clf_data(n=200, f=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float64)
    # Linear signal on first 3 columns + noise.
    logits = X[:, 0] * 1.5 - X[:, 1] * 0.7 + X[:, 2] * 0.5
    y = (logits + rng.standard_normal(n) * 0.3 > 0).astype(np.int64)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(f)])
    return Xdf, y


def test_compute_shap_matrix_default_no_cache_behaviour_unchanged():
    """``cache_dir=None`` -> identical phi/base vs the legacy call path."""
    from mlframe.feature_selection._shap_proxy_explain import compute_shap_matrix, make_default_estimator

    X, y = _make_clf_data()
    tpl = make_default_estimator(classification=True, random_state=0)

    rng_a = np.random.default_rng(7)
    phi_a, base_a, y_a = compute_shap_matrix(
        tpl, X, y, classification=True, out_of_fold=True, n_splits=3, rng=rng_a,
    )

    rng_b = np.random.default_rng(7)
    phi_b, base_b, y_b = compute_shap_matrix(
        tpl, X, y, classification=True, out_of_fold=True, n_splits=3, rng=rng_b,
        cache_dir=None,
    )

    np.testing.assert_array_equal(phi_a, phi_b)
    np.testing.assert_array_equal(base_a, base_b)
    np.testing.assert_array_equal(y_a, y_b)


def test_compute_shap_matrix_cache_hit_returns_identical(tmp_path: Path):
    """Cache hit returns BYTE-IDENTICAL phi/base/y AND populates the cache directory.

    Perf-side speedup is measured by the C3 bench (``_benchmarks/bench_iter79_disk_cache.py``)
    where the per-fold xgboost cost dominates and the hit/miss delta is unambiguous; on small
    in-test data (n<1000, f<32) the per-fold fit is already ~150ms and competes with the
    cache I/O overhead, producing noisy multipliers unsuited to a CI gate.
    """
    from mlframe.feature_selection._shap_proxy_explain import compute_shap_matrix, make_default_estimator

    X, y = _make_clf_data(n=400, f=16)
    tpl = make_default_estimator(classification=True, random_state=0)
    cache_dir = tmp_path / "shap_cache"

    rng_a = np.random.default_rng(7)
    phi_a, base_a, y_a = compute_shap_matrix(
        tpl, X, y, classification=True, out_of_fold=True, n_splits=3, rng=rng_a,
        cache_dir=cache_dir,
    )
    # Cache directory MUST be populated after the miss path.
    files = list(cache_dir.iterdir())
    assert len(files) >= 1, "cache directory should hold a shap_phi_ entry after the miss"
    assert any(f.name.startswith("shap_phi_") for f in files)

    rng_b = np.random.default_rng(7)
    phi_b, base_b, y_b = compute_shap_matrix(
        tpl, X, y, classification=True, out_of_fold=True, n_splits=3, rng=rng_b,
        cache_dir=cache_dir,
    )
    np.testing.assert_array_equal(phi_a, phi_b)
    np.testing.assert_array_equal(base_a, base_b)
    np.testing.assert_array_equal(y_a, y_b)


def test_compute_shap_matrix_rng_state_matches_after_hit(tmp_path: Path):
    """Cache hit must advance ``rng`` by the SAME number of draws the miss path did,
    so downstream stages that share the rng see identical bits regardless of hit/miss."""
    from mlframe.feature_selection._shap_proxy_explain import compute_shap_matrix, make_default_estimator

    X, y = _make_clf_data()
    tpl = make_default_estimator(classification=True, random_state=0)
    cache_dir = tmp_path / "shap_cache"

    # Miss: capture rng-state after the call.
    rng_a = np.random.default_rng(7)
    compute_shap_matrix(
        tpl, X, y, classification=True, out_of_fold=True, n_splits=3, rng=rng_a,
        cache_dir=cache_dir,
    )
    post_state_a = rng_a.integers(0, 2**31 - 1, size=4).tolist()

    # Hit: same starting rng, must produce same post-state.
    rng_b = np.random.default_rng(7)
    compute_shap_matrix(
        tpl, X, y, classification=True, out_of_fold=True, n_splits=3, rng=rng_b,
        cache_dir=cache_dir,
    )
    post_state_b = rng_b.integers(0, 2**31 - 1, size=4).tolist()

    assert post_state_a == post_state_b


def test_shapproxiedfs_accepts_cache_dir_kwarg(tmp_path: Path):
    """ShapProxiedFS plumbs ``cache_dir`` through to compute_shap_matrix without crashing."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    selector = ShapProxiedFS(
        classification=True,
        n_splits=3,
        n_models=1,
        prefilter_top=None,
        revalidate=False,
        trust_guard=False,
        cluster_features=False,
        random_state=0,
        verbose=False,
        cache_dir=str(tmp_path / "shap_cache"),
    )
    X, y = _make_clf_data(n=120, f=8)
    selector.fit(X, y)
    assert selector.cache_dir is not None
    # Sklearn ``get_params()`` must round-trip the new kwarg (clone-safety).
    assert "cache_dir" in selector.get_params()


# -------------------- per_feature_edges (MRMR bins) --------------------


def test_per_feature_edges_default_no_cache_behaviour_unchanged():
    """``cache_dir=None`` -> identical edge arrays vs the legacy call."""
    from mlframe.feature_selection.filters._adaptive_nbins import per_feature_edges

    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 6)).astype(np.float64)
    e_a = per_feature_edges(X, method="freedman_diaconis")
    e_b = per_feature_edges(X, method="freedman_diaconis", cache_dir=None)
    assert len(e_a) == len(e_b)
    for a, b in zip(e_a, e_b):
        np.testing.assert_array_equal(a, b)


def test_per_feature_edges_cache_hit_identical(tmp_path: Path):
    """Second call on the same X+method MUST return identical edges AND populate the cache."""
    from mlframe.feature_selection.filters._adaptive_nbins import per_feature_edges

    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 8)).astype(np.float64)
    cache_dir = tmp_path / "nbin_cache"

    e_a = per_feature_edges(X, method="freedman_diaconis", cache_dir=cache_dir)
    # After the first call the cache directory has one file per column.
    files = list(cache_dir.iterdir())
    assert len(files) >= 1, "cache directory should be populated after the miss-path call"

    e_b = per_feature_edges(X, method="freedman_diaconis", cache_dir=cache_dir)
    assert len(e_a) == len(e_b)
    for a, b in zip(e_a, e_b):
        np.testing.assert_array_equal(a, b)


def test_per_feature_edges_supervised_y_keyed(tmp_path: Path):
    """A supervised method's edges must change when ``y`` changes (cache key must include y)."""
    from mlframe.feature_selection.filters._adaptive_nbins import per_feature_edges

    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 4)).astype(np.float64)
    # Two distinct y vectors; cache keys must NOT collide.
    y1 = (X[:, 0] > 0).astype(np.int64)
    y2 = (X[:, 1] > 0).astype(np.int64)
    cache_dir = tmp_path / "mdlp_cache"

    e1 = per_feature_edges(X, y=y1, method="mdlp", cache_dir=cache_dir)
    e2 = per_feature_edges(X, y=y2, method="mdlp", cache_dir=cache_dir)
    # At least one column should differ -- if the cache served e1 for the e2 call we'd see
    # full equality, which would mean the y-summary was missing from the key.
    any_diff = False
    for a, b in zip(e1, e2):
        if a.shape != b.shape or not np.array_equal(a, b):
            any_diff = True
            break
    assert any_diff, "supervised method edges must depend on y (cache must be y-keyed)"


def test_mrmr_accepts_cache_dir_kwarg(tmp_path: Path):
    """MRMR plumbs ``cache_dir`` through to categorize_dataset without crashing."""
    pytest.importorskip("polars")
    from mlframe.feature_selection.filters.mrmr import MRMR

    mr = MRMR(cache_dir=str(tmp_path / "mrmr_cache"))
    assert mr.cache_dir is not None
    # Sklearn ``get_params()`` must round-trip the new kwarg (clone-safety).
    assert "cache_dir" in mr.get_params()
