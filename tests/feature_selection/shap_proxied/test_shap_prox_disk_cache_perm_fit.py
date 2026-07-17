"""iter82 cache wiring tests for ``_permutation_importance_ranking`` (stage 2a of refine).

iter80/81 cached every honest-loss retrain inside ShapProxiedFS via the disk cache. The single
remaining heavy fit was the booster trained inside ``_permutation_importance_ranking`` -- it does
NOT go through ``_honest_loss`` (it owns its own predict-on-shuffled loop), so iter81's wiring
missed it. iter82 caches the FITTED BOOSTER (Option A) under a key that depends on (X_tr summary,
y_tr summary, sorted cols, template params, n_estimators_cap, template_id) -- everything that
determines the fit. The permutation seed is intentionally NOT in the key (it affects only the
post-fit shuffle predicts).

Coverage:
  1. Default ``disk_cache=None`` -> permutation-importance behaviour bit-identical vs uncached.
  2. Cache hit at the fit step: second call with the same cache_dir returns bit-identical
     (base_loss, importances) and writes a ``perm_imp_fit_*`` entry to disk.
  3. ``within_cluster_refine`` two-fit pattern produces bit-identical refined subset across
     cold/warm and writes BOTH ``honest_loss_*`` (iter81) and ``perm_imp_fit_*`` (iter82) entries
     into the shared cache_dir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")


def _make_clf_data(n=400, f=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float64)
    logits = X[:, 0] * 1.3 - X[:, 1] * 0.8 + X[:, 2] * 0.5
    y = (logits + rng.standard_normal(n) * 0.4 > 0).astype(np.int64)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(f)])
    return Xdf, y


# -------------------- 1. default disk_cache=None conservation --------------------


def test_permutation_importance_default_no_cache_identical():
    """``disk_cache=None`` (default) keeps the legacy single-fit + shuffle-predicts behaviour."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _permutation_importance_ranking

    X, y = _make_clf_data()
    n_search = 280
    X_search = X.iloc[:n_search].reset_index(drop=True)
    X_hold = X.iloc[n_search:].reset_index(drop=True)
    y_search, y_hold = y[:n_search], y[n_search:]
    tpl = make_default_estimator(classification=True, random_state=0, n_estimators=30)
    cols = [0, 1, 2, 3, 4, 5]

    base_a, imps_a = _permutation_importance_ranking(
        tpl,
        X_search,
        y_search,
        X_hold,
        y_hold,
        cols,
        True,
        "brier",
        n_estimators_cap=20,
        seed=0,
    )
    base_b, imps_b = _permutation_importance_ranking(
        tpl,
        X_search,
        y_search,
        X_hold,
        y_hold,
        cols,
        True,
        "brier",
        n_estimators_cap=20,
        seed=0,
    )
    assert base_a == base_b
    np.testing.assert_array_equal(imps_a, imps_b)


# -------------------- 2. cache hit at fit -> bit-identical output, populates dir --------------------


def test_permutation_importance_cache_hit_identical_and_writes_entry(tmp_path: Path):
    """A second ranking call with a shared ``disk_cache`` reuses the fitted booster pickle and
    produces bit-identical (base_loss, importances). The first call writes a ``perm_imp_fit_*``
    entry into the cache_dir; the second call hits it (its ``misses`` counter does not advance)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _permutation_importance_ranking
    from mlframe.utils.disk_cache import DiskCache

    X, y = _make_clf_data()
    n_search = 280
    X_search = X.iloc[:n_search].reset_index(drop=True)
    X_hold = X.iloc[n_search:].reset_index(drop=True)
    y_search, y_hold = y[:n_search], y[n_search:]
    tpl = make_default_estimator(classification=True, random_state=0, n_estimators=30)
    cols = [0, 1, 2, 3, 4, 5]
    cache_dir = tmp_path / "perm_fit_cache"
    dc = DiskCache(cache_dir)

    base_cold, imps_cold = _permutation_importance_ranking(
        tpl,
        X_search,
        y_search,
        X_hold,
        y_hold,
        cols,
        True,
        "brier",
        n_estimators_cap=20,
        seed=0,
        disk_cache=dc,
        template_id="tpl_v1",
    )
    files_after_cold = [f.name for f in cache_dir.iterdir()]
    assert any(n.startswith("perm_imp_fit_") for n in files_after_cold), f"cold call should have written a perm_imp_fit_ entry; got {files_after_cold}"
    misses_after_cold = dc.misses
    hits_after_cold = dc.hits

    base_warm, imps_warm = _permutation_importance_ranking(
        tpl,
        X_search,
        y_search,
        X_hold,
        y_hold,
        cols,
        True,
        "brier",
        n_estimators_cap=20,
        seed=0,
        disk_cache=dc,
        template_id="tpl_v1",
    )
    # Bit-identical: same booster pickle reloaded, same shuffle seed, same X_ev rows.
    assert base_warm == base_cold
    np.testing.assert_array_equal(imps_warm, imps_cold)
    # The warm call must have registered a hit (and no new misses for the fit step).
    assert dc.hits > hits_after_cold, "warm call should have a cache hit on the fitted booster"
    assert dc.misses == misses_after_cold, f"warm call should not miss the fit cache; misses cold={misses_after_cold} warm={dc.misses}"


# -------------------- 3. within_cluster_refine two-fit -> bit-identical subset + both prefixes --------------------


def test_within_cluster_refine_two_fit_cache_writes_perm_fit_and_honest_loss(tmp_path: Path):
    """End-to-end within_cluster_refine two-fit pattern: the cache_dir accumulates BOTH the iter81
    ``honest_loss_*`` entries (per-trial losses) AND the iter82 ``perm_imp_fit_*`` entries (stage-2a
    fitted booster). The warm second call returns a bit-identical refined subset."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import within_cluster_refine

    X, y = _make_clf_data()
    n_search = 280
    X_search = X.iloc[:n_search].reset_index(drop=True)
    X_hold = X.iloc[n_search:].reset_index(drop=True)
    y_search, y_hold = y[:n_search], y[n_search:]
    tpl = make_default_estimator(classification=True, random_state=0, n_estimators=30)
    member_cols = [0, 1, 2, 3, 4, 5]
    member_groups = [[0, 1, 2], [3, 4, 5]]
    cache_dir = tmp_path / "refine_perm_cache"

    refined_cold = within_cluster_refine(
        member_cols,
        tpl,
        X_search,
        y_search,
        X_hold,
        y_hold,
        classification=True,
        n_jobs=1,
        member_groups=member_groups,
        refine_n_estimators=20,
        disk_cache_dir=cache_dir,
    )
    files_cold = [f.name for f in Path(cache_dir).iterdir()]
    assert any(n.startswith("honest_loss_") for n in files_cold), f"refine should write honest_loss_ entries (iter81); got {files_cold}"
    assert any(n.startswith("perm_imp_fit_") for n in files_cold), f"refine stage-2a should write perm_imp_fit_ entries (iter82); got {files_cold}"

    refined_warm = within_cluster_refine(
        member_cols,
        tpl,
        X_search,
        y_search,
        X_hold,
        y_hold,
        classification=True,
        n_jobs=1,
        member_groups=member_groups,
        refine_n_estimators=20,
        disk_cache_dir=cache_dir,
    )
    assert tuple(refined_cold) == tuple(refined_warm), (
        f"warm refine subset must match cold (cache served same booster + same losses); cold={refined_cold} warm={refined_warm}"
    )
