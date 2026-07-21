"""Wide-data scalability / guard-engagement tests for the feature-selection paths (p >= 1000).

These are LIVENESS + GUARD-ENGAGEMENT tests, not accuracy tests. The rest of the FS suite caps at p=300, so the
wide-data guards (the corr_clusters memory blocker fix in HybridSelector, RFECV's wide_data_fi_fallback, MRMR's core
filter) had no coverage above p=300. Each test asserts that the path COMPLETES within a wall budget at p in {1000, 2000}
and that the relevant guard actually ENGAGED.

n is kept modest (1500-2500) so each test stays well under the pytest --timeout. Datasets are pure-noise-heavy with a
small informative core: that is exactly the wide-noisy regime the guards are designed for and keeps every fit cheap.
"""

from __future__ import annotations

import os
import time

# Cap BLAS / joblib oversubscription: these wide-p paths run several n_jobs=-1 inner pools (LGBM, RandomForest,
# permutation_importance) which on a shared / contended box oversubscribe physical cores and stall in joblib retrieve.
# A modest cap keeps the liveness budget meetable without changing what the guards do. Set before numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
import pytest


def _make_wide(n: int, p: int, n_informative: int = 8, seed: int = 0):
    """Wide frame: ``n_informative`` columns carry a linear-logit signal, the rest are pure Gaussian noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p).astype(np.float64)
    w = rng.randn(n_informative)
    logit = X[:, :n_informative] @ w
    p1 = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(n) < p1).astype(int)
    cols = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


# Per-test wall budget. Generous (path liveness, not a microbenchmark) but bounded well under the harness --timeout
# (run these with pytest --timeout=600). p=2000 RF/perm fits on a shared/contended box legitimately run minutes.
WALL_BUDGET_S = 500.0


@pytest.mark.parametrize("p", [1000, 2000])
def test_mrmr_core_filter_completes_wide(p):
    """MRMR core filter fit must RETURN (not hang / OOM) on a wide frame and emit a non-empty selection."""
    from mlframe.feature_selection.filters import MRMR

    n = 2000 if p == 1000 else 1500
    X, y = _make_wide(n, p, seed=1)
    # n_jobs=1: avoid loky multiprocessing stalls on a contended box; the core filter path / guard is unaffected.
    m = MRMR(verbose=0, fe_max_steps=0, n_jobs=1, random_seed=0)
    t0 = time.time()
    m.fit(X, y)
    dt = time.time() - t0
    assert dt < WALL_BUDGET_S, f"MRMR p={p} took {dt:.1f}s > {WALL_BUDGET_S}s"
    sel = [c for c in m.get_feature_names_out() if c in X.columns]
    assert len(sel) >= 1, "MRMR returned an empty selection on the wide frame"


@pytest.mark.parametrize("p", [1000, 2000])
def test_rfecv_wide_data_fi_fallback_engages(p):
    """RFECV with the default permutation importance_getter on a wide frame must FIRE the wide_data_fi_fallback
    guard (native importance for the elimination ranking) and complete; verified via the ``_wide_data_fi_applied_``
    attribute carrying the 'fallback_to_native' reason. The guard decision is made on the candidate count alone, so a
    deliberately CHEAP estimator + tight runtime budget exercises the same code path without the multi-iteration RF
    elimination cost (this is a guard-engagement + liveness test, not an accuracy / curve-quality test)."""
    from sklearn.tree import DecisionTreeClassifier
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    n = 1200
    X, y = _make_wide(n, p, seed=2)
    # RUN UNDER A REAL TIME BUDGET (max_runtime_mins), not the bare path. NB max_runtime_mins is checked BETWEEN
    # outer-loop iterations -- it cannot preempt a single in-progress fit -- so at wide p the first full-feature
    # iteration must itself be cheap for the budget to engage. A single DecisionTree (native feature_importances_
    # drives the wide-data fallback ranking) keeps each iteration ~instant, so the 0.5-min budget genuinely bounds
    # the loop. The guard fires on the candidate COUNT (> threshold), independent of estimator cost.
    r = RFECV(
        estimator=DecisionTreeClassifier(max_depth=6, random_state=0),
        importance_getter="permutation",
        wide_data_fi_fallback=True,
        wide_data_fi_threshold=200,
        cv=2,
        max_runtime_mins=0.5,  # real budget; bounds the elimination loop now that each iteration is cheap
    )
    t0 = time.time()
    r.fit(X, y)
    dt = time.time() - t0
    assert dt < WALL_BUDGET_S, f"RFECV p={p} took {dt:.1f}s > {WALL_BUDGET_S}s"
    applied = getattr(r, "_wide_data_fi_applied_", None)
    assert applied is not None, "RFECV wide-data guard did not record _wide_data_fi_applied_"
    assert applied.get("reason") == "fallback_to_native", f"unexpected guard reason: {applied}"
    assert applied.get("n_candidates", 0) > applied.get("threshold", 1e9)


@pytest.mark.parametrize("p", [1000, 2000])
def test_hybrid_selector_completes_and_narrows_wide(p):
    """HybridSelector.fit must complete on a wide frame WITHOUT materializing the full p x p correlation matrix.

    Guard engagement is checked two ways:
      (1) ``_cluster_cols_`` (the relevance-survivor set actually handed to corr_clusters) is far smaller than p -- the
          pure-noise singletons were dropped before clustering, so clustering never saw the full augmented frame.
      (2) corr_clusters itself is monkeypatched to ASSERT it never receives more than ``hybrid_corr_max_features``
          columns (a hard upper bound on the dense matrix it could build).
    """
    import mlframe.feature_selection.hybrid_selector as hs

    n = 1500 if p == 1000 else 1200
    X, y = _make_wide(n, p, seed=3)

    cap = 2000
    seen = {}
    orig = hs.corr_clusters

    def _guarded(Xc, *a, **k):
        """Returns ``orig(Xc, *a, **k)`` (after 2 setup steps)."""
        seen["ncols"] = Xc.shape[1]
        assert Xc.shape[1] <= cap, f"corr_clusters got {Xc.shape[1]} cols > cap {cap} (full p x p risk)"
        return orig(Xc, *a, **k)

    # This test targets the WIDE-DATA clustering guard (corr_clusters narrowing), NOT the member-selector accuracy. The
    # heavy ShapProxiedFS / BorutaShap members (each n_jobs=-1) dominate the wall and oversubscribe cores on a shared
    # box; stub them to no-ops so the test isolates + reliably bounds the path under test (the shared perm-FI prescreen
    # + the guarded corr_clusters). Guard engagement (_cluster_cols_ << p, slab cols <= cap) is what is asserted.
    hs.corr_clusters = _guarded
    sel = hs.HybridSelector(
        use_fe=False,
        use_tree_member=False,
        use_mrmr=False,
        hybrid_corr_max_features=cap,
        random_state=0,
    )
    sel._run_shap = lambda X_, y_, rel, art: list(rel)[:5]
    sel._run_boruta_premerge = lambda X_, y_, rel: list(rel)[:5]

    # Replace the shared permutation-FI (its sklearn permutation_importance(n_jobs=-1) spawns loky workers that stall
    # under a contended box) with a cheap single-fit LGBM gain importance. The corr_clusters narrowing guard under test
    # is agnostic to HOW the FI was produced -- it only uses FI>0 to pick the relevance-survivor set to cluster.
    def _cheap_fi(X_, y_):
        """Cheap fi."""
        import lightgbm as lgb

        mm = lgb.LGBMClassifier(n_estimators=60, num_leaves=31, n_jobs=1, verbose=-1)
        mm.fit(X_, y_)
        imp = mm.feature_importances_.astype(float)
        sel._fi_sum_ = {c: float(v) for c, v in zip(X_.columns, imp)}
        return {c: float(v) for c, v in zip(X_.columns, imp)}

    sel._shared_perm_fi = _cheap_fi
    try:
        t0 = time.time()
        sel.fit(X, y)
        dt = time.time() - t0
    finally:
        hs.corr_clusters = orig

    assert dt < WALL_BUDGET_S, f"HybridSelector p={p} took {dt:.1f}s > {WALL_BUDGET_S}s"
    assert seen.get("ncols", p + 1) <= cap
    # the relevance prescreen should have narrowed clustering well below the full noisy frame
    cluster_cols = getattr(sel, "_cluster_cols_", None)
    assert cluster_cols is not None, "HybridSelector did not record _cluster_cols_"
    assert len(cluster_cols) < p, f"clustering ran on {len(cluster_cols)} cols == full frame p={p} (no narrowing)"
    assert len(sel.raw_selected_) >= 1
