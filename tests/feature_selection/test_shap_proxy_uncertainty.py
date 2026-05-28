"""Unit + biz_val for config-jitter SHAP (#8) and uncertainty-aware ranking (#7).

#8: averaging SHAP over depth-jittered models is a Monte-Carlo over the path-order arbitrariness that
splits credit between correlated features -> more STABLE attributions than a single model.
#7: the per-model attribution variance lets the optimiser penalise unstable subsets.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection._shap_proxy_objective import subset_uncertainty

pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _data(seed=0, n=700):
    rng = np.random.default_rng(seed)
    inf = rng.normal(size=(n, 4))
    noise = rng.normal(size=(n, 4))
    X = np.column_stack([inf, noise])
    import pandas as pd

    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    y = (0.9 * inf[:, 0] + 0.8 * inf[:, 1] - 0.7 * inf[:, 2] + 0.4 * inf[:, 3] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


def test_subset_uncertainty_ranks_stable_below_unstable():
    n, f = 100, 5
    phi_var = np.zeros((n, f))
    phi_var[:, 2] = 1.0  # feature 2 is unstable
    assert subset_uncertainty(phi_var, [0, 1]) == 0.0          # stable subset
    assert subset_uncertainty(phi_var, [0, 2]) > 0.0           # includes unstable feature
    assert subset_uncertainty(phi_var, [2]) > subset_uncertainty(phi_var, [0])
    assert subset_uncertainty(None, [0, 1]) == 0.0             # no variance computed


def test_config_jitter_returns_nonneg_variance():
    from mlframe.feature_selection._shap_proxy_explain import compute_shap_matrix, make_default_estimator

    X, y = _data(0)
    model = make_default_estimator(classification=True)
    phi, base, y_phi, phi_var = compute_shap_matrix(
        model, X, y, classification=True, out_of_fold=False, n_models=4, config_jitter=True,
        return_variance=True, rng=np.random.default_rng(0))
    assert phi.shape == (len(X), 8)
    assert phi_var.shape == (len(X), 8)
    assert np.all(phi_var >= 0.0)
    assert phi_var.sum() > 0.0  # depth-jittered models genuinely disagree -> non-zero variance


def test_oof_shap_parallel_folds_match_serial():
    """Parallelising the out-of-fold model fits must be byte-identical to the serial path: seeds are
    pre-drawn in fold order, so each fold sees the SAME seed regardless of which thread finishes first.
    A correctness guard for the wide-data OOF-SHAP speedup."""
    import pandas as pd

    from mlframe.feature_selection._shap_proxy_explain import compute_shap_matrix, make_default_estimator

    rng = np.random.default_rng(0)
    n, f = 1500, 80  # >= treeshap numba crossover so the parallelised fold path is exercised
    Xnp = rng.normal(size=(n, f))
    X = pd.DataFrame(Xnp, columns=[f"f{i}" for i in range(f)])
    y = (1.0 * Xnp[:, 0] + 0.8 * Xnp[:, 1] - 0.6 * Xnp[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(int)
    model = make_default_estimator(classification=True, random_state=0)

    serial = compute_shap_matrix(model, X, y, classification=True, out_of_fold=True, n_splits=4,
                                 rng=np.random.default_rng(0), n_jobs=1)
    parallel = compute_shap_matrix(model, X, y, classification=True, out_of_fold=True, n_splits=4,
                                   rng=np.random.default_rng(0), n_jobs=-1)
    phi_s, base_s, _ = serial
    phi_p, base_p, _ = parallel
    assert np.array_equal(phi_s, phi_p), f"phi differs by up to {np.abs(phi_s - phi_p).max()}"
    assert np.array_equal(base_s, base_p)


def test_compute_shap_matrix_n_estimators_cap_is_clamp():
    """Iter19 unit: ``n_estimators_cap`` clamps the per-fold booster's tree count via
    ``min(template_n_estimators, cap)``. Behaviour-test by asserting the fitted boosters carry the
    expected tree count and the produced phi has the right shape -- no implementation-string
    inspection."""
    import pandas as pd
    from unittest.mock import patch

    from mlframe.feature_selection import _shap_proxy_explain
    from mlframe.feature_selection._shap_proxy_explain import compute_shap_matrix, make_default_estimator

    rng = np.random.default_rng(0)
    n, f = 800, 24  # narrow path: stays on the shap-lib backend so the test runs fast
    Xnp = rng.normal(size=(n, f))
    X = pd.DataFrame(Xnp, columns=[f"f{i}" for i in range(f)])
    y = (1.0 * Xnp[:, 0] + 0.8 * Xnp[:, 1] - 0.6 * Xnp[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(int)
    template = make_default_estimator(classification=True, random_state=0, n_estimators=300)

    captured: list[int] = []
    real_fit_one = _shap_proxy_explain._fit_one

    def _spy_fit_one(model_template, X, y, classification, seed, jitter_depth=None,
                     inner_n_jobs=None, n_estimators_cap=None):
        est = real_fit_one(model_template, X, y, classification, seed, jitter_depth=jitter_depth,
                           inner_n_jobs=inner_n_jobs, n_estimators_cap=n_estimators_cap)
        captured.append(int(est.get_params()["n_estimators"]))
        return est

    # cap=100 with template n_estimators=300 -> every fitted booster carries 100 trees.
    with patch.object(_shap_proxy_explain, "_fit_one", _spy_fit_one):
        phi, base, _ = compute_shap_matrix(template, X, y, classification=True, out_of_fold=True,
                                            n_splits=3, rng=np.random.default_rng(0), n_jobs=1,
                                            n_estimators_cap=100)
    assert phi.shape == (n, f)
    assert len(captured) == 3
    assert all(c == 100 for c in captured), f"cap=100 did not clamp; got {captured}"

    # cap=500 with template n_estimators=300 -> clamp via min(): template wins (300), cap can't INCREASE.
    captured.clear()
    with patch.object(_shap_proxy_explain, "_fit_one", _spy_fit_one):
        compute_shap_matrix(template, X, y, classification=True, out_of_fold=True, n_splits=2,
                            rng=np.random.default_rng(0), n_jobs=1, n_estimators_cap=500)
    assert all(c == 300 for c in captured), f"cap=500 should leave 300-tree template untouched; got {captured}"

    # cap=None -> legacy uncapped path (the booster carries the template's n_estimators).
    captured.clear()
    with patch.object(_shap_proxy_explain, "_fit_one", _spy_fit_one):
        compute_shap_matrix(template, X, y, classification=True, out_of_fold=True, n_splits=2,
                            rng=np.random.default_rng(0), n_jobs=1, n_estimators_cap=None)
    assert all(c == 300 for c in captured), f"cap=None should leave 300-tree template untouched; got {captured}"


def test_facade_oof_shap_n_estimators_default_is_100():
    """Iter19 unit: facade default ``oof_shap_n_estimators=100`` so a fresh selector ships the
    cap-the-OOF-ranker behaviour without per-caller opt-in. Regression guard against silent revert."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS()
    assert sel.oof_shap_n_estimators == 100


def test_facade_oof_shap_n_estimators_passes_to_compute_shap_matrix():
    """Iter19 unit: the facade plumbs its ``oof_shap_n_estimators`` into ``compute_shap_matrix``
    via the ``n_estimators_cap`` kwarg (and uncapped passes ``None`` when the caller opts out)."""
    import pandas as pd
    from unittest.mock import patch

    # The facade lazy-imports ``compute_shap_matrix`` from ``_shap_proxy_explain`` inside ``fit``;
    # patch it on the SOURCE module so the lazy import picks up the spy.
    from mlframe.feature_selection import _shap_proxy_explain as explain_module
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    n, f = 600, 12
    Xnp = rng.normal(size=(n, f))
    X = pd.DataFrame(Xnp, columns=[f"f{i}" for i in range(f)])
    y = (1.0 * Xnp[:, 0] + 0.8 * Xnp[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(int)

    real_compute = explain_module.compute_shap_matrix
    captured: list[object] = []

    def _spy(*args, **kwargs):
        captured.append(kwargs.get("n_estimators_cap"))
        return real_compute(*args, **kwargs)

    with patch.object(explain_module, "compute_shap_matrix", _spy):
        ShapProxiedFS(classification=True, metric="brier", optimizer="bruteforce", max_features=4,
                      top_n=5, n_splits=3, trust_guard=False, random_state=0, verbose=False,
                      n_jobs=1).fit(X, y)
    assert captured == [100], f"facade default should pass cap=100 to compute_shap_matrix; got {captured}"

    captured.clear()
    with patch.object(explain_module, "compute_shap_matrix", _spy):
        ShapProxiedFS(classification=True, metric="brier", optimizer="bruteforce", max_features=4,
                      top_n=5, n_splits=3, trust_guard=False, oof_shap_n_estimators=None,
                      random_state=0, verbose=False, n_jobs=1).fit(X, y)
    assert captured == [None], f"oof_shap_n_estimators=None must pass through; got {captured}"


@pytest.mark.slow
def test_biz_val_config_jitter_stabilizes_importance_ranking():
    """Averaging over depth-jittered models should make the SHAP-importance ranking at least as stable
    across resamples as a single model (denoising the path-order arbitrariness)."""
    from scipy.stats import spearmanr

    from mlframe.feature_selection._shap_proxy_explain import compute_shap_matrix, make_default_estimator

    def importance(seed, n_models, jitter):
        X, y = _data(seed)
        model = make_default_estimator(classification=True, random_state=seed)
        phi, *_ = compute_shap_matrix(model, X, y, classification=True, out_of_fold=False,
                                      n_models=n_models, config_jitter=jitter, rng=np.random.default_rng(seed))
        return np.abs(phi).mean(axis=0)

    seeds = [0, 1, 2]
    single = [importance(s, 1, False) for s in seeds]
    jittered = [importance(s, 4, True) for s in seeds]

    def mean_pairwise_stability(rankings):
        cs = []
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                cs.append(spearmanr(rankings[i], rankings[j]).statistic)
        return float(np.mean(cs))

    s_single = mean_pairwise_stability(single)
    s_jit = mean_pairwise_stability(jittered)
    # Jitter-averaging must not be less stable than a single model (usually strictly more stable).
    assert s_jit >= s_single - 0.02, f"jitter stability {s_jit:.3f} < single {s_single:.3f}"


@pytest.mark.slow
def test_facade_uncertainty_penalty_runs_and_reports():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _data(0, n=1200)
    sel = ShapProxiedFS(classification=True, metric="brier", optimizer="bruteforce", max_features=5,
                        top_n=10, n_splits=3, n_models=2, config_jitter=True, uncertainty_penalty=0.5,
                        n_revalidation_models=1, trust_guard=False, random_state=0, verbose=False)
    sel.fit(X, y)
    assert sel.shap_proxy_report_.get("uncertainty", {}).get("applied") is True
    assert len(sel.selected_features_) >= 1
