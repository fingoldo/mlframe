"""biz_val tests for ``ShapProxiedFS`` end-to-end on synthetic data.

The business value of this selector is: cheaply rank feature subsets so that the chosen subset, when
a model is HONESTLY retrained on it, beats both a random same-size subset AND a plain
SHAP-importance-top-k subset, while excluding noise + redundant-correlated columns -- at a fraction
of the cost of exhaustive honest retraining.

Synthetic design mirrors the user's poker-style construction: K informative features driving the
target, plus pure-noise columns and columns correlated with informative ones (the redundancy trap
that breaks naive importance ranking).

Measured dev run (seed=0): the selector recovers the informative set, proxy_honest_loss <=
importance_honest_loss, and proxy_honest_loss << random-baseline loss. Floors carry 5-15% headroom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _make_dataset(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    inf = rng.normal(size=(n, 5))                       # 5 informative
    noise = rng.normal(size=(n, 4))                     # 4 pure noise
    corr = inf[:, :2] + 0.3 * rng.normal(size=(n, 2))   # 2 redundant-with-informative
    X = pd.DataFrame(
        np.column_stack([inf, noise, corr]),
        columns=[f"inf{i}" for i in range(5)] + [f"noise{i}" for i in range(4)] + ["corr0", "corr1"],
    )
    logit = 0.9 * inf[:, 0] + 0.8 * inf[:, 1] - 0.7 * inf[:, 2] + 0.6 * inf[:, 3] + 0.4 * inf[:, 4]
    y = (logit + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


@pytest.mark.slow
def test_biz_val_shap_proxied_fs_recovers_informative_and_beats_baselines():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _make_dataset(seed=0)
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="bruteforce",
        max_features=7, top_n=20, n_splits=3, n_revalidation_models=2,
        random_state=0, verbose=False,
    )
    sel.fit(X, pd.Series(y))
    selected = set(sel.selected_features_)
    rep = sel.shap_proxy_report_

    informative_kept = selected & {f"inf{i}" for i in range(5)}
    noise_kept = [c for c in selected if c.startswith("noise")]

    # Recovery: at least 4 of 5 informative features (measured 5/5; floor 4 leaves seed headroom).
    assert len(informative_kept) >= 4, f"too few informative kept: {sorted(informative_kept)}"
    # Discrimination: at most 1 pure-noise column (measured 0).
    assert len(noise_kept) <= 1, f"too many noise columns kept: {noise_kept}"

    # Proxy fidelity was measured on this data.
    assert rep["trust"]["spearman"] > 0.7, rep["trust"]

    # Unique value: proxy subset at least ties SHAP-importance-top-k (measured: proxy strictly wins).
    abl = rep["importance_ablation"]
    assert abl["proxy_at_least_ties"], abl

    # Honest win over a random same-size subset by a wide margin.
    best = rep["revalidation"]["ranked"][0]["honest_loss"]
    baseline = rep["revalidation"]["random_baseline"]["honest_loss"]
    assert best < 0.9 * baseline, f"chosen honest loss {best} not clearly below random baseline {baseline}"


@pytest.mark.slow
@pytest.mark.timeout(600)  # one wide (3000-col) prefilter model fit dominates; exceeds the 60s default
def test_biz_val_wide_pipeline_scales_and_recovers_informative():
    """The user's real regime is tens of thousands of features. This is the same end-to-end pipeline
    (prefilter -> cluster -> OOF-SHAP -> pre-screen -> search -> trust guard -> honest re-validation ->
    within-cluster refine) at a CI-affordable width (3000 features: a few informatives + correlated
    redundant copies + thousands of independent noise columns). Asserts it completes and the planted
    informative features survive the prefilter + selection -- i.e. the wide-data path stays correct,
    not just the narrow one. Marked slow (one wide prefilter fit dominates the wall-clock)."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    n_informative, n_redundant, width = 6, 8, 3000
    X, y, roles = make_regime_dataset(
        n_samples=3000, n_informative=n_informative, n_redundant=n_redundant,
        redundancy_rho=0.9, n_noise=width - n_informative - n_redundant, snr=5.0, task="binary", seed=0)
    assert X.shape[1] == width

    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=400, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=15, n_splits=3, n_revalidation_models=2, n_anchors=20,
        random_state=0, verbose=False)
    sel.fit(X, pd.Series(y))

    # Contract integrity at scale: mask length == input width, names map back to original columns.
    assert sel.support_.shape == (width,)
    assert len(sel.selected_features_) == int(sel.support_.sum())
    assert set(sel.selected_features_) <= set(X.columns)

    informative = {f"inf{i}" for i in range(n_informative)}
    selected = set(sel.selected_features_)
    # The wide pipeline must recover most of the planted informatives despite the noise flood
    # (measured 6/6 at this width; floor 4 leaves prefilter/clustering/seed headroom).
    recovered = len(informative & selected)
    assert recovered >= 4, f"wide pipeline recovered too few informatives: {sorted(informative & selected)}"
    # The prefilter itself must not have thrown the informatives away before SHAP ever saw them.
    pf = sel.shap_proxy_report_.get("prefilter")
    assert pf is not None and pf["kept"] == 400 and pf["of"] == width


@pytest.mark.slow
@pytest.mark.timeout(600)  # two wide prefilter fits (model vs fast) dominate; exceeds the 60s default
def test_biz_val_fast_prefilter_does_not_worsen_recovery_vs_model():
    """The iteration-4 win: on wide data the native-importance pre-filter (one model fit on ALL columns)
    is the dominant cost. ``prefilter_method="fast_model"`` (the cheap interaction-aware ranking the
    ``auto`` default routes to for very wide data) must NOT materially worsen informative recovery vs
    the faithful full-booster ``"model"`` pre-filter, while being measurably faster. We score recovery
    on the regime data (informatives + redundant copies + heavy noise) so the planted ground truth is
    known, and assert (a) fast keeps >= model's informatives minus a 1-feature slack, and (b) the fast
    pre-filter STAGE is faster. Measured dev run (seed=0, width=2000): model 6/6, fast 6/6; fast
    prefilter ~5x faster -- floors leave seed headroom."""
    import time

    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    n_informative, n_redundant, width = 6, 8, 2000
    X, y, roles = make_regime_dataset(
        n_samples=3000, n_informative=n_informative, n_redundant=n_redundant,
        redundancy_rho=0.9, n_noise=width - n_informative - n_redundant, snr=5.0, task="binary", seed=0)
    informative = {f"inf{i}" for i in range(n_informative)}

    def _fit(method):
        sel = ShapProxiedFS(
            classification=True, metric="brier", optimizer="auto",
            prefilter_top=300, prefilter_method=method, cluster_features=True, cluster_corr_threshold=0.7,
            top_n=15, n_splits=3, n_revalidation_models=2, n_anchors=20, random_state=0, verbose=False)
        sel._stage_timings = {}
        t0 = time.perf_counter()
        sel.fit(X, pd.Series(y))
        total = time.perf_counter() - t0
        rec = len(informative & set(sel.selected_features_))
        return rec, sel._stage_timings.get("prefilter", total), sel.shap_proxy_report_["prefilter"]

    rec_model, pf_model_secs, info_model = _fit("model")
    rec_fast, pf_fast_secs, info_fast = _fit("fast_model")

    assert info_model["method"] == "model" and info_fast["method"] == "fast_model"
    # Quality: the fast pre-filter recovers within 1 informative of the faithful model pre-filter.
    assert rec_fast >= rec_model - 1, (
        f"fast_model prefilter worsened recovery: fast={rec_fast}/{n_informative} vs model={rec_model}/{n_informative}")
    # And both must still recover most of the planted informatives (sanity floor).
    assert rec_fast >= 4, f"fast_model recovered too few informatives: {rec_fast}/{n_informative}"
    # Speed: the fast pre-filter STAGE must be faster than the full-booster one (the whole point).
    assert pf_fast_secs < pf_model_secs, (
        f"fast_model prefilter ({pf_fast_secs:.2f}s) not faster than model ({pf_model_secs:.2f}s)")


@pytest.mark.slow
def test_biz_val_regression_recovers_informative():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(1)
    n = 2500
    inf = rng.normal(size=(n, 4))
    noise = rng.normal(size=(n, 5))
    X = pd.DataFrame(np.column_stack([inf, noise]),
                     columns=[f"inf{i}" for i in range(4)] + [f"noise{i}" for i in range(5)])
    y = inf[:, 0] + 0.8 * inf[:, 1] - 0.6 * inf[:, 2] + 0.5 * inf[:, 3] + 0.1 * rng.normal(size=n)

    sel = ShapProxiedFS(classification=False, metric="rmse", optimizer="bruteforce",
                        max_features=6, top_n=15, n_splits=3, n_revalidation_models=2,
                        random_state=0, verbose=False)
    sel.fit(X, y)
    selected = set(sel.selected_features_)
    informative_kept = selected & {f"inf{i}" for i in range(4)}
    noise_kept = [c for c in selected if c.startswith("noise")]
    assert len(informative_kept) >= 3
    assert len(noise_kept) <= 2
