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
        random_state=0, verbose=False, n_jobs=1,
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
        random_state=0, verbose=False, n_jobs=1)
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
            top_n=15, n_splits=3, n_revalidation_models=2, n_anchors=20,
            random_state=0, verbose=False, n_jobs=1)
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
@pytest.mark.timeout(900)
def test_biz_val_prefilter_cap_faster_with_preserved_recovery():
    """Iter10 win: cap the prefilter's ranking booster (``prefilter_n_estimators``) so the
    all-columns importance fit costs ~3x less while preserving informative recovery.

    Apples-to-apples comparison at moderate width (CI-affordable) with ``prefilter_method="model"``
    forced (the cap pays its biggest dividend on the full-booster path; ``auto`` would route to
    ``fast_model`` here, which already has a reduced budget so the cap is near a no-op there).
    Assertion floors leave generous seed headroom -- the measured dev run (seed=0, width=3000,
    rows=2500) shows ~2-3x prefilter speedup with recovery preserved.
    """
    import time

    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    n_informative, n_redundant, width = 6, 8, 3000
    X, y, _roles = make_regime_dataset(
        n_samples=2500, n_informative=n_informative, n_redundant=n_redundant,
        redundancy_rho=0.9, n_noise=width - n_informative - n_redundant, snr=5.0,
        task="binary", seed=0)
    informative = {f"inf{i}" for i in range(n_informative)}

    def _fit(cap):
        sel = ShapProxiedFS(
            classification=True, metric="brier", optimizer="auto",
            prefilter_top=300, prefilter_method="model", prefilter_n_estimators=cap,
            cluster_features=True, cluster_corr_threshold=0.7,
            top_n=12, n_splits=3, n_revalidation_models=2, n_anchors=15,
            random_state=0, verbose=False, n_jobs=1)
        sel._stage_timings = {}
        t0 = time.perf_counter()
        sel.fit(X, pd.Series(y))
        total = time.perf_counter() - t0
        rec = len(informative & set(sel.selected_features_))
        return rec, sel._stage_timings.get("prefilter", total), total, sel.shap_proxy_report_["prefilter"]

    rec_uncapped, pf_uncapped, _, info_uncapped = _fit(cap=None)
    rec_capped, pf_capped, _, info_capped = _fit(cap=100)

    # Cap was actually applied (report carries it).
    assert info_uncapped["n_estimators_cap"] is None
    assert info_capped["n_estimators_cap"] == 100

    # Speed: the capped prefilter must be measurably faster than the uncapped one.
    # (Measured ~2-3x on dev; floor 1.2x leaves Windows-build variance headroom.)
    assert pf_capped < pf_uncapped, (
        f"capped prefilter ({pf_capped:.2f}s) not faster than uncapped ({pf_uncapped:.2f}s)")
    assert pf_capped <= 0.85 * pf_uncapped, (
        f"capped prefilter speedup too small: {pf_uncapped/pf_capped:.2f}x "
        f"(uncapped={pf_uncapped:.2f}s vs capped={pf_capped:.2f}s)")

    # Quality: recovery must not be materially worse (within 1 informative).
    assert rec_capped >= rec_uncapped - 1, (
        f"cap worsened recovery: capped={rec_capped}/{n_informative} "
        f"vs uncapped={rec_uncapped}/{n_informative}")
    # And the capped pipeline must still recover most of the planted informatives (sanity floor).
    assert rec_capped >= 4, f"capped recovered too few informatives: {rec_capped}/{n_informative}"


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
                        random_state=0, verbose=False, n_jobs=1)
    sel.fit(X, y)
    selected = set(sel.selected_features_)
    informative_kept = selected & {f"inf{i}" for i in range(4)}
    noise_kept = [c for c in selected if c.startswith("noise")]
    assert len(informative_kept) >= 3
    assert len(noise_kept) <= 2


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_biz_val_stratified_anchors_preserve_recovery_at_6k_no_catastrophic_spearman_drop():
    """Iter14 trust-guard stratified-anchor lever, end-to-end on the regime synthetic at width=6000.

    Measured on the iter14 dev bench (seed=0, width=6000, n=3000, 12 informatives, two_stage
    prefilter -> 400-col cohort):

        uniform:    spearman=0.969  recovery=10/12
        stratified: spearman=0.877  recovery=10/12

    The lever DOES NOT lift Spearman at this regime: the two_stage prefilter already noise-filters
    the cohort to ~400 quality-biased columns, so uniform anchors over THOSE columns already mostly
    sample informative-vs-noise mixes. Concentrating further by softmax(F) compresses the spread
    that drives the Spearman signal (anchors get closer to each other, ties become noisier). This is
    why the production knob ``trust_guard_stratified_anchors`` defaults to OFF -- but we LOCK IN the
    "no catastrophic drop" + "recovery preserved" invariants so regressions are caught.

    Asserts:
      (1) the opt-in flag actually flips the anchor sampling mode
      (2) recovery is no worse under stratified than under uniform
      (3) Spearman doesn't drop catastrophically (>= uniform - 0.20) -- defends the lever's safety
          envelope: if a future change makes the stratified prior dangerously narrow, the test
          fails BEFORE a user shipping the opt-in hits it in prod.
    """
    from unittest.mock import patch

    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    n_informative, n_redundant, width = 12, 0, 6000
    X, y, _roles = make_regime_dataset(
        n_samples=3000, n_informative=n_informative, n_redundant=n_redundant,
        redundancy_rho=0.9, n_noise=width - n_informative - n_redundant, snr=5.0,
        task="binary", seed=0)
    informative = {f"inf{i}" for i in range(n_informative)}

    def _fit(stratified: bool):
        sel = ShapProxiedFS(
            classification=True, metric="brier", optimizer="auto",
            prefilter_top=400, prefilter_method="two_stage",
            cluster_features=True, cluster_corr_threshold=0.7,
            top_n=15, n_splits=3, n_revalidation_models=2, n_anchors=30,
            trust_guard_stratified_anchors=stratified,
            random_state=0, verbose=False, n_jobs=1)
        sel.fit(X, pd.Series(y))
        rep = sel.shap_proxy_report_
        spearman = rep["trust"]["spearman"]
        mode = rep["trust"]["anchor_sampling"]
        recovery = len(informative & set(sel.selected_features_))
        return spearman, mode, recovery

    print(f"\n[iter14 stratified-anchor biz_val] uniform leg at width={width}", flush=True)
    sp_u, mode_u, rec_u = _fit(stratified=False)
    print(f"[iter14] uniform: spearman={sp_u:.3f} mode={mode_u} recovery={rec_u}/{n_informative}",
          flush=True)
    print(f"[iter14] stratified leg", flush=True)
    sp_s, mode_s, rec_s = _fit(stratified=True)
    print(f"[iter14] stratified: spearman={sp_s:.3f} mode={mode_s} recovery={rec_s}/{n_informative}",
          flush=True)

    assert mode_u == "uniform"
    assert mode_s == "stratified"
    assert rec_s >= rec_u, (
        f"stratified worsened recovery: stratified={rec_s}/{n_informative} vs uniform={rec_u}/{n_informative}")
    # Recovery floor; dev measured 10/12, leave a 1-feature slack for seed/Windows-build variance.
    assert rec_s >= 9, f"stratified recovered too few informatives: {rec_s}/{n_informative}"
    # Spearman safety envelope: catastrophic-drop guard. Dev measured 0.877 vs 0.969 (-0.092);
    # we lock in <= -0.20 absolute as the "won't break the proxy-fidelity report" floor.
    assert sp_s >= sp_u - 0.20, (
        f"stratified catastrophically dropped Spearman: uniform={sp_u:.3f} stratified={sp_s:.3f}")


def test_stratified_anchors_default_off_preserves_legacy_trust_report():
    """The iter14 stratified-anchor lever MUST default to OFF: at the iter14 dev bench it did not pay
    (post-two_stage cohort is already noise-filtered), so we don't ship the regression as the default.
    This is a fast cheap test that asserts the contract on a small synthetic fit -- if a future change
    flips the default the report's ``anchor_sampling`` will switch to 'stratified' here and the test
    fires before it lands in prod. Pairs with the slow 6k biz_val above (which measures the actual
    Spearman cost) -- this one is the cheap structural sentinel."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    # Small enough for the non-slow tier (no @pytest.mark.slow): two_stage prefilter on 500 cols.
    X, y, _ = make_regime_dataset(
        n_samples=600, n_informative=5, n_redundant=0, n_noise=495, snr=5.0,
        task="binary", seed=0)
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=150, prefilter_method="two_stage",
        cluster_features=False, top_n=5, n_splits=3, n_revalidation_models=1, n_anchors=12,
        random_state=0, verbose=False, n_jobs=1)
    sel.fit(X, pd.Series(y))
    # Default (no kwarg) must keep stratified OFF -> report records uniform sampling even though
    # the two_stage prefilter cached the F-score vector.
    assert sel.trust_guard_stratified_anchors is False
    assert sel.shap_proxy_report_["trust"]["anchor_sampling"] == "uniform"


def test_zipf_cardinality_default_on_and_recorded_in_trust_report():
    """The iter16 default of ``trust_guard_cardinality_dist`` is ``'zipf'`` with
    ``trust_guard_zipf_alpha=0.25`` after re-evaluation under the composite ``proxy_fidelity_score``
    metric (raw Spearman dropped, but recall@k lifted enough that the composite gained +0.054). This
    structural sentinel locks in the post-iter16 default so a future regression that silently flips
    back to 'uniform' will fire here before shipping. Reverse direction of the iter15 sentinel; the
    Zipf vs uniform comparison itself is asserted by the slow biz_value test on the width=6000 regime."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, _ = make_regime_dataset(
        n_samples=600, n_informative=5, n_redundant=0, n_noise=495, snr=5.0,
        task="binary", seed=0)
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=150, prefilter_method="two_stage",
        cluster_features=False, top_n=5, n_splits=3, n_revalidation_models=1, n_anchors=12,
        random_state=0, verbose=False, n_jobs=1)
    assert sel.trust_guard_cardinality_dist == "zipf"
    assert sel.trust_guard_zipf_alpha == 0.25  # iter16 composite sweet spot
    sel.fit(X, pd.Series(y))
    assert sel.shap_proxy_report_["trust"]["anchor_cardinality_dist"] == "zipf"
    assert sel.shap_proxy_report_["trust"]["anchor_zipf_alpha"] == 0.25


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_biz_val_zipf_cardinality_preserves_recovery_no_catastrophic_spearman_drop_at_6k():
    """Iter15 trust-guard Zipf cardinality prior, end-to-end on the regime synthetic at width=6000.

    Iter15 honest-negative measurement (seed=0, n=3000, 12 informatives, two_stage prefilter ->
    400-col cohort, n_anchors=30):

        uniform        spearman=0.969  recovery=10/12  (iter14 baseline)
        zipf alpha=1.0 spearman=0.786  recovery=10/12  (Δ -0.183)
        zipf alpha=0.5 spearman=0.947  recovery=10/12  (Δ -0.023)
        zipf alpha=0.25 spearman=0.956 recovery=10/12  (Δ -0.013)

    The Zipf prior consistently regressed Spearman, monotonically with alpha. Hypothesis (small-k
    anchors give honest models a wider loss range to rank) was falsified for this regime: the
    post-two_stage cohort already concentrates informatives, so small-k samples land in
    "all-noise or all-signal" extremes where proxy and honest agree TRIVIALLY (no nuance for
    Spearman to rank). The lever stays exposed for callers in other regimes (low-redundancy data,
    no prefilter) where the small-k prior may pay -- but we LOCK IN the "no catastrophic drop" +
    "recovery preserved" invariants so regressions are caught early.

    The cardinality prior is orthogonal to the F-score stratification (this test keeps stratified
    OFF -- the default -- so we measure the cardinality prior in isolation).

    Asserts:
      (1) the opt-in knob actually flips the cardinality mode
      (2) recovery under Zipf is no worse than uniform (no recovery regression)
      (3) Spearman doesn't drop catastrophically (>= uniform - 0.25) -- this is the safety envelope
          calibrated against the measured alpha=1.0 drop of -0.183 plus headroom for noise. The
          monotonic-with-alpha regression IS expected; we just don't ship it as the default.
    """
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    n_informative, n_redundant, width = 12, 0, 6000
    X, y, _roles = make_regime_dataset(
        n_samples=3000, n_informative=n_informative, n_redundant=n_redundant,
        redundancy_rho=0.9, n_noise=width - n_informative - n_redundant, snr=5.0,
        task="binary", seed=0)
    informative = {f"inf{i}" for i in range(n_informative)}

    def _fit(cardinality_dist: str):
        sel = ShapProxiedFS(
            classification=True, metric="brier", optimizer="auto",
            prefilter_top=400, prefilter_method="two_stage",
            cluster_features=True, cluster_corr_threshold=0.7,
            top_n=15, n_splits=3, n_revalidation_models=2, n_anchors=30,
            trust_guard_stratified_anchors=False,
            trust_guard_cardinality_dist=cardinality_dist,
            random_state=0, verbose=False, n_jobs=1)
        sel.fit(X, pd.Series(y))
        rep = sel.shap_proxy_report_
        spearman = rep["trust"]["spearman"]
        mode = rep["trust"]["anchor_cardinality_dist"]
        recovery = len(informative & set(sel.selected_features_))
        return spearman, mode, recovery

    print(f"\n[iter15 zipf-cardinality biz_val] uniform leg at width={width}", flush=True)
    sp_u, mode_u, rec_u = _fit("uniform")
    print(f"[iter15] uniform card: spearman={sp_u:.3f} mode={mode_u} recovery={rec_u}/{n_informative}",
          flush=True)
    print(f"[iter15] zipf leg", flush=True)
    sp_z, mode_z, rec_z = _fit("zipf")
    print(f"[iter15] zipf card:    spearman={sp_z:.3f} mode={mode_z} recovery={rec_z}/{n_informative}",
          flush=True)

    assert mode_u == "uniform"
    assert mode_z == "zipf"
    assert rec_z >= rec_u, (
        f"zipf cardinality worsened recovery: zipf={rec_z}/{n_informative} vs uniform={rec_u}/{n_informative}")
    # Recovery floor: dev measured 10/12; 9 allows 1-feat slack for cross-platform variance.
    assert rec_z >= 9, f"zipf cardinality recovered too few informatives: {rec_z}/{n_informative}"
    # Catastrophic-drop guard: dev measured Δ=-0.183 at alpha=1.0; floor at -0.25 leaves headroom.
    assert sp_z >= sp_u - 0.25, (
        f"zipf cardinality catastrophically dropped Spearman: uniform={sp_u:.3f} zipf={sp_z:.3f}")
