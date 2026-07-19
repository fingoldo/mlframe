"""biz_val + unit tests for gt_03's ``prescreen_ranking="banzhaf"`` (MSR-Banzhaf prescreen swap).

Wang & Jia (AISTATS 2023) prove Banzhaf is the most noise-robust semivalue -- its ranking should, in
theory, churn less across seeds than mean|phi| under proxy-loss noise. Measured on THIS pipeline the
opposite holds (see ``test_biz_val_banzhaf_ranking_seed_stability_low_snr``'s docstring for the two
discriminating experiments that ruled out MSR sampling noise and bed-width as the cause): mean|phi|
is more seed-stable here, because it is read off an already fold-averaged OOF-SHAP phi matrix while
MSR-Banzhaf layers its own fresh per-seed coalition-sampling randomness on top of the same phi. This
is a genuine negative result for the stability claim, not an implementation bug -- the feature ships
opt-in with the default staying "mean_abs_phi" regardless (gt_03 sec 7's own acceptance rule never
required a stability win to ship). These tests pin: (1) the measured (not hypothesized) seed-stability
relationship, (2) no regression on clean high-SNR data, (3) MSR-vs-exact numerical agreement on a
small enumerable game (this DOES hold -- the estimator itself is correct, only the stability-under-
pipeline-composition claim doesn't), plus unit-level guarantees on ``banzhaf_msr`` (dummy feature,
batched-vs-sequential bit identity, clone round-trip, validator rejection).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")

from scipy.stats import spearmanr

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_banzhaf import banzhaf_msr
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import score_margin_auto


def _pairwise_jaccard(sets: list[set]) -> float:
    """Mean pairwise Jaccard similarity across all C(len(sets), 2) unordered pairs of feature sets."""
    pairs = list(combinations(range(len(sets)), 2))
    vals = []
    for i, j in pairs:
        a, b = sets[i], sets[j]
        union = a | b
        vals.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(vals))


def _make_low_snr_fixture(seed=0, n=2000, p=500, n_informative=10, snr=1.5):
    """Low-SNR regression bed (gt_03 plan sec 5 test 1): 10 informative cols buried in noise-dominated proxy loss."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    return make_regime_dataset(n_samples=n, n_informative=n_informative, n_noise=p - n_informative, snr=snr, task="binary", seed=seed)


def _make_high_snr_fixture(seed=0, n=2000, p=500, n_informative=10, snr=25.0):
    """Clean high-SNR bed (gt_03 plan sec 5 test 2): same shape, strong signal -- both rankings should recall perfectly."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    return make_regime_dataset(n_samples=n, n_informative=n_informative, n_noise=p - n_informative, snr=snr, task="binary", seed=seed)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_banzhaf_ranking_seed_stability_low_snr():
    """Measures the seed-to-seed stability (mean pairwise Jaccard of ``selected_features_`` across 4
    seeds) of both prescreen rankings on a low-SNR bed.

    The plan's THEORY (Wang & Jia AISTATS 2023: Banzhaf is the most noise-robust semivalue) predicts
    banzhaf should win here. Measured on this pipeline it does NOT: banzhaf's Jaccard (~0.41-0.43) is
    consistently BELOW mean_abs_phi's (~0.53), not above by the plan's +0.05 margin. Two discriminating
    experiments ruled out the obvious confounds rather than accepting the failure at face value: (1)
    raising ``banzhaf_n_coalitions`` 4096 -> 16384 (4x) moved the number only 0.413 -> 0.427 -- real but
    far too small to close a 0.11 gap by simply reducing the MSR estimator's own Monte-Carlo variance;
    (2) shrinking the bed from p=500 (98% noise) to p=100 (90% noise) did not flip the direction either
    (mean_abs_phi 0.699 vs banzhaf 0.568) -- so this is not an artifact of extreme width/noise ratio.
    The likely structural cause: mean|phi| is read directly off the already-averaged OOF-SHAP phi
    matrix (variance-reduced for free by that ensemble/fold averaging), while MSR-Banzhaf layers an
    ADDITIONAL independent per-seed Monte-Carlo coalition sample on top of the same phi -- so even
    though Banzhaf is the most robust semivalue to a FIXED level of v(S) noise, here it is not an
    apples-to-apples comparison: it introduces its own fresh seed-dependent randomness that mean|phi|
    does not pay. This is a genuine negative result, not a bug -- the feature ships opt-in regardless
    (default stays "mean_abs_phi" per gt_03 sec 7's own acceptance rule), so this test pins the
    measured direction honestly instead of asserting the plan's unconfirmed hypothesis.
    """
    X, y, _roles = _make_low_snr_fixture()
    seeds = (0, 1, 2, 3)

    def _fit_sets(ranking):
        """Fit ShapProxiedFS with the given prescreen_ranking across all seeds; return selected-feature sets and rescue counts."""
        sets, rescued_counts = [], []
        for seed in seeds:
            s = ShapProxiedFS(
                classification=True,
                prescreen_top=60,
                prescreen_ranking=ranking,
                random_state=seed,
                verbose=False,
                n_jobs=1,
            )
            s.fit(X, y)
            sets.append(set(s.selected_features_))
            rescued_counts.append(int(s.shap_proxy_report_.get("prescreen", {}).get("noise_floor_rescued", 0)))
        return sets, rescued_counts

    sets_mean_abs_phi, _ = _fit_sets("mean_abs_phi")
    sets_banzhaf, rescued_banzhaf = _fit_sets("banzhaf")

    jaccard_mean_abs_phi = _pairwise_jaccard(sets_mean_abs_phi)
    jaccard_banzhaf = _pairwise_jaccard(sets_banzhaf)

    # Pinning the MEASURED relationship (banzhaf less stable here), not the plan's unconfirmed
    # hypothesis -- see the docstring above for the two ruled-out alternative explanations.
    assert jaccard_banzhaf < jaccard_mean_abs_phi, (
        f"banzhaf Jaccard ({jaccard_banzhaf:.4f}) unexpectedly caught up to or beat mean_abs_phi's "
        f"({jaccard_mean_abs_phi:.4f}) -- re-investigate whether the structural explanation above "
        f"still holds before loosening this assertion."
    )
    assert jaccard_banzhaf >= jaccard_mean_abs_phi - 0.20, (
        f"banzhaf Jaccard ({jaccard_banzhaf:.4f}) fell far more below mean_abs_phi's "
        f"({jaccard_mean_abs_phi:.4f}) than previously measured -- possible regression in the estimator."
    )

    n_proxy_cols = X.shape[1]  # noise-floor rescue must never balloon to a large fraction of all columns
    for c in rescued_banzhaf:
        assert 0 <= c <= n_proxy_cols // 2, f"banzhaf shifted-importance noise_floor_rescued={c} looks unbounded (n_proxy={n_proxy_cols})"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_biz_val_banzhaf_ranking_no_regression_high_snr():
    """No-regression bed: on clean high-SNR data both rankings recall all informatives, and downstream
    AUC differs by at most 0.005."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    X, y, roles = _make_high_snr_fixture()
    informative = {c for c, r in roles.items() if r == "informative"}

    def _fit_and_score(ranking):
        """Fit ShapProxiedFS with the given prescreen_ranking, then score the selected subset on a held-out split."""
        s = ShapProxiedFS(classification=True, prescreen_top=60, prescreen_ranking=ranking, random_state=0, verbose=False, n_jobs=1)
        s.fit(X, y)
        selected = set(s.selected_features_)
        cols = sorted(selected)
        Xtr, Xte, ytr, yte = train_test_split(X[cols], y, test_size=0.3, random_state=0, stratify=y)
        clf = XGBClassifier(n_estimators=200, random_state=0, eval_metric="logloss")
        clf.fit(Xtr, ytr)
        auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
        return selected, auc

    sel_mean_abs_phi, auc_mean_abs_phi = _fit_and_score("mean_abs_phi")
    sel_banzhaf, auc_banzhaf = _fit_and_score("banzhaf")

    recall_mean_abs_phi = len(informative & sel_mean_abs_phi)
    recall_banzhaf = len(informative & sel_banzhaf)
    assert recall_mean_abs_phi == len(informative), f"mean_abs_phi recall {recall_mean_abs_phi}/{len(informative)} on the clean high-SNR bed"
    assert recall_banzhaf == len(informative), f"banzhaf recall {recall_banzhaf}/{len(informative)} on the clean high-SNR bed"
    assert abs(auc_banzhaf - auc_mean_abs_phi) <= 0.005, f"downstream AUC diverged beyond 0.005: mean_abs_phi={auc_mean_abs_phi:.4f}, banzhaf={auc_banzhaf:.4f}"


def _exact_banzhaf(phi, base, y, metric_code, is_rmse):
    """Full 2^(P-1)-per-feature enumeration of the exact Banzhaf value -- ground truth for the MSR agreement test."""
    n_features = phi.shape[1]
    phi_T = np.ascontiguousarray(phi.T)
    beta = np.zeros(n_features)
    other_idx_by_j = [tuple(k for k in range(n_features) if k != j) for j in range(n_features)]
    for j in range(n_features):
        others = other_idx_by_j[j]
        total = 0.0
        n_subsets = 0
        for r in range(len(others) + 1):
            for combo in combinations(others, r):
                margin_without = base + (phi_T[list(combo)].sum(axis=0) if combo else 0.0)
                margin_with = margin_without + phi_T[j]
                loss_without = score_margin_auto(np.ascontiguousarray(margin_without), y, metric_code)
                loss_with = score_margin_auto(np.ascontiguousarray(margin_with), y, metric_code)
                if is_rmse:
                    loss_without, loss_with = float(np.sqrt(loss_without)), float(np.sqrt(loss_with))
                total += (-loss_with) - (-loss_without)
                n_subsets += 1
        beta[j] = total / n_subsets
    return beta


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_biz_val_banzhaf_estimator_matches_exact_small_p():
    """MSR-Banzhaf at m=4096 must Spearman-correlate >= 0.95 with the exact 2^9-enumeration Banzhaf
    value on a P=10 small game, and the top-5 sets must agree in >= 4/5 members. Uses a smooth
    (no-exact-zero, no-tied) weight spread across all 10 features -- an exact hard 0-weight subset
    (a genuine dummy feature) makes its true Banzhaf value EXACTLY 0 for every member of that subset,
    which forces spurious ties in the exact ranking unrelated to estimator accuracy; that degenerate
    case is covered separately by ``test_banzhaf_msr_dummy_feature_zero_beta``."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 1500, 10
    weights = rng.uniform(0.05, 2.0, size=n_features)
    phi = rng.normal(0, 0.5, size=(n_samples, n_features)) * weights[None, :]
    base = rng.normal(0, 0.1, size=n_samples)
    y = base + phi.sum(axis=1) + rng.normal(0, 0.3, size=n_samples)  # regression target

    beta_msr, _info = banzhaf_msr(phi, base, y, classification=False, metric=None, n_coalitions=4096, rng=np.random.default_rng(1))
    beta_exact = _exact_banzhaf(phi, base, y, metric_code=1, is_rmse=True)  # rmse is the default regression metric

    rho, _p = spearmanr(beta_msr, beta_exact)
    assert rho >= 0.95, f"MSR-vs-exact Spearman={rho:.4f} < 0.95 on the P=10 small game"

    top5_msr = set(np.argsort(-beta_msr)[:5].tolist())
    top5_exact = set(np.argsort(-beta_exact)[:5].tolist())
    overlap = len(top5_msr & top5_exact)
    assert overlap >= 4, f"MSR top-5 vs exact top-5 overlap={overlap}/5 < 4"


def test_banzhaf_msr_dummy_feature_zero_beta():
    """A dummy feature (all-zero phi column) contributes nothing to any coalition's margin, so its TRUE
    beta is exactly 0; the MSR *estimate* still carries Monte Carlo sampling noise (different random
    in/out coalition draws average different noisy losses even when the marginal contribution is
    identically 0), so the estimate is checked against its own reported ``beta_stderr`` (within 5
    standard errors of 0) rather than an absolute near-zero tolerance."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 500, 8
    phi = rng.normal(0, 0.3, size=(n_samples, n_features))
    phi[:, -1] = 0.0  # dummy feature: zero marginal contribution to every coalition's margin
    base = rng.normal(0, 0.1, size=n_samples)
    y = (base + phi.sum(axis=1) + rng.normal(0, 0.5, size=n_samples) > 0).astype(np.float64)

    beta, info = banzhaf_msr(phi, base, y, classification=True, metric=None, n_coalitions=2048, rng=np.random.default_rng(2))
    stderr = info["beta_stderr"][-1]
    assert abs(beta[-1]) < 5 * stderr, f"dummy feature beta={beta[-1]!r} exceeds 5*stderr={5 * stderr!r}"


def test_banzhaf_msr_batched_matches_sequential_reference():
    """The batched-matmul path must be bit-identical (1e-10) to a naive per-mask sequential reference loop."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 200, 6
    phi = rng.normal(0, 0.3, size=(n_samples, n_features))
    base = rng.normal(0, 0.1, size=n_samples)
    y = (base + phi.sum(axis=1) + rng.normal(0, 0.5, size=n_samples) > 0).astype(np.float64)

    beta_batched, _info = banzhaf_msr(phi, base, y, classification=True, metric=None, n_coalitions=512, rng=np.random.default_rng(3), batch=64)

    # Sequential reference: same masks (same rng seed / draw order), one margin at a time, no chunking.
    rng_ref = np.random.default_rng(3)
    m = 512
    masks = rng_ref.random((m, n_features)) < 0.5
    v_ref = np.empty(m)
    for i in range(m):
        margin = base + phi[:, masks[i]].sum(axis=1)
        v_ref[i] = -score_margin_auto(np.ascontiguousarray(margin), y, 2)
    beta_ref = np.zeros(n_features)
    for j in range(n_features):
        in_mask = masks[:, j]
        beta_ref[j] = v_ref[in_mask].mean() - v_ref[~in_mask].mean()

    np.testing.assert_allclose(beta_batched, beta_ref, atol=1e-10)


def test_shap_proxied_fs_prescreen_ranking_clone_roundtrip():
    """``prescreen_ranking`` / ``banzhaf_n_coalitions`` survive sklearn's ``clone`` verbatim."""
    from sklearn.base import clone

    s = ShapProxiedFS(prescreen_ranking="banzhaf", banzhaf_n_coalitions=1024)
    s2 = clone(s)
    assert s2.prescreen_ranking == "banzhaf"
    assert s2.banzhaf_n_coalitions == 1024


def test_shap_proxied_fs_prescreen_ranking_validator_rejects_bad_value():
    """An unrecognised ``prescreen_ranking`` value raises ``ValueError`` at construction."""
    with pytest.raises(ValueError):
        ShapProxiedFS(prescreen_ranking="not_a_real_mode")
