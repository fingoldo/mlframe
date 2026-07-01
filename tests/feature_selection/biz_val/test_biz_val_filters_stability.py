"""biz_value tests for StabilityMRMR decision params.

Each test pins a quantitative win of a constructor param on a noisy frame where a single-shot mRMR leaks false positives:
support_threshold (FP-rate monotone in the threshold), n_bootstraps (more resamples -> fewer FPs), sample_fraction (a too-high
fraction correlates the bootstraps and leaks noise). Baseline is the single-shot selector / the param's weaker value.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from mlframe.feature_selection.filters.stability import StabilityMRMR

pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)


class _NoisyTopK(BaseEstimator):
    """Keeps k columns by abs-corr to y; on small subsamples the corr ranking of irrelevant columns is unstable, so noise
    features intermittently enter support -- exactly the regime stability selection is designed to filter."""

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X, y):
        Xv = X.values if hasattr(X, "values") else np.asarray(X)
        yv = np.asarray(y)
        cors = np.array([abs(np.corrcoef(Xv[:, j], yv)[0, 1]) if np.std(Xv[:, j]) > 0 else 0.0 for j in range(Xv.shape[1])])
        self.support_ = np.argsort(-np.nan_to_num(cors))[: self.k]
        return self


def _noisy_signal_frame(seed: int = 2, n: int = 600, p: int = 20):
    """3 strong signal columns (0,1,2) + 17 pure-noise columns; y = 1[x0+x1+x2+eps > 0]."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, p))
    y = (X[:, 0] + X[:, 1] + X[:, 2] + rng.normal(0, 0.5, n) > 0).astype(int)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    return Xdf, pd.Series(y), {0, 1, 2}


def test_biz_val_stability_beats_single_shot_on_false_positives():
    """Stability selection (threshold 0.6) drives noise-feature FP to 0 where a single-shot top-k leaks ~2 noise features."""
    X, y, true = _noisy_signal_frame()
    rng = np.random.default_rng(2)
    n = X.shape[0]
    single_fp = []
    for _ in range(10):
        sub = rng.choice(n, int(0.75 * n), replace=False)
        est = _NoisyTopK(k=5).fit(X.iloc[sub], y.iloc[sub])
        single_fp.append(len(set(est.support_.tolist()) - true))
    mean_single_fp = float(np.mean(single_fp))

    stab = StabilityMRMR(_NoisyTopK(k=5), n_bootstraps=25, sample_fraction=0.75, support_threshold=0.6, random_state=0).fit(X, y)
    stab_fp = len(set(stab.support_.tolist()) - true)

    assert mean_single_fp >= 1.5, f"baseline single-shot should leak noise; got {mean_single_fp:.2f}"
    assert stab_fp == 0, f"stability selection should eliminate FPs; got {stab_fp}"
    assert true.issubset(set(stab.support_.tolist())), "all true signals must survive stability selection"


def test_biz_val_stability_support_threshold_monotone_fp_reduction():
    """Raising support_threshold monotonically removes false positives: thr=0.0 keeps all noise, thr=0.6 keeps only signals."""
    X, y, true = _noisy_signal_frame()
    fps = {}
    for t in (0.0, 0.3, 0.6):
        s = StabilityMRMR(_NoisyTopK(k=5), n_bootstraps=25, sample_fraction=0.75, support_threshold=t, random_state=0).fit(X, y)
        sel = set(s.support_.tolist())
        fps[t] = len(sel - true)
        assert true.issubset(sel), f"signals must survive at threshold {t}"
    assert fps[0.0] >= 10, f"thr=0 keeps every touched feature; FP={fps[0.0]}"
    assert fps[0.3] < fps[0.0]
    assert fps[0.6] == 0
    assert fps[0.0] - fps[0.6] >= 10


def test_biz_val_stability_n_bootstraps_reduces_false_positives():
    """More bootstraps stabilise the inclusion-probability estimate, so noise leaks shrink: mean FP falls from B=2 to B=40."""
    X, y, true = _noisy_signal_frame()

    def mean_fp(B):
        fps = [
            len(set(StabilityMRMR(_NoisyTopK(k=5), n_bootstraps=B, sample_fraction=0.6, support_threshold=0.6, random_state=s).fit(X, y).support_.tolist()) - true)
            for s in range(8)
        ]
        return float(np.mean(fps))

    fp_low = mean_fp(2)
    fp_high = mean_fp(40)
    assert fp_high <= fp_low - 0.3, f"more bootstraps should cut FP leakage; B=2 -> {fp_low:.2f}, B=40 -> {fp_high:.2f}"
    assert fp_high <= 0.6


def test_biz_val_stability_sample_fraction_too_high_correlates_bootstraps():
    """A near-1 sample_fraction makes every bootstrap nearly the full data -> correlated picks -> noise leaks; the default 0.75 stays clean."""
    X, y, true = _noisy_signal_frame()
    default = StabilityMRMR(_NoisyTopK(k=5), n_bootstraps=25, sample_fraction=0.75, support_threshold=0.6, random_state=0).fit(X, y)
    too_high = StabilityMRMR(_NoisyTopK(k=5), n_bootstraps=25, sample_fraction=0.95, support_threshold=0.6, random_state=0).fit(X, y)

    fp_default = len(set(default.support_.tolist()) - true)
    fp_high = len(set(too_high.support_.tolist()) - true)
    assert fp_default == 0, f"default sample_fraction should stay clean; FP={fp_default}"
    assert fp_high >= 2, f"near-1 sample_fraction should leak noise via correlated bootstraps; FP={fp_high}"
    assert true.issubset(set(default.support_.tolist()))


def test_biz_val_stability_balanced_target_skips_stratification_quota():
    """On a near-balanced target the default ``stratify=True`` must be a NO-OP vs ``stratify=False``: every class survives an unstratified subsample with
    overwhelming probability, so forcing per-class quotas only perturbs the subsample composition and can push a borderline noise feature over threshold.
    The stratify gate engages only when a class is genuinely at risk (rare-class regime), so here the two modes produce identical support / probabilities."""
    X, y, true = _noisy_signal_frame()  # balanced binary (~50/50), n=600
    kw = dict(n_bootstraps=25, sample_fraction=0.75, support_threshold=0.6, random_state=0)
    strat_on = StabilityMRMR(_NoisyTopK(k=5), stratify=True, **kw).fit(X, y)
    strat_off = StabilityMRMR(_NoisyTopK(k=5), stratify=False, **kw).fit(X, y)
    assert set(strat_on.support_.tolist()) == set(strat_off.support_.tolist()), (
        "stratify must not change support on a balanced target where no class is at risk"
    )
    assert np.array_equal(strat_on.selection_probabilities_, strat_off.selection_probabilities_)
    assert len(set(strat_on.support_.tolist()) - true) == 0
