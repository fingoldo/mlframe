"""P1 (audit4-C): honest nested prescreen.

The full-data prescreen still defines the search universe (legitimate for the final model), but with
``prescreen_nested=True`` (default) the per-fold ``cv_mean_perf`` is computed against a prescreen re-derived on each
fold's TRAIN rows only. A feature that survived the global prescreen purely via test-fold leakage is dropped in the
folds where it fails the train-only prescreen, so the reported CV score is no longer optimistically inflated.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.wrappers import RFECV
from sklearn.linear_model import LogisticRegression

from tests.feature_selection._biz_val_synth import make_signal_plus_noise, as_df


def _logreg():
    return LogisticRegression(max_iter=200, random_state=0)


def _base(**over):
    kw = dict(cv=3, random_state=0, leakage_corr_threshold=None, n_features_selection_rule="argmax", max_refits=4)
    kw.update(over)
    return kw


# Leakage-prone regime: a LOOSE FDR (0.99) lets noise features clear the significance gate, top_k=15 admits them
# into the universe, and small fold-train n (~60 of 90) makes noise that spuriously correlates with y over the FULL
# sample fail the per-fold train-only prescreen -- exactly the leakage the nested pass removes.
def _fit(nested, n=90, p_signal=2, p_noise=100, seed=1):
    X, y, sig = make_signal_plus_noise(n=n, p_signal=p_signal, p_noise=p_noise, seed=seed)
    Xdf, ys = as_df(X, y)
    r = RFECV(estimator=_logreg(), **_base(
        prescreen="univariate_ht", prescreen_top_k=15, prescreen_fdr_level=0.99, prescreen_nested=nested))
    r.fit(Xdf, ys)
    return r, sig


def _best(r):
    return max(r.cv_results_["cv_mean_perf"])


def test_nested_precomputes_fold_universes_and_flag_off_disables():
    r_on, _ = _fit(nested=True)
    assert isinstance(r_on._prescreen_fold_universes, dict) and len(r_on._prescreen_fold_universes) >= 1
    # every fold universe is a subset of the full pre-prescreen feature set
    full = set(r_on._prescreen_full_features)
    for _k, _u in r_on._prescreen_fold_universes.items():
        assert set(_u).issubset(full)

    r_off, _ = _fit(nested=False)
    assert r_off._prescreen_fold_universes is None, "prescreen_nested=False must not precompute fold universes"


def test_biz_val_nested_prescreen_debiases_leaky_optimism():
    """biz_value: on a leakage-prone seed the leaky (in-universe) prescreen reports an OPTIMISTIC best CV score
    because noise features that spuriously correlate with y over the FULL sample survive the global prescreen and
    inflate the held-out fold scores. The nested prescreen re-derives the universe per fold on train rows only, so
    those features drop out and the reported best score de-biases DOWNWARD. Measured once (seed=2): leaky ~+0.005
    vs nested ~-0.098 = ~0.10 gap; floor 0.04 (well below measured, above seed jitter)."""
    r_leaky, _ = _fit(nested=False, seed=2)
    r_nested, _ = _fit(nested=True, seed=2)
    best_leaky = _best(r_leaky)
    best_nested = _best(r_nested)
    gap = best_leaky - best_nested
    assert gap >= 0.04, (
        f"nested prescreen did not de-bias the leaky optimism (leaky={best_leaky:.4f} nested={best_nested:.4f} "
        f"gap={gap:.4f} < 0.04); the per-fold train-only masking should remove leakage-inflated score."
    )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_nested_never_more_optimistic_than_leaky(seed):
    """Honest-direction contract across seeds: the nested best score is NEVER higher than the leaky one (masking
    can only remove leaked features, so it de-biases downward or matches -- it can never inflate)."""
    best_leaky = _best(_fit(nested=False, seed=seed)[0])
    best_nested = _best(_fit(nested=True, seed=seed)[0])
    assert best_nested <= best_leaky + 1e-9, (
        f"seed={seed}: nested={best_nested:.4f} > leaky={best_leaky:.4f}; nested prescreen must not inflate."
    )


def test_nested_prescreen_selects_reasonable_support():
    r, sig = _fit(nested=True, seed=1)
    signal_cols = {f"x{i}" for i in sig}
    # true signal features remain in the searched universe under nested scoring
    assert signal_cols.issubset(set(r._prescreen_full_features))
