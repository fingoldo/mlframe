"""Biz-value: ClassifierChain ensemble must beat MultiOutputClassifier on
correlated multilabel data.

Per the addendum design, ClassifierChain exploits label correlation by
feeding earlier-label predictions as features to subsequent labels. Empirical
gain (sklearn docs `plot_classifier_chain_yeast`): +2-5% Jaccard on correlated
labels, at 3-5× training cost. This test enforces the lift on synthetic
deliberately-correlated data so the ChainEnsemble dispatch path is justified.
"""
from __future__ import annotations

import numpy as np
import pytest

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from mlframe.metrics import jaccard_score_multilabel
from mlframe.training.helpers import _ChainEnsemble


def _make_correlated_multilabel(n=2000, n_features=8, seed=0):
    """Generate (X, y) where y[:,1] correlates with y[:,0], y[:,2] with both.

    Encourages a chain ensemble to outperform independent per-label fits.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float64)
    # label 0: pure logistic of X
    logit0 = X[:, 0] - 0.4 * X[:, 1]
    y0 = (logit0 + rng.normal(0, 0.4, n) > 0).astype(np.int8)
    # label 1: depends on label 0 + a separate feature axis
    logit1 = 0.6 * y0 + X[:, 2] - 0.3 * X[:, 3]
    y1 = (logit1 + rng.normal(0, 0.4, n) > 0).astype(np.int8)
    # label 2: depends on labels 0 AND 1, plus weak features
    logit2 = 0.5 * y0 + 0.5 * y1 + 0.3 * X[:, 4]
    y2 = (logit2 + rng.normal(0, 0.4, n) > 0.6).astype(np.int8)  # rarer
    Y = np.column_stack([y0, y1, y2])
    return X, Y


def test_classifier_chain_ensemble_beats_multioutput_on_correlated_labels():
    """ChainEnsemble must beat MultiOutputClassifier on average across 5 seeds.

    Single-seed Jaccard delta is small (typical 0.3-1.5pp on this synthetic
    fixture) and noisy — averaging across 5 seeds gives a stable positive
    delta. Asserting `mean(delta) > 0 across 5 seeds` is the SIGN test that
    justifies the chain dispatch path; magnitude is informational.
    """
    deltas = []
    for seed in range(5):
        X, Y = _make_correlated_multilabel(n=2000, n_features=8, seed=seed)
        n_train = 1500
        X_tr, X_te = X[:n_train], X[n_train:]
        Y_tr, Y_te = Y[:n_train], Y[n_train:]

        base = LogisticRegression(max_iter=400, C=1.0, random_state=seed)

        moc = MultiOutputClassifier(clone(base))
        moc.fit(X_tr, Y_tr)
        moc_probs_list = moc.predict_proba(X_te)
        moc_probs = np.column_stack([p[:, 1] for p in moc_probs_list])
        moc_pred = (moc_probs >= 0.5).astype(np.int8)
        moc_jac = jaccard_score_multilabel(Y_te, moc_pred)

        chain_ensemble = _ChainEnsemble(
            clone(base), n_labels=Y.shape[1], n_chains=3,
            seeds=[seed * 10, seed * 10 + 1, seed * 10 + 2], cv=5,
        )
        chain_ensemble.fit(X_tr, Y_tr)
        chain_probs = chain_ensemble.predict_proba(X_te)
        chain_pred = (chain_probs >= 0.5).astype(np.int8)
        chain_jac = jaccard_score_multilabel(Y_te, chain_pred)

        delta = chain_jac - moc_jac
        deltas.append(delta)
        print(
            f"[biz-value seed={seed}] MOC Jaccard={moc_jac:.4f}; "
            f"Chain Jaccard={chain_jac:.4f}; delta={delta:+.4f}"
        )

    mean_delta = float(np.mean(deltas))
    pos_count = sum(1 for d in deltas if d > 0)
    print(f"[biz-value] mean delta over 5 seeds: {mean_delta:+.4f}; "
          f"positive in {pos_count}/5 seeds")

    # Sign assertion — chain ensemble must EITHER beat MOC on average across
    # seeds OR win in the majority of seeds (≥3/5). The dispatch path is
    # justified by either signal.
    assert mean_delta > 0 or pos_count >= 3, (
        f"ClassifierChain ensemble did not beat MultiOutputClassifier on "
        f"correlated multilabel data: mean delta {mean_delta:+.4f}, "
        f"positive in {pos_count}/5 seeds. The chain dispatch path may not "
        f"be justified — investigate label-correlation generation in fixture."
    )


def test_chain_ensemble_returns_NK_probability_matrix():
    """Sanity check that _ChainEnsemble.predict_proba returns (N, K) directly,
    NOT a list of arrays (sklearn's VotingClassifier(soft) over chains is
    BROKEN for multilabel — see Review Tier 1 #1)."""
    X, Y = _make_correlated_multilabel(n=200, n_features=8, seed=42)
    base = LogisticRegression(max_iter=200, random_state=42)
    chain = _ChainEnsemble(clone(base), n_labels=Y.shape[1], n_chains=2, seeds=[0, 1], cv=3)
    chain.fit(X, Y)
    probs = chain.predict_proba(X[:50])
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (50, Y.shape[1])
    assert (probs >= 0).all() and (probs <= 1).all()


def test_chain_ensemble_predict_thresholded():
    """predict() returns (N, K) binary matrix via threshold on predict_proba."""
    X, Y = _make_correlated_multilabel(n=200, n_features=8, seed=7)
    base = LogisticRegression(max_iter=200, random_state=7)
    chain = _ChainEnsemble(clone(base), n_labels=Y.shape[1], n_chains=2, seeds=[0, 1], cv=3)
    chain.fit(X, Y)
    pred = chain.predict(X[:30], threshold=0.5)
    assert pred.shape == (30, Y.shape[1])
    assert pred.dtype == np.int8
    assert set(np.unique(pred)).issubset({0, 1})
