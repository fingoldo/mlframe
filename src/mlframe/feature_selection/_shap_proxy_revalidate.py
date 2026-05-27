"""Honest re-validation + trust diagnostics for the SHAP-proxied selector.

The proxy ranks subsets cheaply but is a *biased* estimator of a retrained model's quality (it
attributes the full model restricted to S, not a model retrained on S). Three guards close that gap:

  - ``proxy_trust_guard``: on a sample of anchor subsets, score BOTH the cheap proxy and an honest
    retrain, then report Spearman/Kendall rank-correlation + recall@k of proxy-top vs honest-top.
    Converts "trust me" into measured proxy fidelity on the user's own data; warns below a floor.
  - ``revalidate_top_n``: honestly retrains each proxy-top-N candidate and evaluates it on a holdout
    DISJOINT from the SHAP/objective data (avoids winner's curse over millions of combos). Returns
    the best / most stability-penalised subset, plus a same-size random-subset baseline for context.
  - ``importance_topk_ablation``: gating check that proxy search beats plain SHAP-importance-top-k
    (Aleksey Pichugin's concern: does this reduce to feature importance + a wrapper?).

Honest scoring uses sklearn metrics (not the hot path); rank correlation uses scipy. Lower = better
loss everywhere, to match the proxy objective.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone

from mlframe.feature_selection._shap_proxy_objective import coalition_margin, proxy_loss, resolve_metric

logger = logging.getLogger(__name__)


def _honest_loss(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric, seed=None,
                 inner_n_jobs=None):
    """Train ``model_template`` on selected feature columns; return holdout loss (lower=better)."""
    from sklearn.metrics import (
        brier_score_loss, log_loss, mean_absolute_error, roc_auc_score, root_mean_squared_error,
    )

    cols = list(idx)
    est = clone(model_template)
    if seed is not None and hasattr(est, "random_state"):
        try:
            est.set_params(random_state=int(seed))
        except (ValueError, TypeError):
            pass
    # Cap per-fit threads when many fits run in parallel (avoid CPU oversubscription).
    if inner_n_jobs is not None and hasattr(est, "n_jobs"):
        try:
            est.set_params(n_jobs=int(inner_n_jobs))
        except (ValueError, TypeError):
            pass
    est.fit(X_tr.iloc[:, cols], y_tr)
    if classification:
        p = est.predict_proba(X_ev.iloc[:, cols])[:, 1]
        if metric == "brier":
            return float(brier_score_loss(y_ev, p))
        if metric == "logloss":
            return float(log_loss(y_ev, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
        # auc
        if len(np.unique(y_ev)) < 2:
            return 1.0
        return float(1.0 - roc_auc_score(y_ev, p))
    pred = est.predict(X_ev.iloc[:, cols])
    if metric == "mae":
        return float(mean_absolute_error(y_ev, pred))
    return float(root_mean_squared_error(y_ev, pred))


def _parallel_honest_losses(tasks, model_template, X_tr, y_tr, X_ev, y_ev, classification, metric, n_jobs):
    """Run many independent honest retrains concurrently. xgb/lgbm release the GIL during training,
    so a threading backend shares the (large) DataFrames without per-task pickling; we cap each fit's
    own thread count so the outer pool x inner threads doesn't oversubscribe the cores."""
    import os

    if not tasks:
        return []
    n_cores = os.cpu_count() or 1
    if n_jobs in (None, 0):
        n_jobs = 1
    outer = n_cores if n_jobs == -1 else n_jobs
    outer = max(1, min(outer, len(tasks), n_cores))
    inner = max(1, n_cores // outer)
    if outer == 1:
        return [_honest_loss(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric,
                             seed=seed, inner_n_jobs=inner) for idx, seed in tasks]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=outer, prefer="threads")(
        delayed(_honest_loss)(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric,
                              seed=seed, inner_n_jobs=inner)
        for idx, seed in tasks
    )


def _sample_anchor_subsets(n_features, n_anchors, rng, min_card=1, max_card=None):
    max_card = n_features if max_card is None else min(max_card, n_features)
    anchors = set()
    guard = 0
    while len(anchors) < n_anchors and guard < n_anchors * 50:
        guard += 1
        k = int(rng.integers(min_card, max_card + 1))
        anchors.add(tuple(sorted(rng.choice(n_features, size=k, replace=False).tolist())))
    return [list(a) for a in anchors]


def proxy_trust_guard(
    phi, base, y_search, model_template, X_search, X_holdout, y_holdout,
    *, classification, metric=None, n_anchors=30, rng=None, min_card=1, max_card=None,
    spearman_floor=0.6, n_jobs=-1,
):
    """Measure proxy-vs-honest rank fidelity on anchor subsets. Returns a report dict."""
    from scipy.stats import kendalltau, spearmanr

    metric = resolve_metric(classification, metric)
    rng = np.random.default_rng(0) if rng is None else rng
    f = phi.shape[1]
    anchors = _sample_anchor_subsets(f, n_anchors, rng, min_card, max_card)

    proxy_losses = [proxy_loss(coalition_margin(phi, base, idx), y_search, metric) for idx in anchors]
    honest_losses = _parallel_honest_losses(
        [(idx, None) for idx in anchors], model_template, X_search, y_search, X_holdout, y_holdout,
        classification, metric, n_jobs)
    proxy_losses = np.asarray(proxy_losses)
    honest_losses = np.asarray(honest_losses)
    ok = np.isfinite(proxy_losses) & np.isfinite(honest_losses)
    proxy_losses, honest_losses = proxy_losses[ok], honest_losses[ok]

    sp = float(spearmanr(proxy_losses, honest_losses).statistic) if len(proxy_losses) > 2 else float("nan")
    kt = float(kendalltau(proxy_losses, honest_losses).statistic) if len(proxy_losses) > 2 else float("nan")
    # recall@k: do the proxy's best-k anchors overlap the honest best-k?
    k = max(1, len(proxy_losses) // 5)
    proxy_best = set(np.argsort(proxy_losses)[:k].tolist())
    honest_best = set(np.argsort(honest_losses)[:k].tolist())
    recall = len(proxy_best & honest_best) / k if k else float("nan")

    trustworthy = np.isfinite(sp) and sp >= spearman_floor
    report = dict(n_anchors=int(len(proxy_losses)), spearman=sp, kendall=kt,
                  recall_at_k=recall, k=int(k), spearman_floor=spearman_floor, trustworthy=bool(trustworthy))
    if not trustworthy:
        logger.warning(
            "ShapProxiedFS: proxy fidelity LOW (spearman=%.3f < floor=%.2f, recall@%d=%.2f). "
            "The SHAP-coalition proxy may mis-rank subsets on this data; treat the result with caution "
            "(consider a smaller exhaustive honest search or more selected features).",
            sp, spearman_floor, int(k), recall,
        )
    else:
        logger.info("ShapProxiedFS: proxy fidelity OK (spearman=%.3f, kendall=%.3f, recall@%d=%.2f).",
                    sp, kt, int(k), recall)
    return report


def revalidate_top_n(
    candidates, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, n_models=1, lambda_stab=0.5, parsimony_tol=0.02, rng=None, n_jobs=-1,
):
    """Honestly retrain each candidate subset on X_search, evaluate on the disjoint X_holdout.

    Returns ``(best_idx_tuple, ranked, baseline)`` where ``ranked`` is a list of dicts with proxy +
    honest loss, std, and stability-penalised score (``mean + lambda_stab * std``).

    Final selection uses a parsimony / one-standard-error rule (matching RFECV's philosophy): among
    candidates whose stable score is within ``parsimony_tol`` (relative) of the best, pick the one
    with the FEWEST features (tie-break: lower stable score). This counters the proxy's bias toward
    larger subsets -- a noise feature that buys <2% honest improvement should not be kept.
    """
    metric = resolve_metric(classification, metric)
    rng = np.random.default_rng(0) if rng is None else rng
    # Build all (subset, seed) retrain tasks up front so they run in one parallel batch.
    tasks, task_owner = [], []
    for ci, (_, idx) in enumerate(candidates):
        for _ in range(n_models):
            tasks.append((tuple(idx), int(rng.integers(0, 2**31 - 1))))
            task_owner.append(ci)
    losses = _parallel_honest_losses(tasks, model_template, X_search, y_search, X_holdout, y_holdout,
                                     classification, metric, n_jobs)
    per_candidate: dict[int, list[float]] = {}
    for owner, loss in zip(task_owner, losses):
        per_candidate.setdefault(owner, []).append(loss)

    ranked = []
    for ci, (proxy_loss_val, idx) in enumerate(candidates):
        scores = np.asarray(per_candidate[ci], dtype=np.float64)
        mean, std = float(scores.mean()), float(scores.std())
        ranked.append(dict(features=tuple(idx), proxy_loss=float(proxy_loss_val),
                           honest_loss=mean, honest_std=std, stable_score=mean + lambda_stab * std))
    ranked.sort(key=lambda d: d["stable_score"])
    if ranked:
        best_score = ranked[0]["stable_score"]
        threshold = best_score + parsimony_tol * abs(best_score)
        eligible = [d for d in ranked if d["stable_score"] <= threshold]
        chosen = min(eligible, key=lambda d: (len(d["features"]), d["stable_score"]))
        best_idx = chosen["features"]
    else:
        best_idx = ()

    # Same-size random-subset baseline for the winner (winner's-curse context).
    baseline = None
    if best_idx:
        k = len(best_idx)
        f = X_search.shape[1]
        rnd = tuple(sorted(rng.choice(f, size=k, replace=False).tolist()))
        baseline = dict(features=rnd, honest_loss=_honest_loss(
            model_template, X_search, y_search, X_holdout, y_holdout, list(rnd), classification, metric))
    return best_idx, ranked, baseline


def importance_topk_ablation(
    phi, proxy_best_idx, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None,
):
    """Compare the proxy-chosen subset against a SHAP-importance-top-k subset of the same size.

    Returns a dict with both honest losses and whether the proxy strictly wins (the method's
    unique-value gate vs plain SHAP global importance).
    """
    metric = resolve_metric(classification, metric)
    k = len(proxy_best_idx)
    importance = np.abs(phi).mean(axis=0)
    imp_idx = tuple(sorted(np.argsort(-importance)[:k].tolist()))
    proxy_honest = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout,
                                list(proxy_best_idx), classification, metric)
    imp_honest = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout,
                              list(imp_idx), classification, metric)
    return dict(proxy_features=tuple(proxy_best_idx), proxy_honest_loss=proxy_honest,
                importance_features=imp_idx, importance_honest_loss=imp_honest,
                proxy_wins=bool(proxy_honest < imp_honest), proxy_at_least_ties=bool(proxy_honest <= imp_honest))
