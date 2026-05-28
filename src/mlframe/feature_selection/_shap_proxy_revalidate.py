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
from sklearn.metrics import (
    brier_score_loss, log_loss, mean_absolute_error, roc_auc_score, root_mean_squared_error,
)

from mlframe.feature_selection._shap_proxy_objective import coalition_margin, proxy_loss, resolve_metric

logger = logging.getLogger(__name__)


def _expand(idx, unit_to_members):
    """Map unit indices (proxy / phi space) to original member feature columns (honest-retrain space).

    Identity when ``unit_to_members is None`` (non-clustering mode: idx already are feature columns).
    In clustering mode the proxy ranks in unit space (one column per denoised cluster), but honest
    re-validation must train on the REAL member columns we actually deploy -- so a unit subset expands
    to the union of its clusters' member columns.
    """
    if unit_to_members is None:
        return list(idx)
    cols: list[int] = []
    for u in idx:
        cols.extend(int(c) for c in unit_to_members[int(u)])
    return sorted(set(cols))


class HonestLossCache:
    """Memoizes ``_honest_loss`` across the FOUR honest-retrain stages of one ``ShapProxiedFS.fit``.

    Within a single fit ``(X_tr, y_tr, X_ev, y_ev, model_template, metric, classification)`` are fixed,
    so a retrain's holdout loss is fully determined by the column subset + the model seed. The trust
    guard, the importance ablation, and within-cluster refine all use ``seed=None`` (the template's own
    fixed seed) and frequently re-evaluate the SAME large member subset (e.g. the chosen winner is
    retrained as the ablation's proxy baseline AND as within-cluster-refine's starting ``base``), so
    caching returns those identical results without a duplicate fit. Keyed on
    ``(frozenset(cols), seed)`` -- order-independent, so column permutations of one subset collide
    correctly. Random-seeded re-validation fits get distinct seeds and so are never wrongly merged.
    Thread-safe for the threading-backend parallel pool (dict get/set under a lock)."""

    __slots__ = ("_store", "_lock", "hits", "misses")

    def __init__(self):
        import threading

        self._store: dict = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(idx, seed):
        return (frozenset(int(c) for c in idx), seed)

    def get(self, idx, seed):
        key = self._key(idx, seed)
        with self._lock:
            if key in self._store:
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    def put(self, idx, seed, value):
        key = self._key(idx, seed)
        with self._lock:
            self._store[key] = value


def _honest_loss(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric, seed=None,
                 inner_n_jobs=None, cache=None):
    """Train ``model_template`` on selected feature columns; return holdout loss (lower=better).

    When ``cache`` (a :class:`HonestLossCache`) is supplied, an identical ``(cols, seed)`` retrain is
    served from the cache instead of refitting -- the same model on the same data with the same seed
    is deterministic, so the cached float is numerically identical to a fresh fit."""
    if cache is not None:
        hit = cache.get(idx, seed)
        if hit is not None:
            return hit

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
            loss = float(brier_score_loss(y_ev, p))
        elif metric == "logloss":
            loss = float(log_loss(y_ev, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
        elif len(np.unique(y_ev)) < 2:  # auc undefined on a single class
            loss = 1.0
        else:
            loss = float(1.0 - roc_auc_score(y_ev, p))
    else:
        pred = est.predict(X_ev.iloc[:, cols])
        loss = float(mean_absolute_error(y_ev, pred)) if metric == "mae" else float(root_mean_squared_error(y_ev, pred))
    if cache is not None:
        cache.put(idx, seed, loss)
    return loss


def _parallel_honest_losses(tasks, model_template, X_tr, y_tr, X_ev, y_ev, classification, metric, n_jobs,
                            cache=None):
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
                             seed=seed, inner_n_jobs=inner, cache=cache) for idx, seed in tasks]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=outer, prefer="threads")(
        delayed(_honest_loss)(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric,
                              seed=seed, inner_n_jobs=inner, cache=cache)
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
    spearman_floor=0.6, n_jobs=-1, unit_to_members=None, cache=None,
):
    """Measure proxy-vs-honest rank fidelity on anchor subsets. Returns a report dict.

    Anchors are sampled in proxy (unit) space; honest losses retrain on the expanded member columns,
    so this measures the most decision-relevant question: does the cheap unit-proxy rank subsets the
    way honestly-retrained real-feature models do?"""
    from scipy.stats import kendalltau, spearmanr

    metric = resolve_metric(classification, metric)
    rng = np.random.default_rng(0) if rng is None else rng
    f = phi.shape[1]
    anchors = _sample_anchor_subsets(f, n_anchors, rng, min_card, max_card)

    from mlframe.feature_selection._shap_proxy_calibrate import subset_redundancy

    proxy_losses = [proxy_loss(coalition_margin(phi, base, idx), y_search, metric) for idx in anchors]
    honest_losses = _parallel_honest_losses(
        [(_expand(idx, unit_to_members), None) for idx in anchors], model_template, X_search, y_search,
        X_holdout, y_holdout, classification, metric, n_jobs, cache=cache)
    cards = np.array([len(a) for a in anchors], dtype=np.float64)
    redunds = np.array([subset_redundancy(phi, a) for a in anchors], dtype=np.float64)
    proxy_losses = np.asarray(proxy_losses)
    honest_losses = np.asarray(honest_losses)
    ok = np.isfinite(proxy_losses) & np.isfinite(honest_losses)
    proxy_losses, honest_losses, cards, redunds = proxy_losses[ok], honest_losses[ok], cards[ok], redunds[ok]

    sp = float(spearmanr(proxy_losses, honest_losses).statistic) if len(proxy_losses) > 2 else float("nan")
    kt = float(kendalltau(proxy_losses, honest_losses).statistic) if len(proxy_losses) > 2 else float("nan")
    # recall@k: do the proxy's best-k anchors overlap the honest best-k?
    k = max(1, len(proxy_losses) // 5)
    proxy_best = set(np.argsort(proxy_losses)[:k].tolist())
    honest_best = set(np.argsort(honest_losses)[:k].tolist())
    recall = len(proxy_best & honest_best) / k if k else float("nan")

    trustworthy = np.isfinite(sp) and sp >= spearman_floor
    report = dict(n_anchors=int(len(proxy_losses)), spearman=sp, kendall=kt,
                  recall_at_k=recall, k=int(k), spearman_floor=spearman_floor, trustworthy=bool(trustworthy),
                  # Raw anchor pairs (proxy, honest, cardinality, redundancy) for the bias corrector.
                  _corrector_data=dict(proxy=proxy_losses.tolist(), honest=honest_losses.tolist(),
                                       cards=cards.tolist(), redund=redunds.tolist()))
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
    unit_to_members=None, cache=None,
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
    # Build all (subset, seed) retrain tasks up front so they run in one parallel batch. Honest
    # retrain happens on EXPANDED member columns; the candidate's unit indices are kept for the record.
    tasks, task_owner = [], []
    member_cols = []
    for ci, (_, idx) in enumerate(candidates):
        cols = _expand(idx, unit_to_members)
        member_cols.append(cols)
        for _ in range(n_models):
            tasks.append((cols, int(rng.integers(0, 2**31 - 1))))
            task_owner.append(ci)
    losses = _parallel_honest_losses(tasks, model_template, X_search, y_search, X_holdout, y_holdout,
                                     classification, metric, n_jobs, cache=cache)
    per_candidate: dict[int, list[float]] = {}
    for owner, loss in zip(task_owner, losses):
        per_candidate.setdefault(owner, []).append(loss)

    ranked = []
    for ci, (proxy_loss_val, idx) in enumerate(candidates):
        scores = np.asarray(per_candidate[ci], dtype=np.float64)
        mean, std = float(scores.mean()), float(scores.std())
        # Parsimony cardinality = deployed feature count (expanded members), not unit count.
        ranked.append(dict(features=tuple(idx), n_members=len(member_cols[ci]),
                           proxy_loss=float(proxy_loss_val),
                           honest_loss=mean, honest_std=std, stable_score=mean + lambda_stab * std))
    ranked.sort(key=lambda d: d["stable_score"])
    if ranked:
        best_score = ranked[0]["stable_score"]
        threshold = best_score + parsimony_tol * abs(best_score)
        eligible = [d for d in ranked if d["stable_score"] <= threshold]
        chosen = min(eligible, key=lambda d: (d["n_members"], d["stable_score"]))
        best_idx = chosen["features"]
    else:
        best_idx = ()

    # Same-size (in member columns) random-subset baseline for the winner (winner's-curse context).
    baseline = None
    if best_idx:
        k = len(_expand(best_idx, unit_to_members))
        f = X_search.shape[1]
        k = min(k, f)
        rnd = tuple(sorted(rng.choice(f, size=k, replace=False).tolist()))
        baseline = dict(features=rnd, honest_loss=_honest_loss(
            model_template, X_search, y_search, X_holdout, y_holdout, list(rnd), classification, metric,
            cache=cache))
    return best_idx, ranked, baseline


def active_learning_revalidate(
    candidates, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, corrector_data, phi, budget, batch=4, n_models=1,
    parsimony_tol=0.02, rng=None, n_jobs=-1, unit_to_members=None, cache=None,
):
    """Disagreement-driven honest re-validation (lever #4).

    Instead of honestly retraining the proxy's static top-N, iterate: fit the bias corrector on the
    anchors seen so far, pick the ``batch`` un-evaluated candidates where the corrector most disagrees
    with the raw proxy (the proxy is least trustworthy there), honestly retrain them, fold the results
    back into the corrector, and repeat until ``budget`` candidates have been evaluated. This spends a
    fixed retrain budget where it most reduces winner's-curse risk. The proxy's top-1 is always among
    the first evaluated, so the result is never worse than naive top-1.

    Returns ``(best_idx, ranked, n_evaluated)``.
    """
    from mlframe.feature_selection._shap_proxy_calibrate import fit_proxy_corrector, subset_redundancy

    metric = resolve_metric(classification, metric)
    rng = np.random.default_rng(0) if rng is None else rng
    cd = {k: list(v) for k, v in corrector_data.items()}  # mutable copy we augment each round
    proxy_all = np.array([c[0] for c in candidates], dtype=np.float64)
    idxs = [c[1] for c in candidates]
    cards_all = np.array([len(i) for i in idxs], dtype=np.float64)
    redund_all = np.array([subset_redundancy(phi, i) for i in idxs], dtype=np.float64)
    member_cols = [_expand(i, unit_to_members) for i in idxs]

    honest = {}  # candidate index -> mean honest loss
    budget = min(budget, len(candidates))
    while len(honest) < budget:
        corrector = fit_proxy_corrector(cd["proxy"], cd["honest"], cd["cards"], cd["redund"])
        pred = corrector.predict(proxy_all, cards_all, redund_all)
        disagree = np.abs(pred - proxy_all)  # 0 under fallback -> falls back to proxy ordering
        remaining = [i for i in range(len(candidates)) if i not in honest]
        remaining.sort(key=lambda i: (-disagree[i], pred[i]))
        pick = remaining[: max(1, min(batch, budget - len(honest)))]
        if not pick:
            break
        tasks = [(member_cols[i], int(rng.integers(0, 2**31 - 1))) for i in pick for _ in range(n_models)]
        losses = _parallel_honest_losses(tasks, model_template, X_search, y_search, X_holdout, y_holdout,
                                         classification, metric, n_jobs, cache=cache)
        for j, i in enumerate(pick):
            seg = losses[j * n_models:(j + 1) * n_models]
            m = float(np.mean(seg))
            honest[i] = m
            cd["proxy"].append(float(proxy_all[i]))
            cd["honest"].append(m)
            cd["cards"].append(float(cards_all[i]))
            cd["redund"].append(float(redund_all[i]))

    ranked = [dict(features=tuple(idxs[i]), n_members=len(member_cols[i]), proxy_loss=float(proxy_all[i]),
                   honest_loss=honest[i], honest_std=0.0, stable_score=honest[i]) for i in honest]
    ranked.sort(key=lambda d: d["stable_score"])
    best_idx = ()
    if ranked:
        best_score = ranked[0]["stable_score"]
        thr = best_score + parsimony_tol * abs(best_score)
        eligible = [d for d in ranked if d["stable_score"] <= thr]
        best_idx = min(eligible, key=lambda d: (d["n_members"], d["stable_score"]))["features"]
    return best_idx, ranked, len(honest)


def within_cluster_refine(
    member_cols, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, parsimony_tol=0.02, n_jobs=-1, max_drop_rounds=None, cache=None,
):
    """Greedy-backward prune of the selected clusters' member columns (honest-evaluated).

    After cluster-aware search picks a set of units, expanding them yields ALL members of the chosen
    clusters - often redundant (the whole point of a cluster). This drops members one at a time while
    the honest holdout loss stays within ``parsimony_tol`` of the best seen, yielding a compact real-
    feature subset (e.g. one representative per cluster) without re-inflating the feature count.
    Returns the refined member column list.
    """
    metric = resolve_metric(classification, metric)
    current = sorted(set(int(c) for c in member_cols))
    if len(current) <= 1:
        return current
    base = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout, current, classification,
                        metric, cache=cache)
    rounds = len(current) if max_drop_rounds is None else max_drop_rounds
    for _ in range(rounds):
        if len(current) <= 1:
            break
        trials = [[c for c in current if c != drop] for drop in current]
        losses = _parallel_honest_losses([(t, None) for t in trials], model_template, X_search, y_search,
                                         X_holdout, y_holdout, classification, metric, n_jobs, cache=cache)
        best_i = int(np.argmin(losses))
        if losses[best_i] <= base + parsimony_tol * abs(base):  # dropping this member does not hurt
            current = trials[best_i]
            base = min(base, float(losses[best_i]))
        else:
            break
    return current


def importance_topk_ablation(
    phi, proxy_best_idx, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, unit_to_members=None, cache=None,
):
    """Compare the proxy-chosen subset against a SHAP-importance-top-k subset of the same size.

    Returns a dict with both honest losses and whether the proxy strictly wins (the method's
    unique-value gate vs plain SHAP global importance). In clustering mode, importance ranks UNITS
    and both subsets expand to member columns for the honest comparison.
    """
    metric = resolve_metric(classification, metric)
    k = len(proxy_best_idx)  # match unit count, then expand both sides to members
    importance = np.abs(phi).mean(axis=0)
    imp_units = tuple(sorted(np.argsort(-importance)[:k].tolist()))
    proxy_cols = _expand(proxy_best_idx, unit_to_members)
    imp_cols = _expand(imp_units, unit_to_members)
    proxy_honest = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout,
                                proxy_cols, classification, metric, cache=cache)
    imp_honest = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout,
                              imp_cols, classification, metric, cache=cache)
    return dict(proxy_features=tuple(proxy_best_idx), proxy_honest_loss=proxy_honest,
                importance_features=imp_units, importance_honest_loss=imp_honest,
                proxy_wins=bool(proxy_honest < imp_honest), proxy_at_least_ties=bool(proxy_honest <= imp_honest))
