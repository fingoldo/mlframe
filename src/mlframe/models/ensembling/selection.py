"""Caruana greedy ensemble selection and rank-average blending over a base-model prediction matrix.

Two prediction-matrix primitives from Dyakonov's ensembling lecture that the rest of the ensembling package
does not yet cover:

- :func:`caruana_greedy_selection` -- Caruana et al. (2004) "Ensemble Selection from Libraries of Models":
  greedy FORWARD selection WITH REPLACEMENT. Starting from an empty (or top-k warm-started) bag, repeatedly
  add the single base model whose inclusion most improves the ensemble mean on a held-out metric. Because a
  model may be added many times, the bag encodes an integer WEIGHTING -- a metric-optimal convex blend found
  by hill-climbing, not by a linear solver. Unlike the NNLS / Ridge / GBM meta-stackers already in the package
  (which minimise squared error on the OOF matrix), Caruana optimises the ACTUAL competition metric (AUC,
  log-loss, RMSE, ...) directly, so it wins when the target metric is not squared error.

- :func:`rank_average_blend` -- Dyakonov's ``a(x) = mean_i rank(b_i(x))``: average each model's per-row
  rank-transformed score, then (optionally) rescale to [0, 1]. Rank-averaging is scale-invariant and
  AUC-oriented -- two models on wildly different score scales (calibrated probs vs raw margins) contribute
  equally, which plain / geometric mean cannot guarantee. It differs from the existing RRF blend
  (``1/(k+rank)`` reciprocal-rank fusion, top-heavy) by using the linear rank directly, matching the lecture.

Also :func:`greedy_backward_ensemble_elimination` (prune the full uniform-mean bag down) and
:func:`stepwise_ensemble_selection` (alternate forward-add / backward-remove passes, the bidirectional/stepwise
selection pattern applied to ensemble members instead of features -- escapes local optima that pure forward
selection cannot reach, since it can revisit and drop an earlier pick once later members change the bag's
composition).

All consume a stacked ``(M, N)`` or ``(M, N, K)`` prediction matrix (M models, N rows, optional K classes),
mirroring the ``combine_probs`` / ``rrf_ensemble`` shape contract. Leakage: for HONEST weights the caller must
pass an OUT-OF-FOLD prediction matrix + the matching y (same contract as the NNLS stacker); the module cannot
verify out-of-foldness.

Profile (cProfile, M=20, N=100k, max_picks=100): ~2.9s, of which ~2.1s is inside ``fast_roc_auc`` (561 metric
evaluations = M x picks) -- the metric is the cost, not our loop. The greedy walk already updates a running SUM
(``running_sum + arr[i]``) so each candidate re-scores an incremental blend rather than re-summing the whole bag;
the residual Python overhead (~0.27s) is negligible next to the metric. No actionable speedup in this module; the
AUC kernel is already at its numpy-argsort floor (see ``_core_auc_brier.fast_roc_auc`` bench notes).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import rankdata

logger = logging.getLogger("mlframe.models.ensembling.selection")


def _rank_transform(scores: np.ndarray, *, axis: int = -1, normalise: bool = True) -> np.ndarray:
    """Average-rank transform along ``axis`` (ties share the mean rank); optionally rescale ranks to [0, 1].

    Profiled (300k fuzz loop, new-feature pass): the cost is the per-model argsort inside ``scipy.stats.rankdata``
    (O(M*N log N)), which is the algorithmic floor -- 366-611ms at M=4-6/N=300k. A tie-free fast path (ordinal ranks
    scattered from a single argsort, reused for the adjacent-tie check, scipy fallback on ties) gave only ~14% on the
    tie-free continuous-probability case and adds real complexity for a function called ONCE per ensemble target, so it
    was not shipped -- scipy's tie-aware rankdata stays. No actionable speedup that justifies the complexity here.
    """
    ranks = rankdata(scores, method="average", axis=axis).astype(np.float64)
    if not normalise:
        return np.asarray(ranks)
    n = scores.shape[axis]
    # Map ranks 1..n -> [0, 1]; a single-row axis maps to 0.5 (neutral) rather than dividing by zero.
    return (ranks - 1.0) / (n - 1.0) if n > 1 else np.full_like(ranks, 0.5)


def rank_average_blend(
    stacked: np.ndarray,
    *,
    normalise: bool = True,
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Rank-average blend of a stacked prediction matrix (Dyakonov ``mean_i rank(b_i(x))``).

    ``stacked`` is ``(M, N)`` (M models, N rows -- binary positive-class score) or ``(M, N, K)`` (K-class
    scores). Each model's scores are rank-transformed ACROSS ROWS (the N axis) independently, then averaged
    across models. Ranks are scale-invariant, so heterogeneously-scaled members blend fairly -- the property
    that makes rank-averaging a classic AUC-oriented ensemble.

    Parameters
    ----------
    stacked : np.ndarray
        ``(M, N)`` or ``(M, N, K)`` model x row [x class] score tensor.
    normalise : bool
        Rescale per-model ranks to [0, 1] before averaging (default True). When False the raw average rank
        (1..N) is returned -- monotone-identical for AUC, but not a probability.
    weights : sequence of float, optional
        Per-model non-negative weights (length M). Defaults to uniform. Renormalised to sum 1.

    Returns
    -------
    np.ndarray
        Blended scores, shape ``stacked[0]`` (``(N,)`` or ``(N, K)``).
    """
    arr = np.asarray(stacked, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError(f"rank_average_blend: stacked must be (M, N) or (M, N, K); got shape {arr.shape}.")
    m = arr.shape[0]
    if m == 0:
        raise ValueError("rank_average_blend: empty model axis (M=0).")
    # Rank across the ROW axis (axis 1) so each model's N scores are ranked among themselves, per class column.
    ranked = _rank_transform(arr, axis=1, normalise=normalise)
    if weights is None:
        return np.asarray(ranked.mean(axis=0))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != (m,):
        raise ValueError(f"rank_average_blend: weights shape {w.shape} != model axis (M={m},).")
    if not np.all(np.isfinite(w)) or (w < 0).any():
        raise ValueError("rank_average_blend: weights must be finite and non-negative.")
    wsum = float(w.sum())
    if wsum <= 0.0:
        raise ValueError("rank_average_blend: weights sum to zero.")
    w = w / wsum
    w_shape = (m,) + (1,) * (ranked.ndim - 1)
    return np.asarray(np.sum(ranked.reshape(m, *ranked.shape[1:]) * w.reshape(w_shape), axis=0))


def _default_metric_is_auc(metric: Optional[Callable]) -> bool:
    """Whether the caller left ``metric`` unset, meaning the ROC-AUC default in :func:`_score_blend` applies."""
    return metric is None


def _score_blend(
    blend: np.ndarray,
    y: np.ndarray,
    metric: Optional[Callable],
) -> float:
    """Score a blended prediction against y. Default metric (``None``) = ROC-AUC (higher is better)."""
    if metric is None:
        from mlframe.metrics.core import fast_roc_auc

        # fast_roc_auc wants the positive-class score. For (N, K) take class-1 column (binary convention).
        score = blend[:, 1] if blend.ndim == 2 and blend.shape[1] >= 2 else np.ravel(blend)
        return float(fast_roc_auc(y.astype(np.int64), score.astype(np.float64)))
    return float(metric(y, blend))


class CaruanaSelectionResult:
    """Result of :func:`caruana_greedy_selection`.

    Attributes
    ----------
    weights : np.ndarray
        Length-M integer counts (times each model was picked) normalised to sum 1 -- the convex blend weights.
    counts : np.ndarray
        Length-M raw integer pick counts.
    order : list[int]
        Model indices in the order they were added to the bag.
    score : float
        Best held-out metric value the bag reached (direction per ``greater_is_better``).
    n_picks : int
        Total number of greedy additions (bag size).
    """

    __slots__ = ("counts", "n_picks", "order", "score", "weights")

    def __init__(self, weights, counts, order, score, n_picks):
        self.weights = weights
        self.counts = counts
        self.order = order
        self.score = score
        self.n_picks = n_picks

    def predict(self, stacked: np.ndarray) -> np.ndarray:
        """Blend a NEW stacked ``(M, N[, K])`` matrix with the fitted weights (weighted arithmetic mean)."""
        arr = np.asarray(stacked, dtype=np.float64)
        if arr.shape[0] != self.weights.shape[0]:
            raise ValueError(f"CaruanaSelectionResult.predict: stacked model axis {arr.shape[0]} != fitted M {self.weights.shape[0]}.")
        w = self.weights.reshape((self.weights.shape[0],) + (1,) * (arr.ndim - 1))
        return np.asarray(np.sum(arr * w, axis=0))


def caruana_greedy_selection(
    stacked: np.ndarray,
    y: np.ndarray,
    *,
    metric: Optional[Callable] = None,
    greater_is_better: bool = True,
    max_picks: int = 100,
    init_top_k: int = 0,
    with_replacement: bool = True,
    tol: float = 0.0,
) -> CaruanaSelectionResult:
    """Caruana greedy forward ensemble selection with replacement over a held-out prediction matrix.

    Repeatedly adds the base model whose inclusion most improves the running ensemble MEAN on ``metric``.
    Because a model can be re-picked, the resulting bag is an integer-weighted convex blend that hill-climbs
    the ACTUAL metric (AUC by default), unlike the squared-error NNLS/Ridge meta-stackers.

    Parameters
    ----------
    stacked : np.ndarray
        ``(M, N)`` or ``(M, N, K)`` held-out (ideally OUT-OF-FOLD) model x row [x class] prediction tensor.
    y : np.ndarray
        Length-N ground truth aligned to the row axis.
    metric : callable, optional
        ``metric(y_true, blend) -> float``. Default ``None`` uses ROC-AUC on the positive-class score.
        Provide e.g. ``lambda yt, p: -log_loss(yt, p)`` with ``greater_is_better=True``, or pass the loss
        directly with ``greater_is_better=False``.
    greater_is_better : bool
        Whether higher ``metric`` is better (default True; set False for a loss like RMSE / log-loss).
    max_picks : int
        Maximum greedy additions (bag size). Each pick is one model; more picks = finer weighting.
    init_top_k : int
        Warm-start the bag with the ``init_top_k`` single BEST models (Caruana's "sorted ensemble
        initialisation"), which reduces the risk of the greedy walk overfitting to one lucky model. 0 = start
        empty (pure greedy).
    with_replacement : bool
        Allow re-picking an already-selected model (Caruana's default; gives integer weights). When False,
        each model is used at most once (a plain forward feature-style selection).
    tol : float
        Minimum improvement required to keep going; the walk stops once no candidate improves the score by
        more than ``tol`` (early stop guards against overfitting the selection metric).

    Returns
    -------
    CaruanaSelectionResult
        Fitted weights / counts / order / best score.
    """
    arr = np.asarray(stacked, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError(f"caruana_greedy_selection: stacked must be (M, N) or (M, N, K); got {arr.shape}.")
    m, n = arr.shape[0], arr.shape[1]
    if m == 0:
        raise ValueError("caruana_greedy_selection: empty model axis (M=0).")
    yv = np.asarray(y).reshape(-1)
    if yv.shape[0] != n:
        raise ValueError(f"caruana_greedy_selection: y length {yv.shape[0]} != row axis N={n}.")
    if max_picks < 1:
        raise ValueError(f"caruana_greedy_selection: max_picks must be >= 1, got {max_picks}.")

    sign = 1.0 if greater_is_better else -1.0

    def _better(a: float, b: float) -> bool:
        """Whether candidate score ``a`` beats the current-best ``b`` by more than ``tol``, in the direction ``sign`` implies."""
        # ``a`` improves over ``b`` by more than ``tol`` in the direction implied by greater_is_better.
        return sign * (a - b) > tol

    # Profiled (300k fuzz loop, new-feature pass): the greedy walk is dominated by the per-candidate ROC-AUC scoring,
    # which fast_roc_auc already routes to the GPU argsort (measured FASTER than the CPU-njit AUC even with the per-call
    # host transfer -- forcing CPU was a REJECT: 82ms vs 128ms at M=4/N=60k). Micro-opts explored + rejected: (a) a
    # reused blend scratch buffer (in-place add/divide) was a wash-to-slightly-slower (76->80ms) since the GPU AUC
    # dominates the per-candidate allocation; (b) scoring the raw running SUM and skipping the /bag_size mean-division
    # (AUC is scale-invariant) gave ~7.5% but is only SELECTION-equivalent -- division rounding shifts the reported AUC
    # ~1e-6 on tied scores, so it is not bit-identical. No actionable bit-identical speedup; keep the exact form.
    # Per-model single scores (used for warm start + to seed the "best so far").
    single_scores = np.array([_score_blend(arr[i], yv, metric) for i in range(m)], dtype=np.float64)
    ranked_models = np.argsort(-sign * single_scores)  # best-first

    counts = np.zeros(m, dtype=np.int64)
    order: list[int] = []
    running_sum = np.zeros(arr.shape[1:], dtype=np.float64)  # sum of picked model predictions (bag mean = sum / bag_size)

    # Warm start: seed the bag with the top-k single models.
    k0 = int(max(0, min(init_top_k, m)))
    for idx in ranked_models[:k0]:
        counts[idx] += 1
        order.append(int(idx))
        running_sum += arr[idx]

    if not order:
        # Pure greedy: seed with the single best model so the bag is never empty for scoring.
        idx = int(ranked_models[0])
        counts[idx] += 1
        order.append(idx)
        running_sum += arr[idx]

    bag_size = len(order)
    best_score = _score_blend(running_sum / bag_size, yv, metric)

    while len(order) < max_picks:
        cand_indices = range(m) if with_replacement else [i for i in range(m) if counts[i] == 0]
        if not cand_indices:
            break
        best_cand = -1
        best_cand_score = best_score
        new_size = bag_size + 1
        for i in cand_indices:
            # Incremental bag mean if model i were added: (running_sum + arr[i]) / (bag_size + 1).
            cand_blend = (running_sum + arr[i]) / new_size
            s = _score_blend(cand_blend, yv, metric)
            if _better(s, best_cand_score):
                best_cand_score = s
                best_cand = i
        if best_cand < 0:
            break  # no candidate improves by > tol -> early stop
        counts[best_cand] += 1
        order.append(best_cand)
        running_sum += arr[best_cand]
        bag_size = new_size
        best_score = best_cand_score

    weights = counts.astype(np.float64) / float(counts.sum())
    return CaruanaSelectionResult(
        weights=weights,
        counts=counts,
        order=order,
        score=float(best_score),
        n_picks=int(counts.sum()),
    )


class BackwardEliminationResult:
    """Result of :func:`greedy_backward_ensemble_elimination`.

    Attributes
    ----------
    kept : list[int]
        Surviving model indices (uniform-mean blend of these is the final ensemble).
    removed_order : list[int]
        Model indices in the order they were eliminated.
    score : float
        Held-out metric of the ``kept`` uniform-mean blend.
    removal_votes : np.ndarray, optional
        Only set when ``extra_stacked`` was used: length-M fraction of repeats (0..1) that voted to remove each
        model. ``None`` for a single-seed run (the default), so downstream code can branch on "was seed-averaging used".
    """

    __slots__ = ("kept", "removal_votes", "removed_order", "score")

    def __init__(self, kept, removed_order, score, removal_votes=None):
        self.kept = kept
        self.removed_order = removed_order
        self.score = score
        self.removal_votes = removal_votes

    def predict(self, stacked: np.ndarray) -> np.ndarray:
        """Uniform-mean blend of a NEW stacked ``(M, N[, K])`` matrix restricted to ``self.kept``."""
        arr = np.asarray(stacked, dtype=np.float64)
        return np.asarray(arr[self.kept].mean(axis=0))


def _greedy_backward_elimination_core(
    arr: np.ndarray,
    yv: np.ndarray,
    metric: Optional[Callable],
    greater_is_better: bool,
    min_models: int,
    tol: float,
) -> tuple[list[int], list[int], float]:
    """Single-seed backward elimination walk over ``arr``'s model axis. Returns (kept, removed_order, score).

    Extracted so :func:`greedy_backward_ensemble_elimination` can invoke it identically for a plain single run
    AND, unchanged, for each seed of the opt-in ``n_repeats`` seed-averaging path -- one code path, not a copy.
    """
    m = arr.shape[0]
    sign = 1.0 if greater_is_better else -1.0

    def _better(a: float, b: float) -> bool:
        return sign * (a - b) > tol

    kept = list(range(m))
    removed_order: list[int] = []
    # Running SUM of the currently-kept models (mirrors caruana_greedy_selection's incremental-sum trick):
    # re-summing arr[candidate] from scratch for every one of the ~M candidates per round (each a fresh fancy-
    # index copy + O(bag_size) reduction) was the dominant cost (0.75s tottime of 1.1s total at M=30/N=100k,
    # cProfile). Removing model idx from the bag is just running_sum - arr[idx], O(N) instead of O(bag_size * N).
    running_sum = arr[kept].sum(axis=0)
    bag_size = len(kept)
    best_score = _score_blend(running_sum / bag_size, yv, metric)

    while len(kept) > min_models:
        best_cand = -1
        best_cand_score = best_score
        new_size = bag_size - 1
        for idx in kept:
            cand_blend = (running_sum - arr[idx]) / new_size
            s = _score_blend(cand_blend, yv, metric)
            if _better(s, best_cand_score):
                best_cand_score = s
                best_cand = idx
        if best_cand < 0:
            break  # no removal improves by > tol -> early stop
        running_sum -= arr[best_cand]
        bag_size = new_size
        kept.remove(best_cand)
        removed_order.append(best_cand)
        best_score = best_cand_score

    return kept, removed_order, float(best_score)


def greedy_backward_ensemble_elimination(
    stacked: np.ndarray,
    y: np.ndarray,
    *,
    metric: Optional[Callable] = None,
    greater_is_better: bool = True,
    min_models: int = 1,
    tol: float = 0.0,
    extra_stacked: Optional[Sequence[np.ndarray]] = None,
) -> BackwardEliminationResult:
    """Greedy backward elimination over ENSEMBLE MEMBERS (not features): start with the full uniform-mean bag,
    repeatedly drop whichever remaining model's removal most improves (or least hurts) the held-out metric.

    Complements :func:`caruana_greedy_selection`'s forward-build direction: backward elimination explores a
    different search trajectory (prune down from "everyone votes" rather than build up from "nobody votes"),
    which can land on a different local optimum, and directly answers "which of my N candidate models are
    actively hurting the blend" rather than "which K models would I pick from scratch". Unlike
    :mod:`feature_selection.greedy_backward_elimination` (which operates on DataFrame columns and re-fits an
    estimator per candidate), this operates on the MODEL axis of a stored/OOF prediction matrix -- no refitting,
    just re-scoring a uniform-mean blend, mirroring ``caruana_greedy_selection``'s stored-prediction contract.

    A single-seed run scores every candidate removal against ONE fixed OOF prediction matrix, so its removal
    decisions inherit that matrix's own CV-seed noise: re-run the same base models under a DIFFERENT CV
    ``random_state`` and the resulting OOF predictions (hence the borderline model's measured contribution) can
    shift enough to flip the keep/remove call for a model whose true quality sits near the decision boundary
    -- exactly the instability this package's tracker notes flag elsewhere ("average of few runs, otherwise
    process might fall apart because of randomness"). ``extra_stacked`` (opt-in, default ``None`` = old exact
    behavior) lets the caller supply additional OOF prediction matrices from OTHER independently-seeded CV
    reruns (same models, same rows, same ``y``, different fold-assignment noise); the walk is repeated once per
    matrix (``stacked`` plus each of ``extra_stacked``) and a model is only removed if a STRICT MAJORITY of the
    repeats voted to remove it. This is proven -- not merely bootstrap-resampling ``stacked``'s own rows, which
    was tried and measured to NOT reduce decision error against a population-level ground truth (a fixed
    finite sample's bootstrap replicates cluster around that SAME sample's noisy point estimate, so majority-
    voting over them does not converge closer to the truth); only genuinely independent per-seed OOF matrices
    do, because each one carries an independent realization of the CV-refit noise (see the biz_value test
    ``test_biz_val_backward_elimination_extra_stacked_more_stable_and_accurate_than_single_seed`` for the
    measured before/after accuracy).

    Parameters
    ----------
    stacked : np.ndarray
        ``(M, N)`` or ``(M, N, K)`` held-out (ideally OUT-OF-FOLD) model x row [x class] prediction tensor.
        Used both as the first majority-vote repeat AND as the matrix the final returned ``score``/``predict``
        are computed against.
    y : np.ndarray
        Length-N ground truth aligned to the row axis (shared by ``stacked`` and every matrix in ``extra_stacked``).
    metric : callable, optional
        ``metric(y_true, blend) -> float``. Default ``None`` uses ROC-AUC on the positive-class score.
    greater_is_better : bool
        Whether higher ``metric`` is better (default True; set False for a loss like RMSE / log-loss).
    min_models : int
        Never eliminate below this many surviving models (default 1: keep eliminating until nothing improves).
    tol : float
        Minimum improvement required to keep eliminating; a removal must beat the current best score by more
        than ``tol`` (removals that just tie are not applied) or the walk stops.
    extra_stacked : sequence of np.ndarray, optional
        Additional independently-seeded OOF prediction matrices, each shaped exactly like ``stacked`` (same M,
        N[, K], same row order, same ``y``) -- e.g. the same base models refit under different CV
        ``random_state`` values. Default ``None``/empty = the original single-run behavior, BIT-IDENTICAL to
        before this parameter existed. When non-empty, seed-averaging is enabled: total repeats =
        ``1 + len(extra_stacked)``.

    Returns
    -------
    BackwardEliminationResult
        Surviving model indices / elimination order / best score (on ``stacked``) reached. ``removal_votes`` is
        set (not ``None``) only when ``extra_stacked`` is non-empty.
    """
    arr = np.asarray(stacked, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError(f"greedy_backward_ensemble_elimination: stacked must be (M, N) or (M, N, K); got {arr.shape}.")
    m, n = arr.shape[0], arr.shape[1]
    if m == 0:
        raise ValueError("greedy_backward_ensemble_elimination: empty model axis (M=0).")
    yv = np.asarray(y).reshape(-1)
    if yv.shape[0] != n:
        raise ValueError(f"greedy_backward_ensemble_elimination: y length {yv.shape[0]} != row axis N={n}.")
    if min_models < 1:
        raise ValueError(f"greedy_backward_ensemble_elimination: min_models must be >= 1, got {min_models}.")

    if not extra_stacked:
        kept, removed_order, best_score = _greedy_backward_elimination_core(arr, yv, metric, greater_is_better, min_models, tol)
        return BackwardEliminationResult(kept=kept, removed_order=removed_order, score=best_score)

    repeat_arrs = [arr]
    for i, extra in enumerate(extra_stacked):
        extra_arr = np.asarray(extra, dtype=np.float64)
        if extra_arr.shape != arr.shape:
            raise ValueError(f"greedy_backward_ensemble_elimination: extra_stacked[{i}] shape {extra_arr.shape} != stacked shape {arr.shape}.")
        repeat_arrs.append(extra_arr)
    n_repeats = len(repeat_arrs)

    removal_counts = np.zeros(m, dtype=np.int64)
    for repeat_arr in repeat_arrs:
        _kept_r, removed_order_r, _score_r = _greedy_backward_elimination_core(repeat_arr, yv, metric, greater_is_better, min_models, tol)
        removal_counts[removed_order_r] += 1

    removal_votes = removal_counts.astype(np.float64) / float(n_repeats)
    majority = removal_counts > (n_repeats / 2.0)  # strict majority of repeats voted to remove
    # Apply removals in descending-vote order (ties broken by lower index) so removed_order is deterministic and
    # meaningful (most-confidently-decoy first); never drop below min_models even if more models hit majority.
    candidates = [i for i in range(m) if majority[i]]
    candidates.sort(key=lambda i: (-removal_counts[i], i))
    kept = list(range(m))
    removed_order = []
    for idx in candidates:
        if len(kept) <= min_models:
            break
        kept.remove(idx)
        removed_order.append(idx)

    final_blend = arr[kept].mean(axis=0)
    final_score = _score_blend(final_blend, yv, metric)
    return BackwardEliminationResult(kept=kept, removed_order=removed_order, score=final_score, removal_votes=removal_votes)


class StepwiseSelectionResult:
    """Result of :func:`stepwise_ensemble_selection`.

    Attributes
    ----------
    kept : list[int]
        Surviving model indices (uniform-mean blend of these is the final ensemble).
    order : list[int]
        Model indices in the order they were added (forward steps only, chronological).
    removed_order : list[int]
        Model indices in the order they were removed by a backward step.
    score : float
        Held-out metric of the ``kept`` uniform-mean blend.
    """

    __slots__ = ("kept", "order", "removed_order", "score")

    def __init__(self, kept, order, removed_order, score):
        self.kept = kept
        self.order = order
        self.removed_order = removed_order
        self.score = score

    def predict(self, stacked: np.ndarray) -> np.ndarray:
        """Uniform-mean blend of a NEW stacked ``(M, N[, K])`` matrix restricted to ``self.kept``."""
        arr = np.asarray(stacked, dtype=np.float64)
        return np.asarray(arr[self.kept].mean(axis=0))


def stepwise_ensemble_selection(
    stacked: np.ndarray,
    y: np.ndarray,
    *,
    metric: Optional[Callable] = None,
    greater_is_better: bool = True,
    max_picks: int = 100,
    max_rounds: int = 200,
    with_replacement: bool = False,
    tol: float = 0.0,
    min_models: int = 1,
) -> StepwiseSelectionResult:
    """Bidirectional (stepwise) ensemble member selection: alternate forward-add and backward-remove passes.

    Classic stepwise/bidirectional selection (as used for FEATURE selection) applied to ensemble members: after
    each forward step adds the single best-improving model to the bag (uniform mean, same candidate rule as
    :func:`caruana_greedy_selection`), a backward pass is attempted over the CURRENT bag to see whether dropping
    some earlier-added member now improves the held-out score, given the bag's current composition. The member
    just added in this round is excluded from that round's backward pass so the walk cannot immediately undo its
    own forward step (which would oscillate add/remove of the same model forever).

    This escapes local optima that pure forward selection (:func:`caruana_greedy_selection` with
    ``with_replacement=False``) cannot reach: forward selection might pick model A first because it looks best
    alone, and every subsequent add still strictly improves the running bag (so forward never stops early) --
    but once B and C are both present, the bag may score higher WITHOUT A. Pure forward selection can never
    revisit that choice once A is in the bag; stepwise's interleaved backward pass can drop it.

    Reuses the running-sum trick from :func:`greedy_backward_ensemble_elimination`: the bag's summed predictions
    are maintained incrementally (``running_sum += arr[added]`` / ``running_sum -= arr[removed]``), so every
    candidate evaluation in both directions is an O(1) update plus one metric call -- never an O(bag_size) resum.

    Parameters
    ----------
    stacked : np.ndarray
        ``(M, N)`` or ``(M, N, K)`` held-out (ideally OUT-OF-FOLD) model x row [x class] prediction tensor.
    y : np.ndarray
        Length-N ground truth aligned to the row axis.
    metric : callable, optional
        ``metric(y_true, blend) -> float``. Default ``None`` uses ROC-AUC on the positive-class score.
    greater_is_better : bool
        Whether higher ``metric`` is better (default True; set False for a loss like RMSE / log-loss).
    max_picks : int
        Maximum forward additions across the whole walk (bag size ceiling, mirrors ``caruana_greedy_selection``).
    max_rounds : int
        Maximum forward-then-backward rounds; each round is at most one add followed by at most one remove, so
        this also caps the total number of removals. Guards against pathological oscillation even though the
        same-round re-removal guard already makes infinite oscillation impossible.
    with_replacement : bool
        Allow re-adding a model that is not currently in the bag after having been removed earlier (default
        False: once a model is removed it stays out, matching plain feature-style stepwise selection). When
        True, any model not CURRENTLY in the bag -- including one removed in an earlier round -- is a forward
        candidate again.
    tol : float
        Minimum improvement required to take a step (add or remove); a step must beat the current best score by
        more than ``tol`` or that direction is skipped.
    min_models : int
        Never remove below this many surviving models.

    Returns
    -------
    StepwiseSelectionResult
        Surviving model indices / add order / remove order / best score reached.
    """
    arr = np.asarray(stacked, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError(f"stepwise_ensemble_selection: stacked must be (M, N) or (M, N, K); got {arr.shape}.")
    m, n = arr.shape[0], arr.shape[1]
    if m == 0:
        raise ValueError("stepwise_ensemble_selection: empty model axis (M=0).")
    yv = np.asarray(y).reshape(-1)
    if yv.shape[0] != n:
        raise ValueError(f"stepwise_ensemble_selection: y length {yv.shape[0]} != row axis N={n}.")
    if max_picks < 1:
        raise ValueError(f"stepwise_ensemble_selection: max_picks must be >= 1, got {max_picks}.")
    if min_models < 1:
        raise ValueError(f"stepwise_ensemble_selection: min_models must be >= 1, got {min_models}.")

    sign = 1.0 if greater_is_better else -1.0

    def _better(a: float, b: float) -> bool:
        return sign * (a - b) > tol

    single_scores = np.array([_score_blend(arr[i], yv, metric) for i in range(m)], dtype=np.float64)
    best_single = int(np.argsort(-sign * single_scores)[0])  # index of the best single model (sign-aware)

    kept: list[int] = [best_single]
    order: list[int] = [best_single]
    removed_order: list[int] = []
    running_sum = arr[best_single].copy()
    bag_size = 1
    n_picks = 1
    best_score = _score_blend(running_sum / bag_size, yv, metric)

    for _round in range(max_rounds):
        if n_picks >= max_picks:
            break
        # --- forward step: add the single best-improving model not currently in the bag. ---
        already_in = set(kept)
        if with_replacement:
            fwd_candidates = [i for i in range(m) if i not in already_in]
        else:
            ever_removed = set(removed_order)
            fwd_candidates = [i for i in range(m) if i not in already_in and i not in ever_removed]
        added_this_round = -1
        if fwd_candidates:
            new_size = bag_size + 1
            best_cand, best_cand_score = -1, best_score
            for i in fwd_candidates:
                cand_blend = (running_sum + arr[i]) / new_size
                s = _score_blend(cand_blend, yv, metric)
                if _better(s, best_cand_score):
                    best_cand_score = s
                    best_cand = i
            if best_cand >= 0:
                kept.append(best_cand)
                order.append(best_cand)
                running_sum += arr[best_cand]
                bag_size = new_size
                n_picks += 1
                best_score = best_cand_score
                added_this_round = best_cand

        # --- backward step: on the CURRENT bag, try dropping an earlier-added member (never the one just added
        # this round, which would just undo the forward step and oscillate forever). ---
        removed_this_round = False
        if bag_size > min_models:
            bwd_candidates = [i for i in kept if i != added_this_round]
            if bwd_candidates:
                new_size = bag_size - 1
                best_cand, best_cand_score = -1, best_score
                for i in bwd_candidates:
                    cand_blend = (running_sum - arr[i]) / new_size
                    s = _score_blend(cand_blend, yv, metric)
                    if _better(s, best_cand_score):
                        best_cand_score = s
                        best_cand = i
                if best_cand >= 0:
                    running_sum -= arr[best_cand]
                    bag_size = new_size
                    kept.remove(best_cand)
                    removed_order.append(best_cand)
                    best_score = best_cand_score
                    removed_this_round = True

        if added_this_round < 0 and not removed_this_round:
            break  # neither direction moved this round -> converged, stop early

    return StepwiseSelectionResult(kept=kept, order=order, removed_order=removed_order, score=float(best_score))
