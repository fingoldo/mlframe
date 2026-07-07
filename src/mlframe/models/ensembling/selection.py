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

Both consume a stacked ``(M, N)`` or ``(M, N, K)`` prediction matrix (M models, N rows, optional K classes),
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

    __slots__ = ("weights", "counts", "order", "score", "n_picks")

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
