"""Faith-Shap (Tsai, Yeh, Ravikumar, JMLR 2023) order-2 interaction index over the proxy game.

Faith-Shap is the UNIQUE interaction index satisfying the natural extensions of the Shapley axioms
AND minimizing the Shapley-kernel-weighted least-squares reconstruction error of the coalition game --
the interaction analogue of "SHAP = weighted-least-squares projection". Unlike the raw TreeSHAP
interaction tensor (``compute_interaction_tensor``, O(P^2) memory/compute, hard-gated to P<=16), the
order-2 coefficients here are estimated by weighted ridge regression over SAMPLED coalitions of the
SAME additive proxy game (``v(S) = -subset_loss(S)``) -- no tensor, cost is
``O(n_coalitions * (1 + k + k_pairs))`` where the pairwise design is restricted to a screened candidate-
pair set (never the full k^2 -- that is explicitly rejected, see the plan's known-risks section: at
proxy width 112 a full pairwise design has 6216 columns against a ~2048-coalition sample, an
underdetermined system).
"""

from __future__ import annotations

from math import comb
from typing import Optional

import numpy as np


def faith_shap_order2(
    evaluator,
    n_features: int,
    candidate_pairs: list[tuple[int, int]],
    *,
    n_coalitions: int = 2048,
    rng: np.random.Generator,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, dict[tuple[int, int], float], dict]:
    """Weighted ridge regression estimate of order-2 Faith-Shap coefficients.

    Returns ``(a_lin (n_features,), a_pair {(i,j)->coef}, info)``. Design matrix per sampled
    coalition ``S``: ``[1 | 1{j in S} for all j | 1{i in S and j in S} for candidate pairs]``.
    Weights: the Shapley kernel ``mu(|S|) ~ (n_features-1) / (C(n_features,|S|) * |S| * (n_features-|S|))``;
    coalition SIZES are sampled from the kernel-normalized size distribution, then a uniform random
    subset of that size is drawn (standard KernelSHAP sampling). ``S=empty`` and ``S=full`` are always
    included with a large finite weight (``1e6``, the standard surrogate for the axioms' infinite-
    weight boundary constraints -- the kernel weight is mathematically infinite at those two sizes).
    ``v(S) = -evaluator.loss(S)`` (negate: game value = goodness; loss is lower-better). ``evaluator``
    treats the empty subset as invalid (``+inf`` loss by convention, see ``_shap_proxy_heuristics.py``);
    ``v(empty)`` is instead defined as the negative of the WORST single-feature loss among
    ``range(n_features)`` -- a finite, well-defined "no information" reference point (mirrors gt_02's
    ``project_chi2_ball``'s analogous worst-singleton reference-point choice for the same reason: the
    proxy game has no natural zero baseline of its own).

    The weighted ridge system is solved via ``np.linalg.lstsq`` on the sqrt-weighted, ridge-augmented
    design (append ``sqrt(ridge) * I`` rows with zero target -- the standard lstsq-ridge trick).
    """
    if n_features < 2:
        raise ValueError(f"faith_shap_order2: n_features must be >= 2, got {n_features}")
    n_pairs = len(candidate_pairs)
    n_dims = 1 + n_features + n_pairs

    # Pure-Python int/float arithmetic throughout: comb(n_features, s) is a Python arbitrary-precision
    # int that can exceed numpy's fixed-width int64 (max ~9.2e18) well before it exceeds float64's
    # range (~1.8e308) -- a numpy scalar (e.g. from np.arange) contaminating this expression silently
    # overflows int64 (RuntimeWarning, wrapped/corrupted value) instead of raising, which then produced
    # a downstream OverflowError at width=112 (measured while benchmarking). Plain Python ints avoid
    # the fixed-width cast entirely and are exact up to comb(n_features, n_features//2) for any
    # n_features this selector realistically sees (proxy widths in the hundreds, not thousands).
    sizes = list(range(1, n_features))
    kernel_w = np.array([(n_features - 1) / float(comb(n_features, s) * s * (n_features - s)) for s in sizes], dtype=np.float64)
    size_probs = kernel_w / kernel_w.sum()

    n_sampled = max(int(n_coalitions) - 2, 0)
    sampled_sizes = rng.choice(sizes, size=n_sampled, p=size_probs) if n_sampled > 0 else np.array([], dtype=int)

    coalitions: list[np.ndarray] = []
    weights: list[float] = []
    for size in sampled_sizes:
        members = rng.choice(n_features, size=int(size), replace=False)
        coalitions.append(np.sort(members))
        idx = int(size) - 1
        weights.append(float(kernel_w[idx] / size_probs[idx]))  # importance-reweight the drawn sample back to the kernel's own scale
    coalitions.append(np.array([], dtype=int))  # S = empty
    weights.append(1e6)
    coalitions.append(np.arange(n_features))  # S = full
    weights.append(1e6)

    # Prefer the evaluator's OWN empty-coalition value when it's finite (a well-defined game, e.g. the
    # analytic unit-test game below, may have a real v(empty)); only fall back to the worst-singleton
    # reference point when it's not (the production _Evaluator's documented convention: empty subset
    # -> +inf loss, "not a valid selection").
    _empty_loss = float(evaluator.loss([]))
    if np.isfinite(_empty_loss):
        v_empty = -_empty_loss
    else:
        worst_singleton_loss = max(float(evaluator.loss([j])) for j in range(n_features))
        v_empty = -worst_singleton_loss

    m = len(coalitions)
    X_design = np.zeros((m, n_dims), dtype=np.float64)
    y_target = np.empty(m, dtype=np.float64)
    X_design[:, 0] = 1.0
    for row, members in enumerate(coalitions):
        member_set = set(int(x) for x in members)
        X_design[row, 1 : 1 + n_features][list(member_set)] = 1.0 if member_set else 0.0
        for pi, (a, b) in enumerate(candidate_pairs):
            if a in member_set and b in member_set:
                X_design[row, 1 + n_features + pi] = 1.0
        if len(member_set) == 0:
            y_target[row] = v_empty
        else:
            y_target[row] = -float(evaluator.loss(sorted(member_set)))

    w = np.asarray(weights, dtype=np.float64)
    sqrt_w = np.sqrt(w)
    Xw = X_design * sqrt_w[:, None]
    yw = y_target * sqrt_w

    ridge_rows = np.sqrt(ridge) * np.eye(n_dims, dtype=np.float64)
    Xw_aug = np.vstack([Xw, ridge_rows])
    yw_aug = np.concatenate([yw, np.zeros(n_dims, dtype=np.float64)])

    beta, _residuals, _rank, _sv = np.linalg.lstsq(Xw_aug, yw_aug, rcond=None)

    a0 = float(beta[0])
    a_lin = np.asarray(beta[1 : 1 + n_features])
    a_pair = {pair: float(beta[1 + n_features + i]) for i, pair in enumerate(candidate_pairs)}

    y_pred = X_design @ beta
    ss_res = float(np.sum(w * (y_target - y_pred) ** 2))
    y_wmean = float(np.sum(w * y_target) / np.sum(w))
    ss_tot = float(np.sum(w * (y_target - y_wmean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    info = dict(a0=a0, n_coalitions=m, n_pairs=n_pairs, r2_of_fit=r2, v_empty=v_empty)
    return a_lin, a_pair, info


def faith_interaction_top_n(
    phi: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
    metric: Optional[str],
    candidate_pairs: list[tuple[int, int]],
    min_card: int,
    max_card: Optional[int],
    top_n: int,
    n_coalitions: int = 2048,
    rng: np.random.Generator,
) -> tuple[list[tuple[float, tuple[int, ...]]], dict]:
    """Rank candidate subsets by the order-2 Faith-Shap surrogate game
    ``v_hat(S) = a0 + sum_{j in S} a_j + sum_{(i,j) subset S} a_ij``.

    The surrogate closed form is O(k + |candidate_pairs intersecting S|) per subset -- cheap enough to
    greedily grow candidates directly over it rather than re-running the real (expensive) proxy loss:
    seed one growth path per candidate pair (starting from its 2 operands) and one from the single
    best-``a_j`` feature, then greedily add the locally-best-surrogate feature at each step up to
    ``max_card``, recording every intermediate size in ``[min_card, max_card]``. Returns
    ``(candidates, info)``: ``candidates`` is the top_n distinct ``(surrogate_loss, idx_tuple)`` pairs,
    ``surrogate_loss = -v_hat(S)`` (lower is better, to stay directly mergeable with the rest of the
    search's candidate list, which is loss-sorted ascending); ``info`` is :func:`faith_shap_order2`'s
    own diagnostic dict (``r2_of_fit`` is the key one -- low R^2 means the order-2 surrogate is a poor
    fit and callers should treat the ranking with caution).
    """
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_heuristics import _Evaluator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import resolve_metric

    n_features = phi.shape[1]
    max_card = n_features if max_card is None else min(int(max_card), n_features)
    metric_r = resolve_metric(classification, metric)
    evaluator = _Evaluator(phi, base, y, metric_r)

    a_lin, a_pair, info = faith_shap_order2(evaluator, n_features, candidate_pairs, n_coalitions=n_coalitions, rng=rng)

    def _v_hat(members: set) -> float:
        """Surrogate order-2 Faith-Shap value of the coalition ``members`` (closed form, no proxy loss call)."""
        val = float(np.sum(a_lin[list(members)])) if members else 0.0
        for i, j in candidate_pairs:
            if i in members and j in members:
                val += a_pair[(i, j)]
        return val

    seeds: list[set] = []
    for i, j in candidate_pairs:
        seeds.append({i, j})
    if n_features > 0:
        seeds.append({int(np.argmax(a_lin))})

    # Restrict the greedy EXTENSION pool to the top-ranked features by a_lin, not the full feature
    # space: with hundreds of noise columns the ridge fit's own estimation noise can give a handful
    # of pure-noise features a small positive a_lin by chance, and an unrestricted greedy scan over
    # ALL n_features will occasionally pick one up (measured: 2 spurious noise columns on a wide
    # pure-additive bed). Capping the pool to a generous multiple of max_card bounds that risk while
    # keeping every genuinely-informative feature (which ranks far above noise on a's magnitude) in
    # reach.
    extension_pool_size = min(n_features, max(3 * max_card, 20))
    extension_pool = np.argsort(-a_lin)[:extension_pool_size].tolist()

    results: dict[tuple[int, ...], float] = {}

    def _record(members: set) -> None:
        """Record ``members`` (if within [min_card, max_card]) into ``results`` keeping the best (lowest) surrogate loss."""
        if min_card <= len(members) <= max_card:
            key = tuple(sorted(members))
            loss = -_v_hat(members)
            if key not in results or loss < results[key]:
                results[key] = loss

    for seed in seeds:
        members = set(seed)
        _record(members)
        while len(members) < max_card:
            best_gain = -np.inf
            best_add: Optional[int] = None
            for k in extension_pool:
                if k in members:
                    continue
                gain = _v_hat(members | {k})
                if gain > best_gain:
                    best_gain = gain
                    best_add = k
            if best_add is None:
                break
            members = members | {best_add}
            _record(members)

    ranked = sorted(results.items(), key=lambda kv: kv[1])[:top_n]
    return [(loss, key) for key, loss in ranked], info


__all__ = ["faith_shap_order2", "faith_interaction_top_n"]
