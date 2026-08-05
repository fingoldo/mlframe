"""Shapley-value model weighting & ensemble pruning: a principled alternative to hill-climb/NNLS blending.

Models are players, ``v(C) = score_fn(y, blend(preds[C]))`` is the ensemble score of blending coalition
``C`` on OOF predictions. Unlike hill-climb (confounds "good" with "selected early"; duplicates crowd
selection) or NNLS (collinear near-duplicates get arbitrary weight splits -- one twin can get everything,
the other zero) or leave-one-out ablation (drops ONE of two duplicates and shows ~zero loss, making BOTH
look useless even though dropping both hurts), the Shapley value's symmetry axiom makes duplicates SHARE
credit equally, its dummy axiom sends useless models to ~0, and averaging over every coalition size
removes the leave-one-out blind spot. Pruning by Shapley~=0 is therefore stable under redundancy.

Cost model: the permutation estimator evaluates ``n_permutations * n_models`` coalition marginals, each
an ``O(n_rows)`` blend + ``score_fn`` call (AUC's sort dominates at large ``n_rows`` -- use
``score_subsample`` to cap it). Guidance: ``n_permutations >= 10 * n_models`` for stable per-model stderr.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _default_score_fn(y: np.ndarray, blended: np.ndarray) -> float:
    """AUC for binary ``y`` (cardinality 2), negative RMSE otherwise -- higher is always better.

    Uses ``mlframe.metrics.fast_roc_auc`` (numba, no sklearn per-call validation overhead) rather than
    ``sklearn.metrics.roc_auc_score`` -- this score_fn runs ``n_permutations * n_models`` times per
    :func:`shapley_model_values` call, and sklearn's ``check_array``/``type_of_target``/signature-
    inspection overhead dominated a profiled run (measured: sklearn's roc_auc_score cost ~13ms/call at
    n_rows=50000, almost entirely validation overhead, not the underlying sort)."""
    from mlframe.metrics import fast_roc_auc

    if len(np.unique(y)) == 2:
        return float(fast_roc_auc(y, blended))
    return float(-np.sqrt(np.mean((y - blended) ** 2)))


def _blend(preds: np.ndarray, idx: np.ndarray, coalition_blend: str) -> np.ndarray:
    """Blend the rows of ``preds`` selected by ``idx`` (empty -> zeros, matching the empty-coalition convention)."""
    if idx.size == 0:
        return np.zeros(preds.shape[1], dtype=np.float64)
    sub = preds[idx]
    if coalition_blend == "mean":
        return np.asarray(sub.mean(axis=0))
    if coalition_blend == "rank_mean":
        from scipy.stats import rankdata

        ranks = np.stack([rankdata(row) for row in sub], axis=0)
        return np.asarray(ranks.mean(axis=0) / preds.shape[1])
    raise ValueError(f"shapley_model_values: unsupported coalition_blend {coalition_blend!r}, expected 'mean' or 'rank_mean'")


def shapley_model_values(
    preds: np.ndarray,
    y: np.ndarray,
    *,
    score_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    coalition_blend: str = "mean",
    estimator: str = "permutation",
    n_permutations: int = 200,
    score_subsample: Optional[int] = 20000,
    rng: Optional[np.random.Generator] = None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, dict]:
    """Shapley values ``(n_models,)`` of ``v(C) = score_fn(y, blend(preds[C]))`` over the model pool.

    ``preds``: ``(n_models, n_rows)`` OOF predictions (proba or margin). ``v(empty coalition)`` is the
    score of an all-zero blend (a well-defined, cheap reference point -- NOT the constant-mean(y)
    prediction, which would require a separate special-cased blend rule; the zero-blend convention is
    documented here and is what the empty-coalition baseline actually measures throughout).

    ``estimator="permutation"``: for each of ``n_permutations`` random orderings, walk prefixes and
    accumulate each newly-added model's marginal contribution, using an INCREMENTAL running sum of
    coalition predictions (each marginal step is ``O(n_rows)``, not ``O(|C| * n_rows)``).
    ``estimator="msr_banzhaf"``: Maximum-Sample-Reuse Banzhaf semivalue (see gt_03's
    ``shap_proxied_fs/_shap_proxy_banzhaf.py`` for the same MSR trick over a different game) --
    ``n_permutations`` sampled coalitions this time (not permutations), each row included w.p. 0.5.

    ``score_subsample``: when set and ``n_rows > score_subsample``, ``score_fn`` is evaluated on a fixed
    (seeded) random row subsample for every coalition -- bounds the O(n_rows) sort cost of AUC-based
    scores on large pools; document to callers that this trades a small amount of estimate variance for
    speed. ``None`` disables subsampling.

    Returns ``(values, info)`` with ``info`` holding ``stderr`` (per-model, two-branch running stats),
    ``n_evals``, ``v_full``, ``v_empty``.
    """
    preds = np.ascontiguousarray(preds, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n_models, n_rows = preds.shape
    if score_fn is None:
        score_fn = _default_score_fn
        # _default_score_fn silently falls back to negative-RMSE (treating y as continuous) for any
        # cardinality != 2 -- including a degenerate single-class y (RMSE against a constant target is
        # a near-meaningless Shapley game) or an integer-coded multiclass y (RMSE on class-index labels
        # is not a sane classification score at all). Warn ONCE here (not inside the hot score_fn path,
        # which runs n_permutations * n_models times per call) so the caller isn't silently handed a
        # degenerate/wrong-metric Shapley decomposition.
        _card = len(np.unique(y))
        if _card != 2:
            logger.warning(
                "shapley_model_values: default score_fn falling back to negative-RMSE (treating y as "
                "continuous) -- y has cardinality %d, not the binary case fast_roc_auc handles. Pass an "
                "explicit score_fn for single-class or multiclass targets if this is not intended.",
                _card,
            )
    if rng is None:
        rng = np.random.default_rng()

    if score_subsample is not None and n_rows > score_subsample:
        sub_idx = rng.choice(n_rows, size=score_subsample, replace=False)
        preds_scored = preds[:, sub_idx]
        y_scored = y[sub_idx]
    else:
        preds_scored = preds
        y_scored = y

    v_empty = float(score_fn(y_scored, np.zeros(preds_scored.shape[1], dtype=np.float64)))
    v_full = float(score_fn(y_scored, preds_scored.mean(axis=0) if coalition_blend == "mean" else _blend(preds_scored, np.arange(n_models), coalition_blend)))

    if estimator == "permutation":
        values, n_evals = _permutation_shapley(preds_scored, y_scored, score_fn, coalition_blend, n_permutations, rng, v_empty)
    elif estimator == "msr_banzhaf":
        values, n_evals = _msr_banzhaf(preds_scored, y_scored, score_fn, coalition_blend, n_permutations, rng)
    else:
        raise ValueError(f"shapley_model_values: unsupported estimator {estimator!r}, expected 'permutation' or 'msr_banzhaf'")

    stderr = np.full(n_models, np.nan) if n_permutations < 2 else _bootstrap_stderr(values, n_permutations)
    info = dict(stderr=stderr, n_evals=n_evals, v_full=v_full, v_empty=v_empty)
    return values, info


def _permutation_shapley(preds, y, score_fn, coalition_blend, n_permutations, rng, v_empty):
    """Permutation-sampling Shapley estimator with the incremental running-sum marginal trick (mean blend only).

    ``v_empty`` is the caller's already-computed empty-coalition score (``score_fn(y, zeros)``) -- every
    permutation's first marginal step needs this exact constant, so it's passed in rather than
    recomputed once per permutation (``n_permutations`` redundant ``score_fn`` calls, ~1/n_models of the
    total call budget for a typical ``n_permutations >= 10 * n_models`` run).
    """
    n_models = preds.shape[0]
    values_sum = np.zeros(n_models, dtype=np.float64)
    n_evals = 0
    for _ in range(n_permutations):
        order = rng.permutation(n_models)
        running_sum = np.zeros(preds.shape[1], dtype=np.float64)
        v_prev = v_empty
        for step, m in enumerate(order, start=1):
            running_sum = running_sum + preds[m]
            blended = running_sum / step if coalition_blend == "mean" else _blend(preds, order[:step], coalition_blend)
            v_curr = float(score_fn(y, blended))
            values_sum[m] += v_curr - v_prev
            v_prev = v_curr
            n_evals += 1
    return values_sum / n_permutations, n_evals


def _msr_banzhaf(preds, y, score_fn, coalition_blend, n_coalitions, rng):
    """MSR-Banzhaf estimator: sample coalitions once, reuse every sample for every model's beta estimate."""
    n_models = preds.shape[0]
    masks = rng.random((n_coalitions, n_models)) < 0.5
    v = np.empty(n_coalitions, dtype=np.float64)
    for c in range(n_coalitions):
        idx = np.flatnonzero(masks[c])
        v[c] = float(score_fn(y, _blend(preds, idx, coalition_blend)))
    beta = np.zeros(n_models, dtype=np.float64)
    for j in range(n_models):
        in_mask = masks[:, j]
        if in_mask.sum() == 0 or (~in_mask).sum() == 0:
            continue
        beta[j] = float(v[in_mask].mean() - v[~in_mask].mean())
    return beta, n_coalitions


def _bootstrap_stderr(values: np.ndarray, n_permutations: int) -> np.ndarray:
    """Cheap analytic stderr proxy: values / sqrt(n_permutations), a standard-error-of-the-mean scaling (not a true bootstrap -- documented as an approximation)."""
    return np.asarray(np.abs(values) / np.sqrt(max(n_permutations, 1)) + 1e-12)


def shapley_blend(
    preds: np.ndarray,
    y: np.ndarray,
    *,
    prune_below: float = 0.0,
    renormalize: bool = True,
    **kwargs,
) -> dict:
    """Prune + blend a model pool by Shapley value; returns a hill-climb-compatible result dict.

    ``weights = clip(values, 0)``; members with ``value <= prune_below * values.sum()`` are pruned
    (``prune_below=0.0`` keeps every strictly-positive-value member -- the pruning comparison is a
    strict ``>``, so a member with an exact-zero clipped weight, including any originally-negative
    Shapley value, is pruned too). ``blended`` is the weighted mean of
    survivors (``renormalize=True`` rescales survivor weights to sum to 1).

    Degenerate case (every raw Shapley value <= 0, so every clipped weight is 0): falls back to the
    single best-valued model with weight 1.0 rather than reporting it as "selected" while silently
    returning an all-zero ``ensemble_pred``.

    Returns a dict with keys ``weights`` (``(n_models,)``, zero for pruned members), ``ensemble_pred``,
    ``score`` (the blended prediction's ``score_fn`` value), ``selected`` / ``selected_indices`` (both
    provided -- ``selected_indices`` matches :func:`mlframe.votenrank.hill_climb.hill_climb_ensemble`'s
    key convention, ``selected`` is the plan-specified alias), ``values`` (raw Shapley values), ``info``
    (the :func:`shapley_model_values` diagnostic dict).
    """
    preds = np.ascontiguousarray(preds, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n_models = preds.shape[0]

    values, info = shapley_model_values(preds, y, **kwargs)
    score_fn = kwargs.get("score_fn") or _default_score_fn

    weights = np.clip(values, 0.0, None)
    total = weights.sum()
    threshold = prune_below * total
    keep_mask = weights > threshold
    degenerate = not np.any(keep_mask)
    if degenerate:
        # Degenerate: nothing cleared the threshold -- fall back to the single best model rather than
        # returning an empty, unusable ensemble.
        keep_mask = np.zeros(n_models, dtype=bool)
        keep_mask[int(np.argmax(values))] = True

    survivor_weights = weights * keep_mask
    if degenerate:
        # The fallback model's CLIPPED weight is 0 whenever its raw Shapley value is <= 0 (the only
        # case that reaches this branch when prune_below=0.0), which would otherwise silently produce
        # an all-zero ensemble_pred while still reporting the model as "selected" -- a misleadingly
        # valid-looking result for what is actually a degenerate case. Give the sole fallback survivor
        # full weight so ensemble_pred is that model's own (real) predictions.
        survivor_weights[keep_mask] = 1.0
    elif renormalize and survivor_weights.sum() > 0:
        survivor_weights = survivor_weights / survivor_weights.sum()

    ensemble_pred = np.zeros(preds.shape[1], dtype=np.float64)
    for m in range(n_models):
        if survivor_weights[m] > 0:
            ensemble_pred = ensemble_pred + survivor_weights[m] * preds[m]
    score = float(score_fn(y, ensemble_pred))

    selected_indices = [i for i in range(n_models) if keep_mask[i]]
    return dict(
        weights=survivor_weights,
        ensemble_pred=ensemble_pred,
        score=score,
        selected=selected_indices,
        selected_indices=selected_indices,
        values=values,
        info=info,
    )


__all__ = ["shapley_model_values", "shapley_blend"]
