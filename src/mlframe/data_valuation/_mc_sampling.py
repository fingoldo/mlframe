"""Monte-Carlo row-valuation engines: TMC-Shapley and MSR-Banzhaf over a caller-supplied utility.

Both engines are model-agnostic (unlike :mod:`_knn_shapley`'s closed form) at the cost of one
``utility_fn`` evaluation (typically "fit a model on these rows, return a validation metric") per
sampled coalition/permutation prefix -- expensive for large ``n_rows``, since each evaluation is a
retrain. Intended for small/medium ``n_rows`` or a stratified subsample (propagate values to the rest
by nearest-neighbor imputation -- a caller-side concern, not implemented here). :func:`tmc_shapley`
follows Ghorbani & Zou (ICML 2019); :func:`data_banzhaf` follows Wang & Jia (AISTATS 2023, see also
``feature_selection/shap_proxied_fs/_shap_proxy_banzhaf.py`` for the same MSR estimator over a
different -- feature-column, not row -- game).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def tmc_shapley(
    utility_fn: Callable[[np.ndarray], float],
    n_rows: int,
    *,
    n_permutations: int = 200,
    truncation_tol: float = 1e-3,
    rng: np.random.Generator,
    n_jobs: int = 1,
) -> tuple[np.ndarray, dict]:
    """Truncated Monte Carlo Shapley: average marginal contribution over ``n_permutations`` random orderings.

    For each permutation, walks prefixes and accumulates ``utility_fn(prefix + [next]) -
    utility_fn(prefix)`` as ``next``'s marginal contribution for that round; stops extending a
    permutation once ``|utility_fn(prefix) - v_full| < truncation_tol`` (the players not yet reached
    contribute an implicit 0 marginal for that round -- the standard TMC bias/cost tradeoff, not a bug).
    Cost: up to ``n_permutations * n_rows`` calls to ``utility_fn`` (fewer once truncation fires
    consistently) -- each one is normally a full retrain, so this is expensive; document honestly to
    callers rather than hide the cost. ``n_jobs`` is accepted for API symmetry with
    :func:`data_banzhaf` but permutations run serially in v1 (each permutation's truncation point is
    data-dependent, so batching would not bound peak retrain count predictably); kept as a documented
    no-op rather than a silently-ignored kwarg surprise.

    Returns ``(values, info)`` where ``info`` holds ``v_empty``, ``v_full``, ``n_permutations``,
    ``mean_truncation_position`` (diagnostic: how early truncation typically fires).
    """
    v_empty = float(utility_fn(np.array([], dtype=np.int64)))
    v_full = float(utility_fn(np.arange(n_rows, dtype=np.int64)))

    values_sum = np.zeros(n_rows, dtype=np.float64)
    truncation_positions = []

    for _ in range(n_permutations):
        order = rng.permutation(n_rows)
        prefix: list[int] = []
        u_prev = v_empty
        truncated_at = n_rows
        for pos, idx in enumerate(order):
            prefix.append(int(idx))
            u_curr = float(utility_fn(np.asarray(prefix, dtype=np.int64)))
            values_sum[idx] += u_curr - u_prev
            u_prev = u_curr
            if abs(u_curr - v_full) < truncation_tol:
                truncated_at = pos + 1
                break
        truncation_positions.append(truncated_at)

    values = values_sum / n_permutations
    info = dict(
        v_empty=v_empty,
        v_full=v_full,
        n_permutations=n_permutations,
        mean_truncation_position=float(np.mean(truncation_positions)),
    )
    return values, info


def data_banzhaf(
    utility_fn: Callable[[np.ndarray], float],
    n_rows: int,
    *,
    n_coalitions: int = 2048,
    rng: np.random.Generator,
    n_jobs: int = 1,
) -> tuple[np.ndarray, dict]:
    """Maximum-Sample-Reuse Banzhaf estimate of every row's semivalue over a caller-supplied utility.

    Samples ``n_coalitions`` boolean masks over the ``n_rows`` players (each row included
    independently w.p. 0.5), evaluates ``utility_fn`` once per coalition, and estimates
    ``beta_j = mean(v | j in S) - mean(v | j not in S)`` -- every sampled coalition informs every
    row's estimate (Wang & Jia 2023's MSR trick), unlike a fresh-sample-per-row scheme. ``n_jobs`` is
    accepted for API symmetry but coalitions run serially in v1 (each is a full retrain; joblib
    parallelism across ``utility_fn`` calls is a caller-side concern since retrains may already use
    internal threading that would oversubscribe).

    Returns ``(beta, info)`` with ``info`` holding ``n_coalitions``, ``v_mean``, ``v_std``, and
    ``degenerate_rows`` (rows never/always sampled -- forced to ``beta=0``, guarded though unlikely at
    ``n_coalitions >= 64``).
    """
    masks = rng.random((n_coalitions, n_rows)) < 0.5
    v = np.empty(n_coalitions, dtype=np.float64)
    for m in range(n_coalitions):
        idx = np.flatnonzero(masks[m])
        v[m] = float(utility_fn(idx.astype(np.int64)))

    beta = np.zeros(n_rows, dtype=np.float64)
    degenerate: list[int] = []
    for j in range(n_rows):
        in_mask = masks[:, j]
        n_in = int(in_mask.sum())
        n_out = n_coalitions - n_in
        if n_in == 0 or n_out == 0:
            degenerate.append(j)
            continue
        beta[j] = float(v[in_mask].mean() - v[~in_mask].mean())

    info = dict(
        n_coalitions=n_coalitions,
        v_mean=float(v.mean()),
        v_std=float(v.std(ddof=1)) if n_coalitions > 1 else 0.0,
        degenerate_rows=degenerate,
    )
    return beta, info


def propagate_subsample_values(X_full: np.ndarray, X_subsample: np.ndarray, subsample_values: np.ndarray, *, k: int = 1) -> np.ndarray:
    """Extend a valuation computed on a stratified subsample to the full row set by nearest-neighbor imputation.

    TMC/Banzhaf cost scales with ``n_rows`` (each coalition/permutation-step is a retrain), so applying
    them to the full dataset can be prohibitive; callers instead value a stratified subsample and use
    this helper to assign every OTHER row the mean value of its ``k`` nearest subsample neighbors
    (euclidean, unscaled -- callers should standardize ``X_full``/``X_subsample`` upstream if columns
    are on different scales, mirroring :func:`_knn_shapley.knn_shapley`'s own ``standardize`` contract).
    Rows that ARE in the subsample keep their own value exactly (0-distance nearest neighbor).
    """
    from scipy.spatial.distance import cdist

    X_full = np.ascontiguousarray(X_full, dtype=np.float64)
    X_subsample = np.ascontiguousarray(X_subsample, dtype=np.float64)
    D = cdist(X_full, X_subsample)  # (n_full, n_subsample)
    nearest_k = np.argsort(D, axis=1)[:, :k]
    return np.asarray(subsample_values[nearest_k].mean(axis=1))


__all__ = ["tmc_shapley", "data_banzhaf", "propagate_subsample_values"]
