"""Hermite/orthogonal-poly pair-FE optimisation routines carved out of
``mlframe.feature_selection.filters.hermite_fe``.

Holds the Optuna/CMA outer-loop search + per-pair evaluation. Re-imported
at the parent's module bottom so historical
``from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair``
resolves transparently.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from mlframe.utils.log_throttle import log_throttle
from .fe_baselines import ksg_random_state

logger = logging.getLogger("mlframe.feature_selection.filters.hermite_fe")


from ._hermite_fe_optimise_parts import _eval_coef_pair_batch_part1, _optimise_pair_multimode_part1


def detect_pair_symmetry(
    x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, *, discrete_target: bool = True, mi_estimator: str = "plugin", plugin_n_bins: int = 20
) -> float:
    """Symmetry score in [0, 1] for (x_a, x_b) as predictors of y. Targets of form f(a,b)=f(b,a) score near 1; asymmetric like y=sign(a-2b) score lower.

    Combines (geometric mean of) two indicators:
    1. Marginal MI ratio min(MI(a,y), MI(b,y)) / max(...).
    2. Sub/Add MI ratio MI(|a-b|, y) / MI(a+b, y).

    Score >= 0.95: caller can constrain c_a = c_b to halve search dim. Score <= 0.7: clearly asymmetric - per-feature basis routing matters more.
    """
    from .fe_baselines import _mi_1d
    x_a = np.asarray(x_a, dtype=np.float64)
    x_b = np.asarray(x_b, dtype=np.float64)
    # Marginal MI test
    mi_a = _mi_1d(x_a, y, discrete_target=discrete_target, mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins)
    mi_b = _mi_1d(x_b, y, discrete_target=discrete_target, mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins)
    big_m = max(mi_a, mi_b, 1e-12)
    small_m = min(mi_a, mi_b)
    marginal_score = small_m / big_m
    # Sub vs add test (high = symmetric)
    mi_add = _mi_1d(x_a + x_b, y, discrete_target=discrete_target, mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins)
    mi_sub_abs = _mi_1d(np.abs(x_a - x_b), y, discrete_target=discrete_target, mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins)
    big_d = max(mi_add, mi_sub_abs, 1e-12)
    small_d = min(mi_add, mi_sub_abs)
    diff_score = small_d / big_d if big_d > 1e-9 else 0.0
    # Geometric mean of both signals.
    return float(np.sqrt(marginal_score * diff_score))

def _eval_coef_pair(coef_a, coef_b, *, z_a, z_b, eval_func, bf_callables,
                     bf_names, y, y_njit, mi_estimator, plugin_n_bins,
                     n_neighbors, discrete_target, l2_penalty,
                     l2_penalty_saturation=None,
                     direction_only=False, eval_func_b=None,
                     B_a=None, B_b=None, random_state=42):
    """Shared inner objective: evaluate one (c_a, c_b) pair across all binary funcs; return best (regularised score, raw MI, bf idx).

    ``eval_func_b`` defaults to ``eval_func`` (single-eval). Factory bases like RBF need per-feature preprocess fns, so the caller passes a separate
    ``eval_func_b`` closure over ``preprocess_b`` to evaluate ``h_b``. Without this the b-side silently re-used preprocess_a, biasing RBF fits.

    2026-05-18 PERFORMANCE: when ``B_a`` / ``B_b`` (precomputed basis matrices, PRE-TRUNCATED by the caller
    to exactly ``(n, len(coef_a))`` / ``(n, len(coef_b))``) are supplied, evaluation uses BLAS GEMV
    ``h = B @ c`` instead of recomputing Horner per call. Builds via ``build_basis_matrix`` plus the
    per-degree truncation are done ONCE per pair/degree before CMA-ES; per-trial cost drops from ~340us
    (njit Horner at n=1500) to ~30-80us (BLAS GEMV) - 4-10x speedup. Factory bases (RBF / Sigmoid) need
    per-feature preprocess closures and DON'T support basis-matrix caching; the caller leaves B_a / B_b at
    None for those.
    """
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    from mlframe.feature_selection.filters.hermite_fe.shared import L2_PENALTY_SATURATION_DEFAULT as _L2_PENALTY_SATURATION_DEFAULT, l2_normalize_pair as _l2_normalize_pair, l2_penalty_value as _l2_penalty_value, plugin_mi_regression_batch_njit as _plugin_mi_regression_batch_njit
    if l2_penalty_saturation is None:
        l2_penalty_saturation = _L2_PENALTY_SATURATION_DEFAULT
    if direction_only:
        coef_a, coef_b = _l2_normalize_pair(coef_a, coef_b, target_norm=1.0)
    # BASIS-MATRIX FASTPATH (kept but disabled by default):
    # ``B_a`` / ``B_b`` are optional precomputed basis matrices that allow
    # ``h = B @ c`` (BLAS GEMV) instead of Horner. Numerical
    # identity to Horner verified to 1e-16 across all four orthogonal
    # bases. HOWEVER, measured at n=1M / subsample=200k production budget
    # (cProfile): GEMV on the 1500-sample multi-fidelity
    # inner search shows ZERO measurable speedup vs the existing
    # @njit(parallel=True) Horner kernel - the JIT'd recurrence with
    # prange is already cache-friendly enough that BLAS dgemv has no
    # margin. The build cost (one ``build_basis_matrix`` call per pair
    # x per restart, ~5ms each) roughly cancels the GEMV win on the
    # 1500-sample inner loop.
    #
    # Optimization left in place because it WOULD help when:
    # - ``multi_fidelity=False`` (CMA-ES on full data per call)
    # - n_full < 4000 (multi-fidelity disabled by size threshold)
    # - future popsize-batched evaluation (where many coefs share one
    #   z, GEMM ``H = B @ C.T`` would amortise the build cost)
    #
    # CALLERS PASSING B_a / B_b MUST pre-truncate them to EXACTLY
    # (n, coef_a.shape[0]) / (n, coef_b.shape[0]) - this function no longer
    # re-slices per trial (that re-slice+copy ran on every CMA-ES/random-batch
    # trial, tens of thousands of times per pair; the only caller,
    # ``optimise_hermite_pair``, now truncates once per degree). The
    # refinement step in ``optimise_hermite_pair`` explicitly sets B_a
    # = B_b = None when switching to full-n z - omitting that produced
    # shape (1500,) vs (n,) errors / silent hermite=0 regression measured
    # in development.
    if B_a is not None and B_b is not None:
        # B_a / B_b arrive PRE-TRUNCATED to (ca_size, cb_size) by the only caller that ever sets them
        # (optimise_hermite_pair slices once per degree, before the trial loop) - do not re-slice per trial.
        _ca = np.ascontiguousarray(coef_a, dtype=np.float64)
        _cb = np.ascontiguousarray(coef_b, dtype=np.float64)
        h_a = B_a @ _ca
        h_b = B_b @ _cb
    else:
        h_a = eval_func(z_a, coef_a)
        h_b = (eval_func_b if eval_func_b is not None else eval_func)(z_b, coef_b)
    if not (np.all(np.isfinite(h_a)) and np.all(np.isfinite(h_b))):
        return -np.inf, 0.0, -1
    # Same njit bf twins + fused row-major fill as the batched evaluator (P=1 degenerate case): one prange
    # call computes every bf column, finiteness-scanned in-kernel, rows feeding the copy-free MI rows twin.
    # Optuna's per-trial objective goes through HERE, so the cma_batch-round optimizations carry over to it.
    _bf_name_to_id: dict[str, int]
    try:
        from ._numba_polynom_optimizer import _BF_NAME_TO_ID as _bf_name_to_id, _fill_bf_batch_njit  # noqa: N811
    except ImportError as e:
        logger.warning("_numba_polynom_optimizer import failed: %s", e)
        _bf_name_to_id = {}
        _fill_bf_batch_njit = None
    _bf_ids1 = [_bf_name_to_id.get(nm, -1) for nm in bf_names] if _bf_name_to_id else [-1]
    cols = []
    valid_idx = []
    if _fill_bf_batch_njit is not None and all(i >= 0 for i in _bf_ids1):
        KBF = len(bf_callables)
        rows = np.empty((KBF, h_a.shape[0]), dtype=np.float64)
        finite = np.zeros(KBF, dtype=np.bool_)
        _fill_bf_batch_njit(np.ascontiguousarray(h_a, dtype=np.float64)[None, :],
                            np.ascontiguousarray(h_b, dtype=np.float64)[None, :],
                            np.ones(1, dtype=np.bool_), np.asarray(_bf_ids1, dtype=np.int64), rows, finite)
        nc1 = 0
        for k in range(KBF):
            if finite[k]:
                if k != nc1:
                    rows[nc1] = rows[k]
                valid_idx.append(k)
                nc1 += 1
        if nc1 == 0:
            return -np.inf, 0.0, -1
        X_rows = rows[:nc1]
    else:
        for k, bf in enumerate(bf_callables):
            try:
                combined = bf(h_a, h_b)
            except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed: %s", e)
                continue
            if np.all(np.isfinite(combined)):
                cols.append(combined)
                valid_idx.append(k)
        if not cols:
            return -np.inf, 0.0, -1
        X_rows = np.ascontiguousarray(np.stack(cols, axis=0), dtype=np.float64)
    X_batch = X_rows.T  # (n, K_valid) view for the ksg legs below; plugin uses the rows form directly
    if mi_estimator == "plugin":
        from mlframe.feature_selection.filters.hermite_fe.shared import plugin_mi_classif_batch_rows_njit as _plugin_mi_classif_batch_rows_njit
        if discrete_target:
            mi_arr = _plugin_mi_classif_batch_rows_njit(X_rows, y_njit, plugin_n_bins)
        else:
            mi_arr = _plugin_mi_regression_batch_njit(np.ascontiguousarray(X_batch), y_njit, plugin_n_bins)
    else:  # ksg
        if discrete_target:
            mi_arr = mutual_info_classif(X_batch, y, n_neighbors=n_neighbors, random_state=ksg_random_state(random_state), discrete_features=False)
        else:
            mi_arr = mutual_info_regression(X_batch, y, n_neighbors=n_neighbors,
                                             random_state=ksg_random_state(random_state), discrete_features=False)
    penalty = 0.0 if direction_only else _l2_penalty_value(
        coef_a, coef_b, l2_penalty, l2_penalty_saturation,
    )
    best_score = -np.inf
    best_raw = 0.0
    best_idx = -1
    for j, k in enumerate(valid_idx):
        raw = float(mi_arr[j])
        s = raw - penalty
        if s > best_score:
            best_score = s
            best_raw = raw
            best_idx = k
    return best_score, best_raw, best_idx

def _candidate_h_arrays(
    coefs_a: np.ndarray,
    coefs_b: np.ndarray,
    *,
    z_a: np.ndarray,
    z_b: np.ndarray,
    eval_func: Callable[..., Any],
    eval_func_b: Optional[Callable[..., Any]],
    B_a: Optional[np.ndarray],
    B_b: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-candidate a-side and b-side polynomial evaluations; returns ``(h_a_arr, h_b_arr, cand_valid)``.

    Skip candidates that produce non-finite h_a/h_b - they get -inf score downstream.
    """
    P = int(coefs_a.shape[0])
    # bench-attempt-rejected (2026-07-15): one GEMM (B_a @ coefs_a.T).T instead of P per-candidate GEMVs
    # measured 2.7x faster ISOLATED (n=20000, P=20, cs=7: 4.7ms loop vs 1.7ms batched; ~1e-15 abs FP-reorder
    # divergence, within tolerance) but NET FLAT end-to-end (1.49s -> 1.52s over a full cma_batch search,
    # within run-to-run noise) - the coefs.T ascontiguousarray + single large-P GEMM's BLAS thread dispatch
    # overhead cancels the win at this P/n scale in-process alongside numba's own thread pool. Kept as the
    # simpler per-candidate loop; the batched form would need re-measuring at a much larger P (e.g. a wider
    # cupy-side generation batch) before it nets positive.
    n_rows = z_a.shape[0]
    h_a_arr = np.empty((P, n_rows), dtype=np.float64)
    h_b_arr = np.empty((P, n_rows), dtype=np.float64)
    cand_valid = np.zeros(P, dtype=bool)
    for p in range(P):
        ca = coefs_a[p]
        cb = coefs_b[p]
        if B_a is not None and B_b is not None:
            # B_a / B_b are pre-truncated to (ca_size, cb_size) by the caller (once per degree) - see the
            # matching note in _eval_coef_pair; no per-candidate re-slice.
            _ca = np.ascontiguousarray(ca, dtype=np.float64)
            _cb = np.ascontiguousarray(cb, dtype=np.float64)
            h_a = B_a @ _ca
            h_b = B_b @ _cb
        else:
            h_a = eval_func(z_a, ca)
            h_b = (eval_func_b if eval_func_b is not None else eval_func)(z_b, cb)
        if np.all(np.isfinite(h_a)) and np.all(np.isfinite(h_b)):
            h_a_arr[p] = h_a
            h_b_arr[p] = h_b
            cand_valid[p] = True
    return h_a_arr, h_b_arr, cand_valid


def _eval_coef_pair_batch(
    coefs_a: np.ndarray,
    coefs_b: np.ndarray,
    *,
    z_a: np.ndarray,
    z_b: np.ndarray,
    eval_func: Callable[..., Any],
    bf_callables: Sequence[Callable[..., Any]],
    bf_names: Sequence[str],
    y: np.ndarray,
    y_njit: np.ndarray,
    mi_estimator: str,
    plugin_n_bins: int,
    n_neighbors: int,
    discrete_target: bool,
    l2_penalty: float,
    l2_penalty_saturation: Optional[float] = None,
    direction_only: bool = False,
    eval_func_b: Optional[Callable[..., Any]] = None,
    B_a: Optional[np.ndarray] = None,
    B_b: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched eval over ``P`` coefficient candidates simultaneously.

    Args:
        coefs_a: ndarray (P, ca_size) - candidate coefficients for the a-side.
        coefs_b: ndarray (P, cb_size) - candidate coefficients for the b-side.
        z_a: the a-side transformed operand array shared across all candidates.
        z_b: the b-side transformed operand array shared across all candidates.
        eval_func: closure evaluating one (coef_a, coef_b) candidate's polynomial basis.
        bf_callables: the binary-function candidates scored against each polynomial pair.
        bf_names: names paired 1:1 with ``bf_callables``, for the returned best-bf index.
        y: the target array (binned when ``discrete_target``, else continuous).
        y_njit: numba-friendly cast of ``y`` for the batched MI kernels.
        mi_estimator: MI backend selector forwarded to the batched MI kernels.
        plugin_n_bins: bin count for the plug-in MI estimator.
        n_neighbors: neighbor count for a KSG-style MI estimator, when selected.
        discrete_target: whether ``y`` is already discrete class codes vs continuous.
        l2_penalty: L2 regularisation weight applied to each candidate's coefficients.
        l2_penalty_saturation: saturation point for the L2 penalty; ``None`` uses the module default.
        direction_only: whether to L2-normalise each candidate pair before scoring (direction, not magnitude).
        eval_func_b: optional distinct eval closure for the b-side when it differs from ``eval_func``.
        B_a: optional precomputed basis matrix for the a-side, reused across candidates.
        B_b: optional precomputed basis matrix for the b-side, reused across candidates.
        random_state: seed forwarded to the KSG mutual-information estimators.

    Returns:
      (best_scores, best_raws, best_idxs) each ndarray (P,) - per-candidate best regularised score, raw MI, and bf index.

    The single-call version dispatches one MI batch call across ALL
    (candidate, bf) column combinations: typical (P=20, K_bf=5) gives a
    100-column batch instead of 20 separate 5-column batches. The plugin
    MI kernel is already numba-batched over columns via prange, so this
    feeds 100 columns into ONE prange loop - saturates cores fully and
    removes 19 of the 20 Python-side GIL acquire/release cycles between
    successive MI calls. Combined with the per-candidate Python loop for
    polyeval (cheap numba calls), this is the bulk of the speedup vs
    Optuna's per-trial sequential evaluation.
    """
    _ret_eval_coef_pair_batch_part1 = _eval_coef_pair_batch_part1(
        l2_penalty_saturation, coefs_a, direction_only, coefs_b, z_a, z_b, eval_func, eval_func_b, B_a, B_b, bf_callables, bf_names, mi_estimator,
        discrete_target, y_njit, plugin_n_bins, y, n_neighbors, random_state, l2_penalty,
    )
    if _ret_eval_coef_pair_batch_part1[0]:
        _early_eval_coef_pair_batch_part1: tuple[np.ndarray, np.ndarray, np.ndarray] = _ret_eval_coef_pair_batch_part1[1]
        return _early_eval_coef_pair_batch_part1
    p, col_meta, k, best_scores, best_raws, best_idxs, mi_arr, penalties = _ret_eval_coef_pair_batch_part1[1:]
    for j, (p, k) in enumerate(col_meta):
        raw = float(mi_arr[j])
        s = raw - penalties[p]
        if s > best_scores[p]:
            best_scores[p] = s
            best_raws[p] = raw
            best_idxs[p] = k
    return best_scores, best_raws, best_idxs


def _absorb_batch_scores(scores, raws, idxs, a_batch, b_batch, n, direction_only, track_history, history, best_score, best_raw, best_idx, best_coefs):
    """Fold the first ``n`` scored candidates of one batch into the running best and, when tracked, the history.

    Returns the updated ``(best_score, best_raw, best_idx, best_coefs)``; a non-finite score never becomes the best.
    """
    from mlframe.feature_selection.filters.hermite_fe.shared import l2_normalize_pair as _l2_normalize_pair

    for j in range(n):
        sc = scores[j]
        if not np.isfinite(sc):
            continue
        if track_history and idxs[j] >= 0:
            history.append((float(sc), float(raws[j]), int(idxs[j]), a_batch[j].copy(), b_batch[j].copy()))
        if sc > best_score:
            best_score = sc
            best_raw = raws[j]
            best_idx = idxs[j]
            if direction_only:
                nc_a, nc_b = _l2_normalize_pair(a_batch[j], b_batch[j])
                best_coefs = (nc_a.copy(), nc_b.copy())
            else:
                best_coefs = (a_batch[j].copy(), b_batch[j].copy())
    return best_score, best_raw, best_idx, best_coefs


def _run_cma_search_batch(*, ca_size, cb_size, coef_range, n_trials, seed,
                           direction_only, warm_start_seeds, eval_kwargs,
                           popsize=None, track_history=False,
                           early_stop_no_improve_gens: int | None = None):
    """CMA-ES with batch evaluation: collect all popsize candidates per
    generation, evaluate them in ONE call to ``_eval_coef_pair_batch``,
    feed scores back to ``es.tell``. Removes the per-solution Python
    GIL dance that ``_run_cma_search`` paid.

    Same signature + return contract as ``_run_cma_search`` for drop-in
    replacement via ``optimizer="cma_batch"``."""
    import cma
    dim = ca_size + cb_size
    if popsize is None:
        popsize = max(8, min(20, n_trials // 8))
    sigma0 = (coef_range[1] - coef_range[0]) / 4.0

    best_score = -np.inf
    best_raw = 0.0
    best_idx = -1
    best_coefs = None
    n_evals = 0
    history: list | None = [] if track_history else None

    # Warm-start batch: evaluate canonical seeds in one batch call.
    if warm_start_seeds:
        seeds_arr = np.stack([np.asarray(s, dtype=np.float64) for s in warm_start_seeds])
        ws_a = seeds_arr[:, :ca_size]
        ws_b = seeds_arr[:, ca_size:]
        ws_scores, ws_raws, ws_idxs = _eval_coef_pair_batch(
            ws_a, ws_b, direction_only=direction_only, **eval_kwargs,
        )
        n_evals += len(ws_scores)
        best_score, best_raw, best_idx, best_coefs = _absorb_batch_scores(
            ws_scores, ws_raws, ws_idxs, ws_a, ws_b, len(ws_scores), direction_only, track_history, history,
            best_score, best_raw, best_idx, best_coefs,
        )
        x0 = np.concatenate([best_coefs[0], best_coefs[1]]) if best_coefs is not None else np.zeros(dim, dtype=np.float64)
        sigma0 = sigma0 * 0.5
    else:
        x0 = np.zeros(dim, dtype=np.float64)

    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {
            "popsize": popsize,
            "bounds": [[coef_range[0]] * dim, [coef_range[1]] * dim],
            "verbose": -9, "verb_disp": 0, "verb_log": 0,
            "seed": seed if seed > 0 else 1,
            "tolfun": 1e-6, "tolx": 1e-6,
        },
    )

    _plateau_gens = 0
    _last_gen_best_score = best_score
    while not es.stop() and n_evals < n_trials:
        try:
            solutions = es.ask()
        except Exception as exc:
            # Was a bare except with zero logging, silently
            # truncating the CMA generation loop early on any cma-library fault - a real cma regression
            # (e.g. after a package upgrade) would silently degrade hermite pair-FE recall with no trace.
            log_throttle(
                logger, "hermite_cma_search_batch_ask_failed", logging.WARNING,
                "_run_cma_search_batch: es.ask() raised (%s: %s); stopping CMA early with the best solution found so far.", type(exc).__name__, exc,
            )
            break
        solutions_arr = np.asarray(solutions, dtype=np.float64)
        # Truncate batch if we'd exceed the budget mid-generation.
        _avail = n_trials - n_evals
        if _avail < len(solutions):
            sol_batch = solutions_arr[:_avail]
        else:
            sol_batch = solutions_arr
        ca_batch = sol_batch[:, :ca_size]
        cb_batch = sol_batch[:, ca_size:]
        scores_batch, raws_batch, idxs_batch = _eval_coef_pair_batch(
            ca_batch, cb_batch, direction_only=direction_only, **eval_kwargs,
        )
        n_evals += len(sol_batch)
        best_score, best_raw, best_idx, best_coefs = _absorb_batch_scores(
            scores_batch, raws_batch, idxs_batch, ca_batch, cb_batch, len(sol_batch), direction_only, track_history, history,
            best_score, best_raw, best_idx, best_coefs,
        )
        # Pad scores for any truncated solutions so es.tell still gets
        # popsize entries; truncated tail gets a large penalty so CMA
        # doesn't drift toward them.
        cma_scores = []
        for j in range(len(solutions)):
            if j < len(sol_batch):
                sc = scores_batch[j]
                cma_scores.append(-sc if np.isfinite(sc) else 1e6)
            else:
                cma_scores.append(1e6)
        try:
            es.tell(solutions, cma_scores)
        except Exception as exc:
            # See the matching es.ask fix above.
            log_throttle(
                logger, "hermite_cma_search_batch_tell_failed", logging.WARNING,
                "_run_cma_search_batch: es.tell() raised (%s: %s); stopping CMA early with the best solution found so far.", type(exc).__name__, exc,
            )
            break
        if early_stop_no_improve_gens is not None and early_stop_no_improve_gens > 0:
            if best_score > _last_gen_best_score:
                _plateau_gens = 0
                _last_gen_best_score = best_score
            else:
                _plateau_gens += 1
                if _plateau_gens >= early_stop_no_improve_gens:
                    break

    if best_coefs is None:
        return None
    if track_history:
        return (best_coefs[0], best_coefs[1], best_idx, best_raw, n_evals, history)
    return (best_coefs[0], best_coefs[1], best_idx, best_raw, n_evals)


def _evaluate_warm_starts(warm_start_seeds, ca_size, direction_only, eval_kwargs, history, best_score, best_raw, best_idx, best_coefs, n_evals):
    """Evaluate the canonical warm-start seeds in one batch and fold them into the running best.

    Returns ``(n_evals, best_score, best_raw, best_idx, best_coefs)``. ``history`` is appended to in place when it is not None.
    """
    from mlframe.feature_selection.filters.hermite_fe.shared import l2_normalize_pair as _l2_normalize_pair

    seeds_arr = np.stack([np.asarray(s, dtype=np.float64) for s in warm_start_seeds])
    ws_a = seeds_arr[:, :ca_size]
    ws_b = seeds_arr[:, ca_size:]
    ws_scores, ws_raws, ws_idxs = _eval_coef_pair_batch(
        ws_a, ws_b, direction_only=direction_only, **eval_kwargs,
    )
    n_evals += len(ws_scores)
    for j, sc in enumerate(ws_scores):
        if not np.isfinite(sc):
            continue
        if history is not None and ws_idxs[j] >= 0:
            history.append((float(sc), float(ws_raws[j]), int(ws_idxs[j]), ws_a[j].copy(), ws_b[j].copy()))
        if sc > best_score:
            best_score = sc
            best_raw = ws_raws[j]
            best_idx = ws_idxs[j]
            if direction_only:
                nc_a, nc_b = _l2_normalize_pair(ws_a[j], ws_b[j])
                best_coefs = (nc_a.copy(), nc_b.copy())
            else:
                best_coefs = (ws_a[j].copy(), ws_b[j].copy())
    return n_evals, best_score, best_raw, best_idx, best_coefs


def _run_random_batch_search(*, ca_size, cb_size, coef_range, n_trials, seed,
                               direction_only, warm_start_seeds, eval_kwargs,
                               batch_size: int = 20,
                               elitism_k: int = 4,
                               perturb_sigma_frac: float = 0.1,
                               track_history=False):
    """Custom batch random search + elitism + perturbation. No Optuna,
    no CMA dependency.

    Each iteration: sample ``batch_size`` candidates uniformly within
    ``coef_range``. After iter 0, replace the first ``elitism_k`` slots
    with Gaussian perturbations of the current best (sigma =
    ``perturb_sigma_frac`` * coef_range_width). All candidates evaluated
    in ONE ``_eval_coef_pair_batch`` call - pure Python decision
    overhead per iter is ~5 lines, no GIL-locked sampler loop.

    Same return contract as ``_run_cma_search`` so the dispatcher can
    drop it in via ``optimizer="random_batch"``.
    """
    from mlframe.feature_selection.filters.hermite_fe.shared import l2_normalize_pair as _l2_normalize_pair
    # Was `seed if seed > 0 else 1`, silently substituting 1
    # for seed<=0 (including the valid seed=0) - sibling RNGs in this same call chain
    # (_hermite_fe_optimise_pair.py's multi-fidelity subsample / noise-floor null) correctly keep seed=0 as
    # 0. Reachable via the public optimizer="random_batch" kwarg AND via the DEFAULT cupy_kernel path's
    # no-cupy fallback, so a direct seed=0 caller got a silently different, non-requested RNG stream.
    rng = np.random.default_rng(seed)
    best_score = -np.inf
    best_raw = 0.0
    best_idx = -1
    best_coefs = None
    n_evals = 0
    history: list | None = [] if track_history else None
    sigma = perturb_sigma_frac * (coef_range[1] - coef_range[0])
    n_iters = max(1, int(np.ceil(n_trials / max(1, batch_size))))

    if warm_start_seeds:
        n_evals, best_score, best_raw, best_idx, best_coefs = _evaluate_warm_starts(
            warm_start_seeds, ca_size, direction_only, eval_kwargs, history, best_score, best_raw, best_idx, best_coefs, n_evals,
        )

    for _it in range(n_iters):
        if n_evals >= n_trials:
            break
        eff_batch = min(batch_size, n_trials - n_evals)
        coefs_a_batch = rng.uniform(coef_range[0], coef_range[1], size=(eff_batch, ca_size))
        coefs_b_batch = rng.uniform(coef_range[0], coef_range[1], size=(eff_batch, cb_size))
        # Elitism: replace first K slots with perturbed best.
        if best_coefs is not None:
            k = min(elitism_k, eff_batch)
            coefs_a_batch[:k] = best_coefs[0] + rng.normal(0.0, sigma, (k, ca_size))
            coefs_b_batch[:k] = best_coefs[1] + rng.normal(0.0, sigma, (k, cb_size))
            # Clip to bounds.
            np.clip(coefs_a_batch[:k], coef_range[0], coef_range[1], out=coefs_a_batch[:k])
            np.clip(coefs_b_batch[:k], coef_range[0], coef_range[1], out=coefs_b_batch[:k])
        scores_batch, raws_batch, idxs_batch = _eval_coef_pair_batch(
            coefs_a_batch, coefs_b_batch,
            direction_only=direction_only, **eval_kwargs,
        )
        n_evals += eff_batch
        # Update best + history.
        best_p = int(np.argmax(scores_batch))
        if np.isfinite(scores_batch[best_p]) and scores_batch[best_p] > best_score:
            best_score = float(scores_batch[best_p])
            best_raw = float(raws_batch[best_p])
            best_idx = int(idxs_batch[best_p])
            if direction_only:
                nc_a, nc_b = _l2_normalize_pair(coefs_a_batch[best_p], coefs_b_batch[best_p])
                best_coefs = (nc_a.copy(), nc_b.copy())
            else:
                best_coefs = (coefs_a_batch[best_p].copy(), coefs_b_batch[best_p].copy())
        if track_history:
            for j in range(eff_batch):
                if np.isfinite(scores_batch[j]) and idxs_batch[j] >= 0:
                    history.append((float(scores_batch[j]), float(raws_batch[j]), int(idxs_batch[j]), coefs_a_batch[j].copy(), coefs_b_batch[j].copy()))  # type: ignore[union-attr]  # guarded by track_history, which is True exactly when history was allocated non-None

    if best_coefs is None:
        return None
    if track_history:
        return (best_coefs[0], best_coefs[1], best_idx, best_raw, n_evals, history)
    return (best_coefs[0], best_coefs[1], best_idx, best_raw, n_evals)


def _run_cma_search(*, ca_size, cb_size, coef_range, n_trials, seed,
                     direction_only, warm_start_seeds, eval_kwargs,
                     popsize=None, eval_pair_fn=None,
                     track_history=False,
                     early_stop_no_improve_gens: int | None = None):
    """CMA-ES inner loop. Returns (best_coef_a, best_coef_b, best_bf_idx, best_raw_mi, n_evals). When track_history=True, also returns the full evaluation list.

    CMA minimizes; we negate the MI score. Default popsize=max(8, min(20, n_trials // 8)) - smaller than CMA's default to allow more generations on tight budgets.

    ``early_stop_no_improve_gens`` (2026-05-20 NEW-D): break out of the
    CMA loop when ``best_score`` has not improved for this many
    consecutive GENERATIONS (not trials). Set to None to disable.
    Useful when the warm-start seeds + early CMA generations have
    already found the optimum and the remaining budget is wasted
    exploring around it. Default None (no plateau early-stop).
    """
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    from mlframe.feature_selection.filters.hermite_fe.shared import l2_normalize_pair as _l2_normalize_pair
    import cma
    dim = ca_size + cb_size
    if popsize is None:
        popsize = max(8, min(20, n_trials // 8))
    sigma0 = (coef_range[1] - coef_range[0]) / 4.0  # ~1.0 for [-2, 2]

    # Pre-evaluate canonical warm-start seeds: cheap (single MI eval each), frequently coincide with the global
    # optimum (e.g. He_1(x_a) * He_1(x_b) = x_a * x_b is exactly XOR). Best seed becomes CMA's x0 so CMA never
    # does worse than the warm-start.
    best_score = -np.inf
    best_raw = 0.0
    best_idx = -1
    best_coefs = None
    n_evals = 0
    history: list | None = [] if track_history else None
    if warm_start_seeds:
        for ws in warm_start_seeds:
            ws = np.asarray(ws, dtype=np.float64)
            coef_a = ws[:ca_size]
            coef_b = ws[ca_size:]
            score, raw_mi, bf_idx = (eval_pair_fn or _eval_coef_pair)(
                coef_a, coef_b, direction_only=direction_only, **eval_kwargs,
            )
            n_evals += 1
            if track_history and bf_idx >= 0 and np.isfinite(score):
                history.append((float(score), float(raw_mi), int(bf_idx), coef_a.copy(), coef_b.copy()))  # type: ignore[union-attr]  # guarded by track_history, which is True exactly when history was allocated non-None
            if score > best_score:
                best_score = score
                best_raw = raw_mi
                best_idx = bf_idx
                if direction_only:
                    nc_a, nc_b = _l2_normalize_pair(coef_a, coef_b)
                    best_coefs = (nc_a.copy(), nc_b.copy())
                else:
                    best_coefs = (coef_a.copy(), coef_b.copy())
        # Use the best canonical seed as CMA's starting point.
        x0 = np.concatenate([best_coefs[0], best_coefs[1]]) if best_coefs is not None else np.zeros(dim, dtype=np.float64)
        # Tighter sigma when we already have a good seed - exploit
        # rather than explore.
        sigma0 = sigma0 * 0.5
    else:
        x0 = np.zeros(dim, dtype=np.float64)

    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {
            "popsize": popsize,
            "bounds": [[coef_range[0]] * dim, [coef_range[1]] * dim],
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
            "seed": seed if seed > 0 else 1,
            "tolfun": 1e-6,
            "tolx": 1e-6,
        },
    )
    # Inject remaining canonical seeds into the first generation - CMA
    # 4.x lets us replace ask()'s random samples directly.
    inject_arrays = [np.asarray(s, dtype=np.float64) for s in (warm_start_seeds or [])]

    first_gen = True
    # NEW-D: plateau early-stop state. Tracks how many
    # consecutive CMA generations passed without improving best_score.
    _plateau_gens = 0
    _last_gen_best_score = best_score
    while not es.stop() and n_evals < n_trials:
        try:
            if first_gen and inject_arrays:
                solutions = es.ask()
                # Replace the last len(inject) random samples with seeds.
                k = min(len(inject_arrays), len(solutions))
                for j in range(k):
                    solutions[-(j + 1)] = inject_arrays[j]
                first_gen = False
            else:
                solutions = es.ask()
        except Exception as exc:
            # See _run_cma_search_batch's matching fix above.
            log_throttle(
                logger, "hermite_cma_search_ask_failed", logging.WARNING,
                "_run_cma_search: es.ask() raised (%s: %s); stopping CMA early with the best solution found so far.", type(exc).__name__, exc,
            )
            break
        scores = []
        for sol in solutions:
            if n_evals >= n_trials:
                break
            coef_a = sol[:ca_size]
            coef_b = sol[ca_size:]
            score, raw_mi, bf_idx = (eval_pair_fn or _eval_coef_pair)(
                coef_a, coef_b, direction_only=direction_only, **eval_kwargs,
            )
            n_evals += 1
            if track_history and bf_idx >= 0 and np.isfinite(score):
                history.append((float(score), float(raw_mi), int(bf_idx), coef_a.copy(), coef_b.copy()))  # type: ignore[union-attr]  # guarded by track_history, which is True exactly when history was allocated non-None
            if score > best_score:
                best_score = score
                best_raw = raw_mi
                best_idx = bf_idx
                # Store post-projection coefs if direction_only mode.
                if direction_only:
                    nc_a, nc_b = _l2_normalize_pair(coef_a, coef_b)
                    best_coefs = (nc_a.copy(), nc_b.copy())
                else:
                    best_coefs = (coef_a.copy(), coef_b.copy())
            scores.append(-score if np.isfinite(score) else 1e6)
        if len(scores) < len(solutions):
            scores.extend([1e6] * (len(solutions) - len(scores)))
        es.tell(solutions, scores)
        # Plateau early-stop check (after es.tell so the generation is
        # complete). Compare end-of-generation best_score to start-of-
        # generation; if no improvement, increment plateau counter.
        if early_stop_no_improve_gens is not None and early_stop_no_improve_gens > 0:
            if best_score > _last_gen_best_score:
                _plateau_gens = 0
                _last_gen_best_score = best_score
            else:
                _plateau_gens += 1
                if _plateau_gens >= int(early_stop_no_improve_gens):
                    break
    if best_coefs is None:
        return None
    if track_history:
        return (best_coefs[0], best_coefs[1], best_idx, best_raw, n_evals, history)
    return (best_coefs[0], best_coefs[1], best_idx, best_raw, n_evals)


def _baseline_mi_pair(x_a, x_b, y, *, discrete_target: bool, n_neighbors: int = 3, mi_estimator: str = "plugin", plugin_n_bins: int = 20, random_state: int = 42) -> float:
    """Identity baseline: MI of (x_a, x_b) vs target. Plug-in is 1-D-x by design so we use max(MI(x_a, y), MI(x_b, y)) (lower bound on joint MI); KSG path uses sklearn's multi-D estimator."""
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    from mlframe.feature_selection.filters.hermite_fe.shared import plugin_mi_classif_njit as _plugin_mi_classif_njit, plugin_mi_regression_njit as _plugin_mi_regression_njit
    if mi_estimator == "plugin":
        # Plug-in is 1-D-x by design; use max(MI(x_a, y), MI(x_b, y)) as a lower bound on the true joint MI.
        # Conservative gate (under-estimates baseline so engineered features clear it more easily); for the
        # final uplift number the bias is consistent (same estimator on both sides of the ratio).
        x_a_arr = np.asarray(x_a, dtype=np.float64)
        x_b_arr = np.asarray(x_b, dtype=np.float64)
        if discrete_target:
            y_arr = np.asarray(y, dtype=np.int64)
            mi_a = _plugin_mi_classif_njit(x_a_arr, y_arr, plugin_n_bins)
            mi_b = _plugin_mi_classif_njit(x_b_arr, y_arr, plugin_n_bins)
        else:
            y_arr = np.asarray(y, dtype=np.float64)
            mi_a = _plugin_mi_regression_njit(x_a_arr, y_arr, plugin_n_bins)
            mi_b = _plugin_mi_regression_njit(x_b_arr, y_arr, plugin_n_bins)
        return float(max(mi_a, mi_b))
    Xn = np.column_stack([x_a, x_b])
    if discrete_target:
        return float(mutual_info_classif(Xn, y, n_neighbors=n_neighbors, random_state=ksg_random_state(random_state), discrete_features=False).max())
    return float(mutual_info_regression(Xn, y, n_neighbors=n_neighbors, random_state=ksg_random_state(random_state), discrete_features=False).max())


def optimise_pair_multimode(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    *,
    top_m: int = 3,
    min_l2_distance: float = 0.3,
    discrete_target: bool = True,
    bin_funcs: dict | None = None,
    max_degree: int = 4,
    min_degree: int = 2,
    n_trials: int = 200,
    coef_range: tuple = (-2.0, 2.0),
    l2_penalty: float = 0.05,
    n_neighbors: int | None = None,
    seed: int = 42,
    sweep_degrees: bool = True,
    baseline_uplift_threshold: float = 1.01,
    basis: str = "chebyshev",
    mi_estimator: str = "plugin",
    plugin_n_bins: int = 20,
    warm_start: bool = True,
    direction_only: bool = False,
) -> list:
    """Multi-mode pair-FE: return up to top_m distinct HermiteResult objects, greedily filtered to maintain
    pair-wise L2 distance >= min_l2_distance after L2-normalisation (direction-only comparison).

    A single 2D f(x_a, x_b) can have multiple rank-1 separable approximations of similar MI; emitting all of
    them lets the downstream model exploit the multi-modal structure. Verified on Friedman1-style targets
    where MI is split across 2-3 modes; emitting all 3 raises downstream R^2 by 1-2% over single-mode FE.

    Returns list[HermiteResult] sorted by MI descending; empty if no mode beats baseline.
    """
    _ret_optimise_pair_multimode_part1 = _optimise_pair_multimode_part1(
        bin_funcs, basis, x_a, x_b, y, n_neighbors, mi_estimator, discrete_target, plugin_n_bins, seed, sweep_degrees, min_degree, max_degree, l2_penalty,
        warm_start, coef_range, n_trials, direction_only, top_m, min_l2_distance, baseline_uplift_threshold,
    )
    _early_optimise_pair_multimode_part1: list = _ret_optimise_pair_multimode_part1[1]
    return _early_optimise_pair_multimode_part1


# ----------------------------------------------------------------------
# Sub-sibling re-export. The 468-LOC ``optimise_hermite_pair`` body
# lives in ``_hermite_fe_optimise_pair.py`` so this file stays below
# the 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._hermite_fe_optimise_pair import optimise_hermite_pair, precompute_hermite_pair_basis  # noqa: F401
