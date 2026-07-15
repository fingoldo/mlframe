"""Hermite/orthogonal-poly pair-FE optimisation routines carved out of
``mlframe.feature_selection.filters.hermite_fe``.

Holds the Optuna/CMA outer-loop search + per-pair evaluation. Re-imported
at the parent's module bottom so historical
``from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair``
resolves transparently.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

logger = logging.getLogger("mlframe.feature_selection.filters.hermite_fe")


def detect_pair_symmetry(
    x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, *, discrete_target: bool = True, mi_estimator: str = "plugin", plugin_n_bins: int = 20
) -> float:
    """Symmetry score in [0, 1] for (x_a, x_b) as predictors of y. Targets of form f(a,b)=f(b,a) score near 1; asymmetric like y=sign(a-2b) score lower.

    Combines (geometric mean of) two indicators:
    1. Marginal MI ratio min(MI(a,y), MI(b,y)) / max(...).
    2. Sub/Add MI ratio MI(|a-b|, y) / MI(a+b, y).

    Score >= 0.95: caller can constrain c_a = c_b to halve search dim. Score <= 0.7: clearly asymmetric -- per-feature basis routing matters more.
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
                     B_a=None, B_b=None):
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
    from .hermite_fe import _L2_PENALTY_SATURATION_DEFAULT, _l2_normalize_pair, _l2_penalty_value, _plugin_mi_classif_batch_rows_njit, _plugin_mi_regression_batch_njit
    if l2_penalty_saturation is None:
        l2_penalty_saturation = _L2_PENALTY_SATURATION_DEFAULT
    if direction_only:
        coef_a, coef_b = _l2_normalize_pair(coef_a, coef_b, target_norm=1.0)
    # 2026-05-18 BASIS-MATRIX FASTPATH (kept but disabled by default):
    # ``B_a`` / ``B_b`` are optional precomputed basis matrices that allow
    # ``h = B @ c`` (BLAS GEMV) instead of Horner. Numerical
    # identity to Horner verified to 1e-16 across all four orthogonal
    # bases. HOWEVER, measured at n=1M / subsample=200k production budget
    # (cProfile 2026-05-18): GEMV on the 1500-sample multi-fidelity
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
    # (n, coef_a.shape[0]) / (n, coef_b.shape[0]) -- this function no longer
    # re-slices per trial (that re-slice+copy ran on every CMA-ES/random-batch
    # trial, tens of thousands of times per pair; the only caller,
    # ``optimise_hermite_pair``, now truncates once per degree). The
    # refinement step in ``optimise_hermite_pair`` explicitly sets B_a
    # = B_b = None when switching to full-n z - omitting that produced
    # shape (1500,) vs (n,) errors / silent hermite=0 regression measured
    # in development.
    if B_a is not None and B_b is not None:
        # B_a / B_b arrive PRE-TRUNCATED to (ca_size, cb_size) by the only caller that ever sets them
        # (optimise_hermite_pair slices once per degree, before the trial loop) -- do not re-slice per trial.
        _ca = np.ascontiguousarray(coef_a, dtype=np.float64)
        _cb = np.ascontiguousarray(coef_b, dtype=np.float64)
        h_a = B_a @ _ca
        h_b = B_b @ _cb
    else:
        h_a = eval_func(z_a, coef_a)
        h_b = (eval_func_b if eval_func_b is not None else eval_func)(z_b, coef_b)
    if not (np.all(np.isfinite(h_a)) and np.all(np.isfinite(h_b))):
        return -np.inf, 0.0, -1
    cols = []
    valid_idx = []
    for k, bf in enumerate(bf_callables):
        try:
            combined = bf(h_a, h_b)
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _hermite_fe_optimise.py:119: %s", e)
            continue
        if np.all(np.isfinite(combined)):
            cols.append(combined)
            valid_idx.append(k)
    if not cols:
        return -np.inf, 0.0, -1
    X_batch = np.ascontiguousarray(np.column_stack(cols), dtype=np.float64)
    if mi_estimator == "plugin":
        if discrete_target:
            mi_arr = _plugin_mi_classif_batch_njit(X_batch, y_njit, plugin_n_bins)
        else:
            mi_arr = _plugin_mi_regression_batch_njit(X_batch, y_njit, plugin_n_bins)
    else:  # ksg
        if discrete_target:
            mi_arr = mutual_info_classif(X_batch, y, n_neighbors=n_neighbors, random_state=42, discrete_features=False)
        else:
            mi_arr = mutual_info_regression(X_batch, y, n_neighbors=n_neighbors,
                                             random_state=42, discrete_features=False)
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

def _eval_coef_pair_batch(coefs_a, coefs_b, *, z_a, z_b, eval_func, bf_callables,
                           bf_names, y, y_njit, mi_estimator, plugin_n_bins,
                           n_neighbors, discrete_target, l2_penalty,
                           l2_penalty_saturation=None,
                           direction_only=False, eval_func_b=None,
                           B_a=None, B_b=None):
    """Batched eval over ``P`` coefficient candidates simultaneously.

    Args:
      coefs_a: ndarray (P, ca_size) -- candidate coefficients for the a-side.
      coefs_b: ndarray (P, cb_size) -- candidate coefficients for the b-side.
      All other kwargs match ``_eval_coef_pair``.

    Returns:
      (best_scores, best_raws, best_idxs) each ndarray (P,) -- per-candidate
      best regularised score, raw MI, and bf index.

    The single-call version dispatches one MI batch call across ALL
    (candidate, bf) column combinations: typical (P=20, K_bf=5) gives a
    100-column batch instead of 20 separate 5-column batches. The plugin
    MI kernel is already numba-batched over columns via prange, so this
    feeds 100 columns into ONE prange loop -- saturates cores fully and
    removes 19 of the 20 Python-side GIL acquire/release cycles between
    successive MI calls. Combined with the per-candidate Python loop for
    polyeval (cheap numba calls), this is the bulk of the speedup vs
    Optuna's per-trial sequential evaluation.
    """
    from .hermite_fe import _L2_PENALTY_SATURATION_DEFAULT, _l2_normalize_pair, _l2_penalty_value, _plugin_mi_classif_batch_rows_njit, _plugin_mi_regression_batch_njit
    if l2_penalty_saturation is None:
        l2_penalty_saturation = _L2_PENALTY_SATURATION_DEFAULT

    P = int(coefs_a.shape[0])
    # Direction-only normalisation per candidate. Loop because
    # _l2_normalize_pair operates on a single pair; cheap.
    if direction_only:
        coefs_a = coefs_a.copy()
        coefs_b = coefs_b.copy()
        for p in range(P):
            ca_n, cb_n = _l2_normalize_pair(coefs_a[p], coefs_b[p], target_norm=1.0)
            coefs_a[p] = ca_n
            coefs_b[p] = cb_n

    # Phase 1: compute h_a / h_b per candidate (one numba call each;
    # Python loop). Skip candidates that produce non-finite h_a/h_b --
    # they get -inf score downstream.
    n_rows = z_a.shape[0]
    h_a_arr = np.empty((P, n_rows), dtype=np.float64)
    h_b_arr = np.empty((P, n_rows), dtype=np.float64)
    cand_valid = np.zeros(P, dtype=bool)
    for p in range(P):
        ca = coefs_a[p]
        cb = coefs_b[p]
        if B_a is not None and B_b is not None:
            # B_a / B_b are pre-truncated to (ca_size, cb_size) by the caller (once per degree) -- see the
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

    # Phase 2: accumulate (candidate, bf) columns into one stacked
    # X_batch. Track (cand_idx, bf_idx) per column for the per-candidate
    # argmax later.
    # njit bf twins (bit-parity with the numpy callables pinned by test_numba_bf_dispatch_parity) replace
    # the numpy ufunc chains: cProfile showed _atan2 (0.44s) + _log_abs_signed (0.41s) + np.column_stack
    # (0.43s) = ~40%% of a cma_batch search. Columns are written straight into the preallocated X_batch
    # (no per-column copy, no stack); unknown bf names keep the numpy callable path.
    try:
        from ._numba_polynom_optimizer import _BF_NAME_TO_ID, _bf_dispatch_njit
    except Exception:
        _BF_NAME_TO_ID, _bf_dispatch_njit = {}, None
    col_meta: list = []  # tuples (p, k)
    X_batch = np.empty((P * len(bf_callables), n_rows), dtype=np.float64)  # ROW-major: contiguous writes + copy-free MI rows kernel
    nc = 0
    for p in range(P):
        if not cand_valid[p]:
            continue
        h_a = h_a_arr[p]
        h_b = h_b_arr[p]
        for k, bf in enumerate(bf_callables):
            bfid = _BF_NAME_TO_ID.get(bf_names[k], -1) if _bf_dispatch_njit is not None else -1
            try:
                if bfid >= 0:
                    combined = _bf_dispatch_njit(bfid, h_a, h_b)
                else:
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        combined = bf(h_a, h_b)
            except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _hermite_fe_optimise.py:234: %s", e)
                continue
            if np.all(np.isfinite(combined)):
                X_batch[nc] = combined
                col_meta.append((p, k))
                nc += 1

    best_scores = np.full(P, -np.inf, dtype=np.float64)
    best_raws = np.zeros(P, dtype=np.float64)
    best_idxs = np.full(P, -1, dtype=np.int64)
    if nc == 0:
        return best_scores, best_raws, best_idxs

    # Phase 3: ONE batched MI call across all (P * K_bf_valid) columns.
    # _plugin_mi_classif_batch_njit kernel pranges over columns -- with
    # P=20 candidates and ~5 bf_callables we feed 100 columns into one
    # call, saturating all cores in a single numba launch.
    X_batch = X_batch[:nc]  # (nc, n) row-major view; rows already contiguous
    if mi_estimator == "plugin":
        if discrete_target:
            mi_arr = _plugin_mi_classif_batch_rows_njit(X_batch, y_njit, plugin_n_bins)
        else:
            mi_arr = _plugin_mi_regression_batch_njit(np.ascontiguousarray(X_batch.T), y_njit, plugin_n_bins)
    else:  # ksg
        if discrete_target:
            mi_arr = mutual_info_classif(X_batch.T, y, n_neighbors=n_neighbors, random_state=42, discrete_features=False)
        else:
            mi_arr = mutual_info_regression(X_batch.T, y, n_neighbors=n_neighbors, random_state=42, discrete_features=False)

    # Phase 4: per-candidate l2 penalty + best-bf selection
    penalties = np.zeros(P, dtype=np.float64)
    if not direction_only and l2_penalty > 0:
        for p in range(P):
            penalties[p] = _l2_penalty_value(
                coefs_a[p], coefs_b[p], l2_penalty, l2_penalty_saturation,
            )
    for j, (p, k) in enumerate(col_meta):
        raw = float(mi_arr[j])
        s = raw - penalties[p]
        if s > best_scores[p]:
            best_scores[p] = s
            best_raws[p] = raw
            best_idxs[p] = k
    return best_scores, best_raws, best_idxs


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
    from .hermite_fe import _l2_normalize_pair
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
        for j, sc in enumerate(ws_scores):
            if not np.isfinite(sc):
                continue
            if track_history and ws_idxs[j] >= 0:
                history.append((float(sc), float(ws_raws[j]), int(ws_idxs[j]), ws_a[j].copy(), ws_b[j].copy()))  # type: ignore[union-attr]  # guarded by track_history, which is True exactly when history was allocated non-None
            if sc > best_score:
                best_score = sc
                best_raw = ws_raws[j]
                best_idx = ws_idxs[j]
                if direction_only:
                    nc_a, nc_b = _l2_normalize_pair(ws_a[j], ws_b[j])
                    best_coefs = (nc_a.copy(), nc_b.copy())
                else:
                    best_coefs = (ws_a[j].copy(), ws_b[j].copy())
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
        except Exception:
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
        # Update best + history
        for j in range(len(sol_batch)):
            sc = scores_batch[j]
            if not np.isfinite(sc):
                continue
            if track_history and idxs_batch[j] >= 0:
                history.append((float(sc), float(raws_batch[j]), int(idxs_batch[j]), ca_batch[j].copy(), cb_batch[j].copy()))  # type: ignore[union-attr]  # guarded by track_history, which is True exactly when history was allocated non-None
            if sc > best_score:
                best_score = sc
                best_raw = raws_batch[j]
                best_idx = idxs_batch[j]
                if direction_only:
                    nc_a, nc_b = _l2_normalize_pair(ca_batch[j], cb_batch[j])
                    best_coefs = (nc_a.copy(), nc_b.copy())
                else:
                    best_coefs = (ca_batch[j].copy(), cb_batch[j].copy())
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
        except Exception:
            break
        if early_stop_no_improve_gens and early_stop_no_improve_gens > 0:
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
    in ONE ``_eval_coef_pair_batch`` call -- pure Python decision
    overhead per iter is ~5 lines, no GIL-locked sampler loop.

    Same return contract as ``_run_cma_search`` so the dispatcher can
    drop it in via ``optimizer="random_batch"``.
    """
    from .hermite_fe import _l2_normalize_pair
    rng = np.random.default_rng(seed if seed > 0 else 1)
    best_score = -np.inf
    best_raw = 0.0
    best_idx = -1
    best_coefs = None
    n_evals = 0
    history: list | None = [] if track_history else None
    sigma = perturb_sigma_frac * (coef_range[1] - coef_range[0])
    n_iters = max(1, int(np.ceil(n_trials / max(1, batch_size))))

    # Warm-start: evaluate canonical seeds in one batch first.
    if warm_start_seeds:
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
            if track_history and ws_idxs[j] >= 0:
                history.append((float(sc), float(ws_raws[j]), int(ws_idxs[j]), ws_a[j].copy(), ws_b[j].copy()))  # type: ignore[union-attr]  # guarded by track_history, which is True exactly when history was allocated non-None
            if sc > best_score:
                best_score = sc
                best_raw = ws_raws[j]
                best_idx = ws_idxs[j]
                if direction_only:
                    nc_a, nc_b = _l2_normalize_pair(ws_a[j], ws_b[j])
                    best_coefs = (nc_a.copy(), nc_b.copy())
                else:
                    best_coefs = (ws_a[j].copy(), ws_b[j].copy())

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


def _select_diverse_topm(history: list, top_m: int, min_l2_distance: float = 0.3) -> list:
    """Greedy diverse-top-M selection from (score, raw_mi, bf_idx, coef_a, coef_b) tuples; keeps entries whose joint (L2-normalized) coef vector is >= min_l2_distance from prior kept.

    Module-private; coefficient vectors of differing lengths are zero-padded to a common axis for cross-degree comparison.
    """
    if not history:
        return []
    # Wave 58 (2026-05-20): secondary key on bf_idx (r[2]) so tied top-MI
    # Hermite history doesn't shift `kept[0]` across iteration orders.
    sorted_h = sorted(history, key=lambda r: (-r[0], r[2]))
    # Pad lengths to the max coef vector for cross-degree comparison.
    max_a = max(e[3].shape[0] for e in sorted_h)
    max_b = max(e[4].shape[0] for e in sorted_h)

    def _padded_vec(coef_a, coef_b):
        """Concatenate + zero-pad two coefficient vectors to a fixed ``max_a + max_b`` length for cross-degree comparison."""
        v = np.zeros(max_a + max_b, dtype=np.float64)
        v[: coef_a.shape[0]] = coef_a
        v[max_a : max_a + coef_b.shape[0]] = coef_b
        return v

    kept = [sorted_h[0]]
    kept_dirs = [_padded_vec(sorted_h[0][3], sorted_h[0][4]) / (np.linalg.norm(_padded_vec(sorted_h[0][3], sorted_h[0][4])) + 1e-12)]
    for entry in sorted_h[1:]:
        if len(kept) >= top_m:
            break
        cand_vec = _padded_vec(entry[3], entry[4])
        cn = np.linalg.norm(cand_vec) + 1e-12
        cand_dir = cand_vec / cn
        is_diverse = True
        for k_dir in kept_dirs:
            cos_sim = float(abs(np.dot(cand_dir, k_dir)))
            cos_sim = min(cos_sim, 1.0)  # numerical safety
            l2_dist = np.sqrt(max(2 * (1 - cos_sim), 0.0))
            if l2_dist < min_l2_distance:
                is_diverse = False
                break
        if is_diverse:
            kept.append(entry)
            kept_dirs.append(cand_dir)
    return kept

def _run_cma_search(*, ca_size, cb_size, coef_range, n_trials, seed,
                     direction_only, warm_start_seeds, eval_kwargs,
                     popsize=None, eval_pair_fn=None,
                     track_history=False,
                     early_stop_no_improve_gens: int | None = None):
    """CMA-ES inner loop. Returns (best_coef_a, best_coef_b, best_bf_idx, best_raw_mi, n_evals). When track_history=True, also returns the full evaluation list.

    CMA minimizes; we negate the MI score. Default popsize=max(8, min(20, n_trials // 8)) -- smaller than CMA's default to allow more generations on tight budgets.

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
    from .hermite_fe import _l2_normalize_pair
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
        # Tighter sigma when we already have a good seed -- exploit
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
    # Inject remaining canonical seeds into the first generation -- CMA
    # 4.x lets us replace ask()'s random samples directly.
    inject_arrays = [np.asarray(s, dtype=np.float64) for s in (warm_start_seeds or [])]

    first_gen = True
    # 2026-05-20 NEW-D: plateau early-stop state. Tracks how many
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
        except Exception:
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
        if early_stop_no_improve_gens and early_stop_no_improve_gens > 0:
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


def _baseline_mi_pair(x_a, x_b, y, *, discrete_target: bool, n_neighbors: int = 3, mi_estimator: str = "plugin", plugin_n_bins: int = 20) -> float:
    """Identity baseline: MI of (x_a, x_b) vs target. Plug-in is 1-D-x by design so we use max(MI(x_a, y), MI(x_b, y)) (lower bound on joint MI); KSG path uses sklearn's multi-D estimator."""
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    from .hermite_fe import _plugin_mi_classif_njit, _plugin_mi_regression_njit
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
        return float(mutual_info_classif(Xn, y, n_neighbors=n_neighbors, random_state=42, discrete_features=False).max())
    return float(mutual_info_regression(Xn, y, n_neighbors=n_neighbors, random_state=42, discrete_features=False).max())


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
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    from .hermite_fe import HermiteResult, _CUDA_AVAILABLE, _CUDA_THRESHOLD, _DEFAULT_BIN_FUNCS, _NJIT_FUNCS, _NJIT_PAR_FUNCS, _PAR_THRESHOLD, _POLY_BASES, _canonical_seeds
    # Forced CMA-ES because diverse top-M needs a bag of evaluations, which CMA's population gives naturally;
    # Optuna's TPE samples are less diverse early-on (coupled by the multivariate prior).
    if bin_funcs is None:
        bin_funcs = _DEFAULT_BIN_FUNCS

    if basis not in _POLY_BASES:
        raise ValueError(f"unknown basis {basis!r}; expected one of {list(_POLY_BASES)}")
    basis_info = _POLY_BASES[basis]

    z_a, preprocess_a = basis_info["fit"](x_a)
    z_b, preprocess_b = basis_info["fit"](x_b)
    z_a = np.ascontiguousarray(z_a, dtype=np.float64)
    z_b = np.ascontiguousarray(z_b, dtype=np.float64)

    # Pick eval_func via the size-aware ladder (mirrors optimise_hermite_pair). For factory bases (RBF/Sigmoid) eval_func_b is a separate closure over
    # preprocess_b so the b-side feature evaluates with its OWN centres/thresholds; for njit polynomial bases both are the same callable.
    factory_top = basis_info.get("eval_njit_factory")
    if factory_top is not None:
        eval_func = factory_top(preprocess_a)
        eval_func_b = factory_top(preprocess_b)
    elif basis in _NJIT_FUNCS:
        n_eval = z_a.shape[0]
        if n_eval < _PAR_THRESHOLD:
            eval_func = basis_info["eval_njit"]
        elif n_eval >= _CUDA_THRESHOLD and _CUDA_AVAILABLE:
            eval_func = basis_info["eval_dispatch"]
        else:
            eval_func = _NJIT_PAR_FUNCS[basis]
        eval_func_b = eval_func
    else:
        eval_func = basis_info["eval_njit"]
        eval_func_b = eval_func

    n = len(y)
    if n_neighbors is None:
        if n >= 5000:
            n_neighbors = 3
        elif n >= 1000:
            n_neighbors = 5
        else:
            n_neighbors = 7

    coef_size_func = basis_info.get("coef_size_func", lambda d: d + 1)
    canonical_seeds_func = basis_info.get("canonical_seeds_func")

    if mi_estimator == "plugin":
        y_njit = np.asarray(y, dtype=np.int64) if discrete_target else np.asarray(y, dtype=np.float64)
    else:
        y_njit = None

    bf_names_global = list(bin_funcs.keys())
    bf_callables_global = [bin_funcs[n] for n in bf_names_global]

    baseline = _baseline_mi_pair(z_a, z_b, y, discrete_target=discrete_target, n_neighbors=n_neighbors, mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins)

    # Aggregate history across degrees, then apply diverse top-M.
    full_history = []
    degree_grid = list(range(min_degree, max_degree + 1)) if sweep_degrees else [max_degree]
    for degree in degree_grid:
        ca_size = coef_size_func(degree)
        cb_size = coef_size_func(degree)
        # ``eval_func_b`` carries the per-feature preprocess fn for factory bases (RBF). For njit bases both eval_func / eval_func_b are the same callable
        # and ``_eval_coef_pair`` falls back transparently. Plumbing this through prevents the b-side from silently re-using preprocess_a on RBF/factory fits.
        eval_kwargs = dict(
            z_a=z_a, z_b=z_b, eval_func=eval_func, eval_func_b=eval_func_b,
            bf_callables=bf_callables_global, bf_names=bf_names_global,
            y=y, y_njit=y_njit,
            mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins,
            n_neighbors=n_neighbors, discrete_target=discrete_target,
            l2_penalty=l2_penalty,
        )
        warm_seeds: list = []
        if warm_start:
            seeds_per_feat = canonical_seeds_func(degree) if canonical_seeds_func else _canonical_seeds(basis, degree)
            for s_a in seeds_per_feat:
                warm_seeds.extend(np.concatenate([s_a, s_b]) for s_b in seeds_per_feat)
            if seeds_per_feat:
                s = seeds_per_feat[0]
                warm_seeds.append(np.concatenate([s, -s]))
        try:
            r = _run_cma_search(
                ca_size=ca_size, cb_size=cb_size, coef_range=coef_range,
                n_trials=n_trials, seed=seed,
                direction_only=direction_only, warm_start_seeds=warm_seeds,
                eval_kwargs=eval_kwargs, track_history=True,
            )
        except Exception as e:
            logger.warning("CMA-ES failed in multimode degree %d: %s", degree, e)
            continue
        if r is None:
            continue
        coef_a, coef_b, bf_idx, raw_mi, _n, history = r
        # Tag history entries with degree so we can rebuild HermiteResult.
        full_history.extend([(s, mi, idx, ca, cb, degree) for s, mi, idx, ca, cb in history])

    if not full_history:
        return []

    # Diverse top-M selection (post-process).
    diverse = _select_diverse_topm(
        [(s, mi, idx, ca, cb) for s, mi, idx, ca, cb, _d in full_history],
        top_m=top_m, min_l2_distance=min_l2_distance,
    )

    # Build degree lookup back into the diverse entries (diverse only carries 5-tuples).
    deg_lookup = {(tuple(ca), tuple(cb)): d for s, mi, idx, ca, cb, d in full_history}

    results = []
    for _score, raw_mi, bf_idx, coef_a, coef_b in diverse:
        if raw_mi <= baseline * baseline_uplift_threshold:
            continue
        bf_name = bf_names_global[bf_idx]
        deg = deg_lookup.get((tuple(coef_a), tuple(coef_b)), len(coef_a) - 1)
        results.append(HermiteResult(
            coef_a=coef_a, coef_b=coef_b,
            bin_func_name=bf_name, bin_func=bin_funcs[bf_name],  # type: ignore[arg-type]  # bin_funcs values are the callables per bf_name; _DEFAULT_BIN_FUNCS' inferred dict-value type is imprecise
            mi=raw_mi, baseline_mi=baseline,
            uplift=raw_mi / max(baseline, 1e-12),
            degree_a=deg, degree_b=deg,
            basis=basis,
            preprocess_a=preprocess_a,
            preprocess_b=preprocess_b,
        ))
    # Wave 58 (2026-05-20): secondary key on (deg_a, deg_b, bf_name) so tied
    # mi doesn't make results[0] depend on insertion order.
    results.sort(key=lambda r: (-r.mi, getattr(r, "degree_a", 0), getattr(r, "degree_b", 0), getattr(r, "bf_name", "")))
    return results


# ----------------------------------------------------------------------
# Sub-sibling re-export. The 468-LOC ``optimise_hermite_pair`` body
# lives in ``_hermite_fe_optimise_pair.py`` so this file stays below
# the 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._hermite_fe_optimise_pair import optimise_hermite_pair, precompute_hermite_pair_basis  # noqa: F401
