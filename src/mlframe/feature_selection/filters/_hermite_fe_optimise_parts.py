"""Helper blocks extracted from the over-long functions of ``_hermite_fe_optimise`` (function-length gate); behaviour is unchanged."""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from mlframe.feature_selection.filters._hermite_fe_diverse import _select_diverse_topm
from mlframe.utils.log_throttle import log_throttle
from .fe_baselines import ksg_random_state

logger = logging.getLogger("mlframe.feature_selection.filters.hermite_fe")


from ._hermite_fe_optimise_helpers import (
    _compact_finite_candidate_rows,
)


def _eval_coef_pair_batch_part1(
    l2_penalty_saturation: Optional[float], coefs_a: np.ndarray, direction_only: bool, coefs_b: np.ndarray, z_a: np.ndarray, z_b: np.ndarray,
    eval_func: Callable[..., Any], eval_func_b: Optional[Callable[..., Any]], B_a: Optional[np.ndarray], B_b: Optional[np.ndarray],
    bf_callables: Sequence[Callable[..., Any]], bf_names: Sequence[str], mi_estimator: str, discrete_target: bool, y_njit: np.ndarray, plugin_n_bins: int,
    y: np.ndarray, n_neighbors: int, random_state: int, l2_penalty: float,
) -> tuple[Any, ...]:
    """Extracted block of ``_eval_coef_pair_batch`` (part 1); returns the names the caller still uses."""
    from ._hermite_fe_optimise import _candidate_h_arrays
    p: Any = None
    k: Any = None
    from mlframe.feature_selection.filters.hermite_fe.shared import L2_PENALTY_SATURATION_DEFAULT as _L2_PENALTY_SATURATION_DEFAULT, l2_normalize_pair as _l2_normalize_pair, l2_penalty_value as _l2_penalty_value, plugin_mi_classif_batch_rows_njit as _plugin_mi_classif_batch_rows_njit, plugin_mi_regression_batch_njit as _plugin_mi_regression_batch_njit
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

    # Phase 1: per-candidate h_a / h_b (see _candidate_h_arrays).
    h_a_arr, h_b_arr, cand_valid = _candidate_h_arrays(
        coefs_a, coefs_b, z_a=z_a, z_b=z_b, eval_func=eval_func, eval_func_b=eval_func_b, B_a=B_a, B_b=B_b,
    )
    n_rows = z_a.shape[0]

    # Phase 2: accumulate (candidate, bf) columns into one stacked
    # X_batch. Track (cand_idx, bf_idx) per column for the per-candidate
    # argmax later.
    # njit bf twins (bit-parity with the numpy callables pinned by test_numba_bf_dispatch_parity) replace
    # the numpy ufunc chains: cProfile showed _atan2 (0.44s) + _log_abs_signed (0.41s) + np.column_stack
    # (0.43s) = ~40%% of a cma_batch search. Columns are written straight into the preallocated X_batch
    # (no per-column copy, no stack); unknown bf names keep the numpy callable path.
    _bf_name_to_id: dict[str, int]
    try:
        from ._numba_polynom_optimizer import _BF_NAME_TO_ID as _bf_name_to_id, _bf_dispatch_njit, _fill_bf_batch_njit  # noqa: N811
    except ImportError as e:
        logger.warning("_numba_polynom_optimizer import failed: %s", e)
        _bf_name_to_id, _bf_dispatch_njit, _fill_bf_batch_njit = {}, None, None
    col_meta: list = []  # tuples (p, k)
    KBF = len(bf_callables)
    X_batch = np.empty((P * KBF, n_rows), dtype=np.float64)  # ROW-major: contiguous writes + copy-free MI rows kernel
    nc = 0
    # The all-known-bfs case runs ONE parallel njit fill over every (candidate, bf) row - bf compute,
    # finiteness scan, and the row store all inside a single prange (no 120x Python->njit crossings, allocs,
    # or isfinite passes). Any unknown bf name falls back to the per-column Python loop below.
    _bf_ids = [_bf_name_to_id.get(nm, -1) for nm in bf_names] if _bf_name_to_id else [-1] * KBF
    if _fill_bf_batch_njit is not None and all(i >= 0 for i in _bf_ids):
        finite = np.zeros(P * KBF, dtype=np.bool_)
        _fill_bf_batch_njit(h_a_arr, h_b_arr, cand_valid, np.asarray(_bf_ids, dtype=np.int64), X_batch, finite)
        nc = _compact_finite_candidate_rows(P, KBF, finite, nc, X_batch, col_meta)
    else:
        for p in range(P):
            if not cand_valid[p]:
                continue
            h_a = h_a_arr[p]
            h_b = h_b_arr[p]
            for k, bf in enumerate(bf_callables):
                bfid = _bf_ids[k] if _bf_dispatch_njit is not None else -1
                try:
                    if bfid >= 0:
                        combined = _bf_dispatch_njit(bfid, h_a, h_b)
                    else:
                        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                            combined = bf(h_a, h_b)
                except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed: %s", e)
                    continue
                if np.all(np.isfinite(combined)):
                    X_batch[nc] = combined
                    col_meta.append((p, k))
                    nc += 1

    best_scores = np.full(P, -np.inf, dtype=np.float64)
    best_raws = np.zeros(P, dtype=np.float64)
    best_idxs = np.full(P, -1, dtype=np.int64)
    if nc == 0:
        return True, (best_scores, best_raws, best_idxs)

    # Phase 3: ONE batched MI call across all (P * K_bf_valid) columns.
    # _plugin_mi_classif_batch_njit kernel pranges over columns - with
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
            mi_arr = mutual_info_classif(X_batch.T, y, n_neighbors=n_neighbors, random_state=ksg_random_state(random_state), discrete_features=False)
        else:
            mi_arr = mutual_info_regression(X_batch.T, y, n_neighbors=n_neighbors, random_state=ksg_random_state(random_state), discrete_features=False)

    # Phase 4: per-candidate l2 penalty + best-bf selection
    penalties = np.zeros(P, dtype=np.float64)
    if not direction_only and l2_penalty > 0:
        for p in range(P):
            penalties[p] = _l2_penalty_value(
                coefs_a[p], coefs_b[p], l2_penalty, l2_penalty_saturation,
            )
    return False, p, col_meta, k, best_scores, best_raws, best_idxs, mi_arr, penalties


def _optimise_pair_multimode_part1(
    bin_funcs: dict | None, basis: str, x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, n_neighbors: int | None, mi_estimator: str, discrete_target: bool,
    plugin_n_bins: int, seed: int, sweep_degrees: bool, min_degree: int, max_degree: int, l2_penalty: float, warm_start: bool, coef_range: tuple,
    n_trials: int, direction_only: bool, top_m: int, min_l2_distance: float, baseline_uplift_threshold: float,
) -> tuple[Any, ...]:
    """Extracted block of ``optimise_pair_multimode`` (part 1); returns the names the caller still uses."""
    from ._hermite_fe_optimise import _baseline_mi_pair, _run_cma_search
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    from mlframe.feature_selection.filters.hermite_fe.shared import HermiteResult, CUDA_AVAILABLE as _CUDA_AVAILABLE, CUDA_THRESHOLD as _CUDA_THRESHOLD, DEFAULT_BIN_FUNCS as _DEFAULT_BIN_FUNCS, NJIT_FUNCS as _NJIT_FUNCS, NJIT_PAR_FUNCS as _NJIT_PAR_FUNCS, PAR_THRESHOLD as _PAR_THRESHOLD, POLY_BASES as _POLY_BASES, canonical_seeds as _canonical_seeds
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

    baseline = _baseline_mi_pair(z_a, z_b, y, discrete_target=discrete_target, n_neighbors=n_neighbors, mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins, random_state=seed)

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
            l2_penalty=l2_penalty, random_state=seed,
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
            log_throttle(logger, "hermite_cma_multimode_degree_failed", logging.WARNING, "CMA-ES failed in multimode degree %d: %s", degree, e)
            continue
        if r is None:
            continue
        coef_a, coef_b, bf_idx, raw_mi, _n, history = r
        # Tag history entries with degree so we can rebuild HermiteResult.
        full_history.extend([(s, mi, idx, ca, cb, degree) for s, mi, idx, ca, cb in history])

    if not full_history:
        return True, ([])

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
    # Secondary key on (deg_a, deg_b, bf_name) so tied
    # mi doesn't make results[0] depend on insertion order.
    results.sort(key=lambda r: (-r.mi, getattr(r, "degree_a", 0), getattr(r, "degree_b", 0), r.bin_func_name))
    return True, (results)
    return False, None
