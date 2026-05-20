"""Polynom-pair FE block extracted from ``MRMR._run_fe_step``.

Pre-2026-05-18 this code lived inline inside the ~600 LOC ``_run_fe_step``
method and ran SERIALLY (the ``for raw_vars_pair`` outer loop ignored
``n_jobs``). On n=4M production data with default config this could spend
11+ minutes single-threaded.

Public entry: ``run_polynom_pair_fe``.

Responsibilities:
- Parallel per-pair evaluation via ``joblib.Parallel(backend="threading")``
  (CMA-ES / Optuna release the GIL during numpy / numba MI work).
- Serial reduce: injection of surviving engineered columns back into
  ``data`` / ``cols`` / ``nbins`` / ``X``, plus mutation of
  ``engineered_features`` / ``engineered_recipes`` / ``hermite_features_list``.
- Periodic progress logging so the operator doesn't see silent minutes.

Kept SEPARATE from ``mrmr.py`` so:
1. The MRMR class is easier to read (~200 LOC less).
2. The polynom block is independently profilable / optimisable (numba /
   cupy candidates land here without touching MRMR).
3. Future tests can drive the block directly without spinning a full
   MRMR fit.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from joblib import Parallel, delayed

from .discretization import discretize_array
from .engineered_recipes import build_hermite_pair_recipe
from .hermite_fe import optimise_hermite_pair

logger = logging.getLogger(__name__)


def run_polynom_pair_fe(
    *,
    X: Any,
    is_polars_input: bool,
    prospective_pairs: Dict,
    classes_y: np.ndarray,
    cols: List[str],
    nbins: np.ndarray,
    data: np.ndarray,
    engineered_features: Set[str],
    engineered_recipes: Dict[str, Any],
    hermite_features_list: List[Dict[str, Any]],
    feature_names_in: List[str],
    # MRMR config (polynom-FE knobs)
    fe_smart_polynom_iters: int,
    fe_smart_polynom_optimization_steps: int,
    fe_min_polynom_degree: int,
    fe_max_polynom_degree: int,
    fe_min_polynom_coeff: float,
    fe_max_polynom_coeff: float,
    fe_min_engineered_mi_prevalence: float,
    fe_hermite_l2_penalty: float,
    fe_polynomial_basis: str,
    fe_mi_estimator: str,
    fe_optimizer: str,
    fe_warm_start: bool,
    fe_multi_fidelity: bool,
    # quantization (used by injection)
    quantization_nbins: int,
    quantization_method: str,
    quantization_dtype: Any,
    # dispatch
    n_jobs: int,
    verbose: int,
    # 2026-05-18: subsample inside the CMA-ES / Optuna search to bound
    # per-pair MI compute. On production n=4M data each trial evaluates
    # MI on the full y -- ``_plugin_mi_classif_njit`` then takes ~100ms
    # per call x ~250 trials per restart x N_restarts x N_pairs blows
    # past acceptable wall time. With subsample_n=50_000 the inner
    # optimisation operates on a representative slice; the final
    # injected column is computed from the FULL (n, 1) source so no
    # train-time precision is lost. 0 = use full data (legacy).
    subsample_n: int = 0,
    subsample_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], Any]:
    """Run polynom-pair FE: parallel evaluate prospective pairs, serially inject survivors.

    Mutates ``engineered_features``, ``engineered_recipes``, and
    ``hermite_features_list`` in place. Returns updated
    ``(data, nbins, cols, X)`` because numpy / pandas mutations there are
    non-in-place (np.append returns new array; pd.DataFrame[col]= mutates
    but the caller often holds the same ref, so we return for explicitness).

    ``feature_names_in`` is used only to deduce existing column names; not
    mutated.
    """
    if not fe_smart_polynom_iters:
        return data, nbins, cols, X
    if is_polars_input:
        import polars as pl
    else:
        pl = None  # noqa: F841

    # ``prospective_pairs`` is keyed by ``(raw_vars_pair, _pair_mi)`` composite
    # tuples; the polynom-FE body only needs ``raw_vars_pair`` itself.
    _pair_keys = [k[0] for k in prospective_pairs.keys()]
    _n_pairs_to_eval = len(_pair_keys)
    if _n_pairs_to_eval == 0:
        return data, nbins, cols, X

    _polynom_n_jobs = int(n_jobs) if n_jobs and n_jobs > 0 else 1
    logger.info(
        "Polynomial-pair FE starting: %d pair(s) x %d Optuna restart(s) x "
        "%d trial(s) per restart, n_jobs=%d backend=threading.",
        _n_pairs_to_eval, fe_smart_polynom_iters,
        fe_smart_polynom_optimization_steps, _polynom_n_jobs,
    )

    def _eval_one_pair(raw_vars_pair):
        if is_polars_input:
            vals_a_full = X[:, raw_vars_pair[0]].to_numpy()
            vals_b_full = X[:, raw_vars_pair[1]].to_numpy()
        else:
            vals_a_full = X.iloc[:, raw_vars_pair[0]].values
            vals_b_full = X.iloc[:, raw_vars_pair[1]].values
        if np.std(vals_a_full) < 1e-12 or np.std(vals_b_full) < 1e-12:
            return None
        # 2026-05-18 subsample for CMA-ES inner search. Final transform on
        # FULL source array preserves precision; only the optimiser's MI
        # evaluation uses the slice.
        if subsample_n and 0 < subsample_n < len(vals_a_full):
            _ss_rng = np.random.default_rng(subsample_seed + int(raw_vars_pair[0]) * 1000 + int(raw_vars_pair[1]))
            _ss_idx = _ss_rng.choice(len(vals_a_full), size=subsample_n, replace=False)
            vals_a_sub = vals_a_full[_ss_idx]
            vals_b_sub = vals_b_full[_ss_idx]
            classes_y_sub = classes_y[_ss_idx] if hasattr(classes_y, "__getitem__") else classes_y
        else:
            vals_a_sub = vals_a_full
            vals_b_sub = vals_b_full
            classes_y_sub = classes_y
        best_res = None
        for seed_offset in range(fe_smart_polynom_iters):
            res = optimise_hermite_pair(
                x_a=vals_a_sub, x_b=vals_b_sub, y=classes_y_sub,
                discrete_target=True,
                max_degree=fe_max_polynom_degree,
                min_degree=fe_min_polynom_degree,
                n_trials=fe_smart_polynom_optimization_steps,
                coef_range=(fe_min_polynom_coeff, fe_max_polynom_coeff),
                l2_penalty=fe_hermite_l2_penalty,
                n_neighbors=None,
                seed=42 + seed_offset,
                sweep_degrees=True,
                basis=fe_polynomial_basis,
                mi_estimator=fe_mi_estimator,
                optimizer=fe_optimizer,
                warm_start=fe_warm_start,
                multi_fidelity=fe_multi_fidelity,
            )
            if res is not None and (best_res is None or res.mi > best_res.mi):
                best_res = res
        # Return FULL arrays so the injection step applies the polynomial
        # to all rows (subsampling was only for the optimiser's MI loop).
        return (raw_vars_pair, best_res, vals_a_full, vals_b_full)

    _poly_t0 = time.perf_counter()
    if _polynom_n_jobs > 1 and _n_pairs_to_eval > 1:
        _poly_pair_results = Parallel(
            n_jobs=_polynom_n_jobs, backend="threading",
            prefer="threads", verbose=10 if verbose else 0,
        )(delayed(_eval_one_pair)(rv) for rv in _pair_keys)
    else:
        _poly_pair_results = [_eval_one_pair(rv) for rv in _pair_keys]
    _eval_elapsed = time.perf_counter() - _poly_t0
    logger.info(
        "Polynomial-pair FE eval phase done in %.1fs (%d pairs, "
        "%.2fs/pair median).",
        _eval_elapsed, _n_pairs_to_eval,
        _eval_elapsed / max(_n_pairs_to_eval, 1),
    )

    # Serial reduce: log + uplift gate + inject into data/cols/X.
    _uplift_gate = float(fe_min_engineered_mi_prevalence)
    for _pair_result in _poly_pair_results:
        if _pair_result is None:
            continue
        raw_vars_pair, best_res, vals_a, vals_b = _pair_result
        if best_res is not None and verbose:
            logger.info(
                "Polynomial-pair FE (%s): pair=%s baseline_mi=%.4f best_mi=%.4f "
                "uplift=%.2fx degree=%d bf=%s |c|2=(%.2f, %.2f)",
                best_res.basis, raw_vars_pair, best_res.baseline_mi, best_res.mi,
                best_res.uplift, best_res.degree_a, best_res.bin_func_name,
                np.linalg.norm(best_res.coef_a), np.linalg.norm(best_res.coef_b),
            )
        if best_res is None:
            continue
        if best_res.mi <= best_res.baseline_mi * _uplift_gate:
            continue
        try:
            _t_vals = np.asarray(
                best_res.transform(vals_a, vals_b), dtype=np.float64,
            ).reshape(-1)
            if not np.all(np.isfinite(_t_vals)):
                continue
            _src_a = (
                cols[raw_vars_pair[0]]
                if 0 <= raw_vars_pair[0] < len(cols)
                else f"col{raw_vars_pair[0]}"
            )
            _src_b = (
                cols[raw_vars_pair[1]]
                if 0 <= raw_vars_pair[1] < len(cols)
                else f"col{raw_vars_pair[1]}"
            )
            _new_col_name = (
                f"_polynom_{best_res.basis}_{best_res.bin_func_name}"
                f"__{_src_a}__{_src_b}"
            )
            if _new_col_name in cols:
                continue
            _new_binned = discretize_array(
                arr=_t_vals,
                n_bins=quantization_nbins,
                method=quantization_method,
                dtype=quantization_dtype,
            ).reshape(-1, 1)
            data = np.append(data, _new_binned, axis=1)
            nbins = np.concatenate([
                np.asarray(nbins),
                np.asarray([int(quantization_nbins)], dtype=nbins.dtype),
            ])
            cols = cols + [_new_col_name]
            if is_polars_input:
                X = X.with_columns(pl.Series(_new_col_name, _t_vals))  # noqa: F821
            else:
                X[_new_col_name] = _t_vals
            engineered_features.add(_new_col_name)
            hermite_features_list.append({
                "name": _new_col_name,
                "src_a": _src_a, "src_b": _src_b,
                "basis": best_res.basis,
                "bin_func_name": best_res.bin_func_name,
                "degree_a": int(best_res.degree_a),
                "degree_b": int(best_res.degree_b),
                "best_mi": float(best_res.mi),
                "baseline_mi": float(best_res.baseline_mi),
            })
            engineered_recipes[_new_col_name] = build_hermite_pair_recipe(
                name=_new_col_name,
                src_names=(_src_a, _src_b),
                hermite_result=best_res,
            )
            if verbose:
                logger.info(
                    "Polynomial-pair FE injected new feature '%s' "
                    "(mi=%.4f, baseline_mi=%.4f, uplift=%.2fx).",
                    _new_col_name, float(best_res.mi),
                    float(best_res.baseline_mi), float(best_res.uplift),
                )
        except Exception as _inj_err:
            if verbose:
                logger.warning(
                    "Polynomial-pair FE injection failed for pair=%s: %s. "
                    "Standard FE block below still runs.",
                    raw_vars_pair, _inj_err,
                )
    return data, nbins, cols, X
