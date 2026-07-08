"""``optimise_hermite_pair`` sub-carved out of
``mlframe.feature_selection.filters._hermite_fe_optimise`` for the
2026-05-22 sub-split that brings _hermite_fe_optimise below 1k LOC.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .hermite_fe import HermiteResult
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

logger = logging.getLogger("mlframe.feature_selection.filters.hermite_fe")


def optimise_hermite_pair(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    *,
    discrete_target: bool = True,
    bin_funcs: dict | None = None,
    max_degree: int = 4,
    min_degree: int = 2,
    n_trials: int = 200,
    coef_range: tuple = (-2.0, 2.0),
    l2_penalty: float = 0.05,
    l2_penalty_saturation: float | None = None,
    n_neighbors: int | None = None,
    seed: int = 42,
    sweep_degrees: bool = True,
    baseline_uplift_threshold: float = 1.01,
    early_stop_no_improve: int = 50,
    basis: str = "chebyshev",
    mi_estimator: str = "plugin",
    plugin_n_bins: int = 20,
    optimizer: str = "cma_batch",
    warm_start: bool = True,
    warm_start_als: bool = True,
    # CROSS-FIT recipe warm-start prior (backlog idea #20): joint coefficient
    # vectors (concat(coef_a, coef_b)) from a prior fit on an X-fingerprint-
    # overlapping fold. Injected as EXTRA optimiser warm-start seeds; never
    # changes admission (the winner is re-scored on THIS fold's data + passes
    # the same gates). ``None`` / empty = no cross-fit prior (legacy behaviour,
    # byte-identical warm-start population).
    cross_fit_prior_seeds: list | None = None,
    direction_only: bool = False,
    multi_fidelity: bool = True,
    use_trivial_baseline: bool = True,
    precomputed_trivial_baseline: float | None = None,
    precomputed_trivial_name: str | None = None,
    noise_floor_perm_ratio: float = 1.50,
    noise_floor_n_perms: int = 50,
) -> HermiteResult | None:
    """Find polynomial coefficients c_a, c_b that maximise MI(bin_func(P(x_a, c_a), P(x_b, c_b)), y) over the requested
    Optuna/CMA budget. Standardises inputs, regularises coefficients, and only returns a result when the engineered MI
    strictly beats the identity baseline by baseline_uplift_threshold.

    Knob tuning notes
    -----------------
    * basis="chebyshev" (default) wins empirically across 12 regimes (synthetic + UCI California Housing + UCI Diabetes +
      bounded / heavy-tailed) -- never finishes last, highest minimum MI. Pass basis="hermite" for synthetic Gaussian inputs
      or basis="laguerre" for skewed-positive. See _benchmarks/bench_polynomial_bases.py.
    * l2_penalty=0.05 weights a SCALE-SATURATING coefficient penalty (see ``hermite_fe._l2_penalty_value``): it rises toward a constant
      ``l2_penalty`` ceiling as ``||c||^2`` grows instead of growing without bound, so high-MI / high-coefficient solutions (e.g. a separable
      Chebyshev reconstruction of a non-monotone product, ``||c||^2`` ~ 86) are not crushed while pure-noise small-||c|| candidates still pay
      ~full ``l2_penalty``. ``l2_penalty_saturation`` (default ``hermite_fe._L2_PENALTY_SATURATION_DEFAULT`` = 1.0) sets the ||c||^2 scale at
      which the penalty reaches half its ceiling; pass ``l2_penalty_saturation<=0`` for the legacy raw ``l2_penalty * ||c||^2`` behaviour.
    * warm_start_als=True (default) seeds the optimiser with a per-operand rank-1 ALS fit of ``y ~ f(x_a)*g(x_b)`` in the basis (see
      ``hermite_fe.warm_start_als_seed``). This lands the search directly in the true (possibly large-coefficient) basin -- without it cma_batch
      can be trapped on a deceptive atan2/div plateau for non-monotone inner distortions. Polynomial bases only; factory/KSG paths skip it.
    * n_neighbors (KSG): None auto-picks 3 for n>=5000, 5 for n in [1000,5000), 7 for n<1000.
    * max_degree=4 covers most smooth targets. For high-frequency targets raise to 6-8 (n_trials proportionally).
    * early_stop_no_improve: stop a study early if no improvement in the last N trials.
    * mi_estimator="plugin" (default) uses an njit plug-in estimator on quantile-binned values -- ~50-100x faster than
      sklearn's KSG, rank-equivalent for optimization (constant entropy bias). Pass "ksg" for sklearn's KSG.
    * plugin_n_bins=20 (default): ~sqrt(n) rule-of-thumb; larger bins reduce bias, raise variance.
    * noise_floor_perm_ratio=1.50 (default): a permutation-null guard against the high-capacity optimiser fabricating an
      engineered feature on a target INDEPENDENT of the inputs. The plug-in MI estimator has a binning-bias floor that the
      optimiser can overfit on pure noise -- on a noise target the best engineered MI barely clears both the trivial baseline
      and ``baseline_uplift_threshold``, so the uplift gate alone passes it through. The permutation null measures that floor
      directly: re-evaluate the winning engineered column's MI against ``noise_floor_n_perms`` shuffles of y (destroys any real
      dependence, keeps the binning bias) and reject (return None) when ``mi_real < perm_null_p95 * noise_floor_perm_ratio``.
      A genuine feature beats its own permutation-null p95 by 40x+; a noise feature by only ~1.2x. Set ``noise_floor_perm_ratio<=0``
      (or ``noise_floor_n_perms<=0``) to disable. Measured separation in the MRMR discrete path (n=4000, 20 restarts x 3 noise
      pairs): pure-noise max ratio 1.235 (reject), F-POLY 40.8x / F-OSC 61.7x (pass) -- 1.50 has comfortable margin both ways.

    Returns HermiteResult or None if the search failed to beat the baseline.
    """
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    from .hermite_fe import HermiteResult, _BASIS_BUILDERS, _CUDA_AVAILABLE, _CUDA_THRESHOLD, _DEFAULT_BIN_FUNCS, _L2_PENALTY_SATURATION_DEFAULT, _NJIT_FUNCS, _NJIT_PAR_FUNCS, _PAR_THRESHOLD, _POLY_BASES, _canonical_seeds, _l2_normalize_pair, _l2_penalty_value, _plugin_mi_classif_batch_njit, _plugin_mi_regression_batch_njit, build_basis_matrix, warm_start_als_seed
    # Sister-sibling import: ``_baseline_mi_pair``, ``_eval_coef_pair``,
    # ``_run_cma_search`` stayed in ``_hermite_fe_optimise``. Sister-to-sister
    # is cycle-free because the parent imports each sibling at its bottom
    # without either sibling importing the other at module-top.
    from ._hermite_fe_optimise import _baseline_mi_pair, _eval_coef_pair, _run_cma_search
    if mi_estimator not in ("plugin", "ksg"):
        raise ValueError(f"unknown mi_estimator={mi_estimator!r}; expected 'plugin' or 'ksg'")
    if optimizer not in ("optuna", "cma", "cma_batch", "random_batch", "numba_kernel"):
        raise ValueError(f"unknown optimizer={optimizer!r}; expected one of " f"'optuna', 'cma', 'cma_batch', 'random_batch', 'numba_kernel'")
    if l2_penalty_saturation is None:
        l2_penalty_saturation = _L2_PENALTY_SATURATION_DEFAULT
    # Auto-pick n_neighbors based on n.
    n = len(y)
    if n_neighbors is None:
        if n >= 5000:
            n_neighbors = 3
        elif n >= 1000:
            n_neighbors = 5
        else:
            n_neighbors = 7
    # Optuna is only needed on the ``optimizer="optuna"`` branch. Defer the
    # import (and its verbosity-level mutation) to that branch so installs
    # without optuna can still use ``cma`` / ``cma_batch`` / ``random_batch``
    # / ``numba_kernel`` optimisers. The error message is preserved for the
    # actual Optuna path.
    optuna: Any = None
    TPESampler: Any = None
    if optimizer == "optuna":
        try:
            import optuna
            from optuna.samplers import TPESampler
            # TPESampler(multivariate=True) emits ExperimentalWarning per study
            # init; flag has been "experimental" since 2020 and is the recommended
            # setting for correlated params — suppress the noise.
            import warnings as _w
            try:
                from optuna.exceptions import ExperimentalWarning
                _w.filterwarnings("ignore", category=ExperimentalWarning)
            except ImportError:
                pass
        except ImportError as e:
            raise ImportError(
                "optimise_hermite_pair(optimizer='optuna') requires the optional "
                "optuna package. Install via pip install optuna, or pick an "
                "in-tree optimiser ('cma' / 'cma_batch' / 'random_batch' / "
                "'numba_kernel')."
            ) from e
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    bin_funcs = bin_funcs or _DEFAULT_BIN_FUNCS

    if basis not in _POLY_BASES:
        raise ValueError(f"unknown basis {basis!r}; expected one of {list(_POLY_BASES)}")
    basis_info = _POLY_BASES[basis]

    # Preprocess inputs to the basis's natural domain.
    z_a, preprocess_a = basis_info["fit"](x_a)
    z_b, preprocess_b = basis_info["fit"](x_b)
    z_a = np.ascontiguousarray(z_a, dtype=np.float64)
    z_b = np.ascontiguousarray(z_b, dtype=np.float64)
    # Hoist size-aware dispatch out of the hot trial loop: pick the backend ONCE per call (n is fixed across trials).
    # Saves ~4us/call closure overhead, ~5ms over 1000+ trials.
    n_eval = z_a.shape[0]
    factory_top = basis_info.get("eval_njit_factory")
    eval_func: Any
    if factory_top is not None:
        # Non-polynomial basis with data-dependent eval (RBF/Sigmoid). Factory is invoked below per-feature.
        eval_func = None
    elif basis in _NJIT_FUNCS:
        # Polynomial basis -- size-aware ladder applies.
        if n_eval < _PAR_THRESHOLD:
            eval_func = basis_info["eval_njit"]
        elif n_eval >= _CUDA_THRESHOLD and _CUDA_AVAILABLE:
            eval_func = basis_info["eval_dispatch"]  # cuda path
        else:
            eval_func = _NJIT_PAR_FUNCS[basis]
    else:
        # Other non-polynomial basis with simple eval_njit (Fourier, Pade).
        eval_func = basis_info["eval_njit"]

    baseline = _baseline_mi_pair(z_a, z_b, y, discrete_target=discrete_target, n_neighbors=n_neighbors, mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins)
    logger.debug("baseline MI(pair, y) = %.4f", baseline)

    # Stronger gate than the identity max(MI(x_a, y), MI(x_b, y)): try trivial pair-feature transforms
    # (mul, ratio, sum_sq, atan2, ...) and use BEST trivial MI as baseline. Often a simple mul(x_a, x_b)
    # captures most of the signal a polynomial would (verified on XOR / circle / saddle / UCI).
    #
    # 2026-05-20 NEW-A: callers running multiple ``fe_smart_polynom_iters``
    # restarts per pair can pre-compute the trivial baseline once and feed
    # it in via ``precomputed_trivial_baseline`` (+ ``precomputed_trivial_name``);
    # this elides ~5x duplicated 50-150ms ``best_trivial_pair`` calls per
    # pair on the n=200k production config.
    trivial_baseline_name = precomputed_trivial_name
    if use_trivial_baseline and precomputed_trivial_baseline is None:
        try:
            from .fe_baselines import best_trivial_pair
            trivial = best_trivial_pair(
                np.asarray(x_a, dtype=np.float64),
                np.asarray(x_b, dtype=np.float64), y,
                discrete_target=discrete_target,
                mi_estimator=mi_estimator,
                plugin_n_bins=plugin_n_bins,
                n_neighbors=n_neighbors,
            )
            if trivial is not None:
                trivial_baseline_name, _, trivial_mi = trivial
                if trivial_mi > baseline:
                    logger.debug(
                        "trivial baseline %r raises baseline from %.4f to %.4f",
                        trivial_baseline_name, baseline, trivial_mi,
                    )
                    baseline = trivial_mi
        except Exception as e:
            logger.debug("trivial baseline check failed: %s", e)
    elif precomputed_trivial_baseline is not None:
        # Caller supplied the precomputed value -- use it directly.
        if precomputed_trivial_baseline > baseline:
            logger.debug(
                "trivial baseline %r raises baseline from %.4f to %.4f (precomputed)",
                precomputed_trivial_name, baseline, precomputed_trivial_baseline,
            )
            baseline = float(precomputed_trivial_baseline)

    # Pre-cast y once for the njit fast path.
    if mi_estimator == "plugin":
        y_njit = np.asarray(y, dtype=np.int64) if discrete_target else np.asarray(y, dtype=np.float64)
    else:
        y_njit = None  # KSG path does not need it

    best: HermiteResult | None = None

    degree_grid = list(range(min_degree, max_degree + 1)) if sweep_degrees else [max_degree]

    bf_names_global = list(bin_funcs.keys())
    bf_callables_global = [bin_funcs[n] for n in bf_names_global]

    # Multi-fidelity subsample ladder: for large n, fit coefficients on a small subsample (saves O(n) MI work)
    # and refine on full data at the end. With 2*(d+1) <= 8 coefficients, 1500 samples is enough to estimate stably.
    n_full = z_a.shape[0]
    if multi_fidelity and n_full >= 4000:
        rng_mf = np.random.default_rng(seed if seed > 0 else 0)
        sub_idx = rng_mf.choice(n_full, size=1500, replace=False)
        z_a_search = np.ascontiguousarray(z_a[sub_idx], dtype=np.float64)
        z_b_search = np.ascontiguousarray(z_b[sub_idx], dtype=np.float64)
        y_search = y_njit[sub_idx] if y_njit is not None else None
        y_search_any = y[sub_idx] if isinstance(y, np.ndarray) else np.asarray(y)[sub_idx]
    else:
        z_a_search = z_a
        z_b_search = z_b
        y_search = y_njit
        y_search_any = y

    # Coef-size lookup: polynomial bases use degree + 1; non-poly bases (Fourier 2K, RBF up to 9, Pade 2p+1) override.
    coef_size_func = basis_info.get("coef_size_func", lambda d: d + 1)
    canonical_seeds_func = basis_info.get("canonical_seeds_func")

    # Factory-based bases (RBF, Sigmoid) eval depends on train-fold-fitted centres / thresholds. Build per-basis
    # eval once preprocess params are known. Wave 69 (2026-05-20): separate eval for x_a and x_b already implemented
    # below -- factory is called twice with preprocess_a vs preprocess_b, producing distinct eval kernels per side.
    factory = basis_info.get("eval_njit_factory")
    if factory is not None:
        eval_func = factory(preprocess_a)
        eval_func_b = factory(preprocess_b)
    else:
        eval_func_b = eval_func

    # 2026-05-18 PERFORMANCE: precompute basis matrices once per pair for
    # BLAS GEMV fastpath. Initial 2026-05-18 measurement (different
    # hardware) found zero speedup at multi_fidelity=True scale and
    # gated the basis-matrix path OFF for that case. Re-measured
    # 2026-05-20 on current hardware (numba 0.59, MKL BLAS) at the same
    # 1500-element inner CMA-ES scale showed BLAS GEMV is **1.13-1.19x
    # faster than ``@njit(parallel=True)`` Horner** — slice-copy
    # overhead and recurrence pipelining no longer cancel. Gate flipped
    # to build B matrices for ALL polynomial bases (including under
    # multi_fidelity=True). The refinement step at the bottom of this
    # function still drops B_a / B_b before evaluating on full z (see
    # ``full_kwargs["B_a"] = None`` line below) so the
    # subsample-sized matrices never leak into the full-n evaluation.
    B_a_search = None
    B_b_search = None
    _multi_fidelity_active = bool(multi_fidelity and n_full >= 4000)
    if factory is None and basis in _BASIS_BUILDERS:
        try:
            B_a_search = build_basis_matrix(basis, z_a_search, max_degree)
            B_b_search = build_basis_matrix(basis, z_b_search, max_degree)
        except Exception as _bm_err:
            logger.debug("build_basis_matrix failed for %r: %s", basis, _bm_err)
            B_a_search = None
            B_b_search = None

    for degree in degree_grid:
        ca_size = coef_size_func(degree)
        cb_size = coef_size_func(degree)

        # Shared kwargs for both Optuna and CMA paths. When eval_func differs per feature (factory-based bases
        # like RBF), wrap _eval_coef_pair to use both eval_func and eval_func_b.
        eval_pair_fn: Any
        if factory is not None:
            def _eval_dual(coef_a, coef_b, **kw):
                """Evaluate a candidate (coef_a, coef_b) pair against a factory-based basis (each operand using its own eval_func/eval_func_b); returns -inf score on any non-finite basis output."""
                from numpy import column_stack, ascontiguousarray, all as npall, isfinite
                z_a_loc = kw["z_a"]
                z_b_loc = kw["z_b"]
                bf_call = kw["bf_callables"]
                if kw.get("direction_only"):
                    coef_a, coef_b = _l2_normalize_pair(coef_a, coef_b, 1.0)
                h_a = eval_func(z_a_loc, coef_a)
                h_b = eval_func_b(z_b_loc, coef_b)
                if not (npall(isfinite(h_a)) and npall(isfinite(h_b))):
                    return -np.inf, 0.0, -1
                cols = []
                valid_idx = []
                for k, bf in enumerate(bf_call):
                    try:
                        combined = bf(h_a, h_b)
                    except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                        logger.debug("suppressed in _hermite_fe_optimise_pair.py:311: %s", e)
                        continue
                    if npall(isfinite(combined)):
                        cols.append(combined)
                        valid_idx.append(k)
                if not cols:
                    return -np.inf, 0.0, -1
                X_batch = ascontiguousarray(column_stack(cols), dtype=np.float64)
                if kw["mi_estimator"] == "plugin":
                    if kw["discrete_target"]:
                        mi_arr = _plugin_mi_classif_batch_njit(X_batch, kw["y_njit"], kw["plugin_n_bins"])
                    else:
                        mi_arr = _plugin_mi_regression_batch_njit(X_batch, kw["y_njit"], kw["plugin_n_bins"])
                else:
                    if kw["discrete_target"]:
                        mi_arr = mutual_info_classif(X_batch, kw["y"], n_neighbors=kw["n_neighbors"], random_state=42, discrete_features=False)
                    else:
                        mi_arr = mutual_info_regression(X_batch, kw["y"], n_neighbors=kw["n_neighbors"], random_state=42, discrete_features=False)
                penalty = 0.0 if kw.get("direction_only") else _l2_penalty_value(coef_a, coef_b, kw["l2_penalty"], float(kw["l2_penalty_saturation"]))
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
            eval_pair_fn = _eval_dual
        else:
            eval_pair_fn = _eval_coef_pair

        eval_kwargs = dict(
            z_a=z_a_search, z_b=z_b_search,
            eval_func=eval_func,
            bf_callables=bf_callables_global, bf_names=bf_names_global,
            y=y_search_any, y_njit=y_search,
            mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins,
            n_neighbors=n_neighbors, discrete_target=discrete_target,
            l2_penalty=l2_penalty,
            l2_penalty_saturation=l2_penalty_saturation,
            # Precomputed basis matrices for BLAS GEMV fastpath (None when
            # factory-based basis or polynomial basis not in registry).
            B_a=B_a_search, B_b=B_b_search,
        )

        # Canonical warm-start: low-degree polynomial identities matching common targets (XOR, saddle, radial).
        # Replicate across both feature slots, then concatenate.
        warm_seeds = []
        # Per-operand ALS warm-start (data-fit, highest leverage). Fit the rank-1
        # separable model y ~ f(x_a) * g(x_b) in the basis via 3 alternating
        # lstsq solves and seed the joint optimiser with the resulting
        # coefficients. This lands the optimiser directly in the true
        # (potentially large-coefficient) basin -- the canonical unit-magnitude
        # seeds below never reach it, which is why the deceptive atan2/div
        # plateau trapped cma_batch on the F-POLY pre-distortion case. Gated by
        # ``warm_start_als``; requires a polynomial basis with a precomputed
        # basis matrix (factory bases / KSG-only paths skip it).
        if warm_start_als and B_a_search is not None and B_b_search is not None and ca_size <= B_a_search.shape[1] and cb_size <= B_b_search.shape[1]:
            try:
                als_a, als_b = warm_start_als_seed(
                    np.ascontiguousarray(B_a_search[:, :ca_size]),
                    np.ascontiguousarray(B_b_search[:, :cb_size]),
                    y_search_any,
                    # DEVICE-BORN design (2026-06-30, H2D collapse): the standardised
                    # columns + basis B_a_search/B_b_search were built from are in
                    # scope, so route the resident GPU branch through
                    # warm_start_als_seed_gpu_from_z -- it rebuilds the (degree+1)
                    # design ON DEVICE (max_degree = ca_size-1 / cb_size-1, the SAME
                    # leading columns the [:ca_size]/[:cb_size] slice selects from an
                    # orthogonal-poly basis) instead of uploading the prebuilt slices.
                    # The CPU path ignores z/basis and stays byte-identical.
                    z_a=z_a_search, z_b=z_b_search, basis=basis,
                )
            except Exception as _als_err:
                logger.debug("warm_start_als_seed failed at degree %d: %s", degree, _als_err)
                als_a = als_b = None
            if als_a is not None and als_b is not None:
                # The ALS direction is what matters (mul MI is scale-invariant);
                # rescale jointly so the largest coefficient lands inside
                # ``coef_range`` (CMA-ES / optuna suggest within these bounds, so
                # an un-clipped seed would be silently truncated and lose its
                # direction). The saturating penalty makes the absolute scale
                # harmless either way.
                _max_abs = float(max(np.max(np.abs(als_a)), np.max(np.abs(als_b)), 1e-12))
                _bound = 0.95 * min(abs(coef_range[0]), abs(coef_range[1]))
                if _bound > 0 and _max_abs > _bound:
                    _scale = _bound / _max_abs
                    als_a = als_a * _scale
                    als_b = als_b * _scale
                warm_seeds.append(np.concatenate([als_a, als_b]))
        if warm_start:
            if canonical_seeds_func is not None:
                # Non-polynomial basis ships its own canonical seeds via the registry.
                seeds_per_feature = canonical_seeds_func(degree)
            else:
                seeds_per_feature = _canonical_seeds(basis, degree)
            # Pair every seed with every other seed for c_b (limited to keep init pop small).
            for s_a in seeds_per_feature:
                for s_b in seeds_per_feature:
                    warm_seeds.append(np.concatenate([s_a, s_b]))
            # One symmetric pair (c_a = -c_b) captures antisymmetric targets like saddle.
            if seeds_per_feature:
                s = seeds_per_feature[0]
                warm_seeds.append(np.concatenate([s, -s]))

        # CROSS-FIT RECIPE WARM-START PRIOR (backlog idea #20), default OFF.
        # When a prior fit on an X-fingerprint-overlapping fold survived
        # admission with a polynomial pair recipe, its joint coefficient
        # vector(s) can be threaded in here as EXTRA warm-start seeds (the
        # per-parameter ``cross_fit_prior_seeds``). These only widen the
        # optimiser's INITIAL population / x0; the search then runs the SAME
        # generations and the winner is re-scored on THIS fold's data, so
        # admission stays gate-bound. A prior seed whose halves do not match the
        # current (ca_size, cb_size) for this degree is silently skipped.
        #
        # bench-attempt-rejected (2026-06-10, profiling/bench_warmstart_probe.py):
        # NO measurable win on 5/12 bootstrap folds (85% overlap, n=4000, non-
        # monotone-inner product target -- the regime where CMA must actually
        # search). Median iters COLD=78 / WARM=79 (the prior seed adds ONE eval
        # and saves ZERO generations); wall -0.3%..-3.7% (slightly SLOWER).
        # ROOT CAUSE: the per-pair ALS warm-start (``warm_start_als``, the block
        # above) already re-derives the true basin from THIS fold's data each
        # call and lands x0 there, so a cross-fold coefficient prior is strictly
        # subsumed. WORSE, the extra seed perturbs the CMA population enough to
        # land on a DIFFERENT optimum on 8/12 folds (selection NOT byte-identical
        # -- 6 higher-MI, 2 lower), which fails idea #20's identical-or-stabler
        # ship gate. Kept OFF by default (``None`` => this block is a no-op and
        # the warm-start population is byte-identical to legacy) per keep-all-
        # versions; do NOT flip default-on without a regime that ALS cannot seed.
        if cross_fit_prior_seeds:
            for _ps in cross_fit_prior_seeds:
                _ps = np.asarray(_ps, dtype=np.float64).reshape(-1)
                if _ps.size == ca_size + cb_size:
                    # Clip into the optimiser's bounds so the seed is not silently
                    # truncated (mirrors the ALS-seed rescale rationale above).
                    _bound = 0.999 * min(abs(coef_range[0]), abs(coef_range[1]))
                    if _bound > 0:
                        _m = float(np.max(np.abs(_ps)))
                        if _m > _bound:
                            _ps = _ps * (_bound / _m)
                    warm_seeds.append(_ps)

        coef_a_best = None
        coef_b_best = None
        bf_idx_best = -1
        raw_mi_best = -np.inf

        if optimizer in ("cma", "cma_batch", "random_batch", "numba_kernel"):
            # 2026-05-20 NEW-D: translate the Optuna-trial-based
            # ``early_stop_no_improve`` knob into a CMA-generation count.
            _early_stop_gens = None
            if early_stop_no_improve and early_stop_no_improve < n_trials:
                _eff_popsize = max(8, min(20, n_trials // 8))
                _early_stop_gens = max(
                    2, int(early_stop_no_improve) // _eff_popsize + 1,
                )
            cma_result = None
            try:
                if optimizer == "cma":
                    cma_result = _run_cma_search(
                        ca_size=ca_size, cb_size=cb_size,
                        coef_range=coef_range, n_trials=n_trials, seed=seed,
                        direction_only=direction_only,
                        warm_start_seeds=warm_seeds,
                        eval_kwargs=eval_kwargs,
                        eval_pair_fn=eval_pair_fn,
                        early_stop_no_improve_gens=_early_stop_gens,
                    )
                elif optimizer == "cma_batch":
                    # 2026-05-22: CMA-ES with batch eval -- collects popsize
                    # candidates per generation and runs ONE batched MI call
                    # over all (cand, bf) columns. Removes the per-solution
                    # Python GIL dance the plain CMA path paid. Does NOT
                    # take eval_pair_fn (multi-fidelity is incompatible with
                    # the batch eval signature today); falls back to
                    # _eval_coef_pair_batch directly.
                    from ._hermite_fe_optimise import _run_cma_search_batch
                    cma_result = _run_cma_search_batch(
                        ca_size=ca_size, cb_size=cb_size,
                        coef_range=coef_range, n_trials=n_trials, seed=seed,
                        direction_only=direction_only,
                        warm_start_seeds=warm_seeds,
                        eval_kwargs=eval_kwargs,
                        early_stop_no_improve_gens=_early_stop_gens,
                    )
                elif optimizer == "random_batch":
                    # 2026-05-22: pure batch random search + elitism. No
                    # Optuna, no CMA dependency. One MI batch call per iter.
                    from ._hermite_fe_optimise import _run_random_batch_search
                    cma_result = _run_random_batch_search(
                        ca_size=ca_size, cb_size=cb_size,
                        coef_range=coef_range, n_trials=n_trials, seed=seed,
                        direction_only=direction_only,
                        warm_start_seeds=warm_seeds,
                        eval_kwargs=eval_kwargs,
                    )
                else:  # numba_kernel
                    # 2026-05-22: all-numba single-pair entry point. Zero
                    # joblib / Optuna / cma deps -- one @njit(parallel=True)
                    # kernel inlines polyeval / bf dispatch / plugin MI.
                    # Limitations vs other optimizers: requires plugin MI
                    # (no KSG), polynomial basis only (no RBF/Sigmoid factory
                    # bases), no eval_pair_fn closures (multi_fidelity is
                    # disabled inside the kernel).
                    from ._numba_polynom_optimizer import run_numba_kernel_search
                    cma_result = run_numba_kernel_search(
                        ca_size=ca_size, cb_size=cb_size,
                        coef_range=coef_range, n_trials=n_trials, seed=seed,
                        direction_only=direction_only,
                        warm_start_seeds=warm_seeds,
                        eval_kwargs=eval_kwargs,
                    )
            except Exception as e:
                logger.warning("%s failed at degree %d (%s); " "falling back to Optuna", optimizer, degree, e)
                cma_result = None
            if cma_result is None:
                continue
            coef_a_best, coef_b_best, bf_idx_best, raw_mi_best, _ = cma_result
        else:  # optuna

            def _optuna_obj(trial, _degree=degree, _ca_size=ca_size, _cb_size=cb_size, _eval_pair_fn=eval_pair_fn, _eval_kwargs=eval_kwargs):
                """Optuna trial objective: sample a (coef_a, coef_b) pair, score it via the eval function, and stash bf_idx/raw_mi as trial user attrs for post-hoc inspection."""
                coef_a = np.array([trial.suggest_float(f"a_{i}", *coef_range) for i in range(_ca_size)], dtype=np.float64)
                coef_b = np.array([trial.suggest_float(f"b_{i}", *coef_range) for i in range(_cb_size)], dtype=np.float64)
                score, raw_mi, bf_idx = (_eval_pair_fn or _eval_coef_pair)(
                    coef_a, coef_b, direction_only=direction_only,
                    **_eval_kwargs,
                )
                if bf_idx >= 0:
                    trial.set_user_attr("bf_idx", bf_idx)
                    trial.set_user_attr("raw_mi", raw_mi)
                return score
            sampler = TPESampler(multivariate=True, seed=seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            # Inject canonical warm-start seeds as enqueued trials.
            if warm_seeds:
                for ws in warm_seeds[: min(8, len(warm_seeds))]:
                    params = {f"a_{i}": float(ws[i]) for i in range(ca_size)}
                    params.update({f"b_{i}": float(ws[ca_size + i]) for i in range(cb_size)})
                    try:
                        study.enqueue_trial(params)
                    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                        logger.debug("suppressed in _hermite_fe_optimise_pair.py:554: %s", e)
                        pass
            if early_stop_no_improve and early_stop_no_improve < n_trials:
                stop_state = {"best": -np.inf, "since_improve": 0}
                def _early_stop_cb(s, trial, _stop_state=stop_state):
                    """Optuna study callback: stop the study once ``early_stop_no_improve`` consecutive trials fail to beat the running best."""
                    cur_best = s.best_value if s.best_trial is not None else -np.inf
                    if cur_best > _stop_state["best"]:
                        _stop_state["best"] = cur_best
                        _stop_state["since_improve"] = 0
                    else:
                        _stop_state["since_improve"] += 1
                    if _stop_state["since_improve"] >= early_stop_no_improve:
                        s.stop()

                study.optimize(_optuna_obj, n_trials=n_trials, callbacks=[_early_stop_cb], show_progress_bar=False)
            else:
                study.optimize(_optuna_obj, n_trials=n_trials, show_progress_bar=False)
            try:
                bf_idx_best = study.best_trial.user_attrs.get("bf_idx", -1)
                raw_mi_best = study.best_trial.user_attrs.get("raw_mi", -np.inf)
                coef_a_best = np.array([study.best_params[f"a_{i}"] for i in range(ca_size)], dtype=np.float64)
                coef_b_best = np.array([study.best_params[f"b_{i}"] for i in range(cb_size)], dtype=np.float64)
            except (ValueError, KeyError):
                continue

        if coef_a_best is None or bf_idx_best < 0 or raw_mi_best <= 0 or not np.isfinite(raw_mi_best):
            continue

        # Multi-fidelity refinement: re-evaluate the best coef set on the FULL data for an honest gating MI.
        if multi_fidelity and n_full >= 4000:
            full_kwargs = dict(eval_kwargs)
            full_kwargs.update(z_a=z_a, z_b=z_b, y=y, y_njit=y_njit)
            # CRITICAL: B_a / B_b were precomputed on the 1500-element
            # SUBSAMPLE (z_a_search / z_b_search). Refinement runs on the
            # FULL z_a / z_b (typically 100k-1M elements). We MUST drop
            # the basis matrices here so _eval_coef_pair falls back to
            # the Horner eval_func path on full data. Without this drop,
            # h_a from ``B[:, :len(c)] @ c`` would be 1500-sized while
            # other code expects the full n - produces shape-mismatch
            # OR (silently worse) re-uses subsample-sized h_a but
            # subsample-sized MI -> CMA-ES misjudges which coef is best.
            # Discovered 2026-05-18 via in-flight VERIFY assertion.
            full_kwargs["B_a"] = None
            full_kwargs["B_b"] = None
            _, raw_mi_full, bf_idx_full = _eval_coef_pair(
                coef_a_best, coef_b_best, direction_only=direction_only,
                **full_kwargs,
            )
            if bf_idx_full >= 0 and raw_mi_full > 0:
                raw_mi_best = raw_mi_full
                bf_idx_best = bf_idx_full

        bf_name = bf_names_global[bf_idx_best]
        cand = HermiteResult(
            coef_a=coef_a_best, coef_b=coef_b_best,
            bin_func_name=bf_name, bin_func=bin_funcs[bf_name],  # type: ignore[arg-type]  # bin_funcs values are the callables per bf_name; _DEFAULT_BIN_FUNCS' inferred dict-value type is imprecise
            mi=raw_mi_best, baseline_mi=baseline,
            uplift=raw_mi_best / max(baseline, 1e-12),
            degree_a=degree, degree_b=degree,
            basis=basis,
            preprocess_a=preprocess_a,
            preprocess_b=preprocess_b,
        )
        if best is None or cand.mi > best.mi:
            best = cand
        logger.debug(
            "degree=%s: best MI=%.4f (baseline %.4f, uplift %.2fx), bf=%s",
            degree, raw_mi_best, baseline, cand.uplift, bf_name,
        )

    if best is None or best.mi <= baseline * baseline_uplift_threshold:
        # Failed to beat baseline by enough -- don't recommend an engineered feature.
        return None

    # Permutation-null noise floor. The high-capacity optimiser can overfit the plug-in MI estimator's binning-bias floor on a
    # target independent of (x_a, x_b): the winning engineered MI then sits just above the trivial baseline and clears the uplift
    # gate, fabricating a spurious feature. Re-evaluate the winning column's MI against shuffles of y (which destroy any real
    # dependence but preserve the binning bias) and reject when the real MI does not clear the null p95 by ``noise_floor_perm_ratio``.
    if noise_floor_perm_ratio > 0.0 and noise_floor_n_perms > 0 and mi_estimator == "plugin":
        try:
            from .hermite_fe import _plugin_mi_classif_njit, _plugin_mi_regression_njit
            # Run the noise-floor null on a STRIDED subsample of the operands (cap 30k). The permutation p95 is a COARSE
            # floor (compared against a 1.5x ratio), well-estimated on ~30k, while mi_real + the 50 shuffles on the FULL
            # n were the dominant per-pair cost at large n (measured: per-pair 12.5s@100k -> 68s@1M, ~all of it here) --
            # and the SEARCH itself already runs on a 1500-row multi-fidelity draw, so the full-n null was inconsistent
            # with the fit anyway. mi_real + the null share the SAME subsample so the reject comparison stays consistent;
            # strided preserves the outlier proportion the plug-in null floor depends on. Env-tunable.
            _NF_MAX = int(os.environ.get("MLFRAME_FE_NOISE_FLOOR_MAX_ROWS", "30000") or 0)
            if _NF_MAX > 0 and x_a.shape[0] > _NF_MAX:
                _nf_st = x_a.shape[0] // _NF_MAX
                _xa_nf = np.ascontiguousarray(x_a[::_nf_st]); _xb_nf = np.ascontiguousarray(x_b[::_nf_st])
                _y_nf = y[::_nf_st]
            else:
                _xa_nf, _xb_nf, _y_nf = x_a, x_b, y
            comb = np.ascontiguousarray(best.transform(_xa_nf, _xb_nf), dtype=np.float64).reshape(-1)
            if np.all(np.isfinite(comb)) and float(np.std(comb)) > 1e-12:
                if discrete_target:
                    y_perm_src = np.asarray(_y_nf, dtype=np.int64)
                    mi_real = float(_plugin_mi_classif_njit(comb, y_perm_src, plugin_n_bins))
                    mi_fn = _plugin_mi_classif_njit
                else:
                    y_perm_src = np.asarray(_y_nf, dtype=np.float64)
                    mi_real = float(_plugin_mi_regression_njit(comb, y_perm_src, plugin_n_bins))
                    mi_fn = _plugin_mi_regression_njit
                rng_null = np.random.default_rng(seed if seed and seed > 0 else 0)
                nlen = comb.shape[0]
                null_mis = np.empty(int(noise_floor_n_perms), dtype=np.float64)
                if discrete_target:
                    # ``comb`` is FIXED across the shuffles, so its quantile binning (the argsort -- ~3/4 of a plugin-MI
                    # call per the from-binned kernel's own bench) is identical every permutation: bin ONCE and reuse.
                    # Bit-identical to ``mi_fn(comb, yp)`` because ``_plugin_mi_from_binned_njit(_quantile_bin_njit(comb), y)``
                    # is byte-for-byte ``_plugin_mi_classif_njit(comb, y)`` (same histogram + plug-in MI, only the binning is hoisted).
                    from .hermite_fe import _plugin_mi_from_binned_njit, _quantile_bin_njit as _qbin
                    _comb_binned = _qbin(comb, plugin_n_bins)
                    for _p in range(int(noise_floor_n_perms)):
                        yp = np.ascontiguousarray(y_perm_src[rng_null.permutation(nlen)])
                        null_mis[_p] = float(_plugin_mi_from_binned_njit(_comb_binned, yp, plugin_n_bins))
                else:
                    for _p in range(int(noise_floor_n_perms)):
                        yp = np.ascontiguousarray(y_perm_src[rng_null.permutation(nlen)])
                        null_mis[_p] = float(mi_fn(comb, yp, plugin_n_bins))
                null_p95 = float(np.quantile(null_mis, 0.95))
                if mi_real < null_p95 * noise_floor_perm_ratio:
                    logger.debug(
                        "noise-floor reject: engineered MI %.4f < null p95 %.4f * %.2f (%s on independent target)",
                        mi_real, null_p95, noise_floor_perm_ratio, best.bin_func_name,
                    )
                    return None
        except Exception as _nf_err:
            logger.debug("noise-floor permutation guard skipped: %s", _nf_err)

    return best
