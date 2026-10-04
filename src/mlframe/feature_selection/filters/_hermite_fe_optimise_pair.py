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
from mlframe.utils.log_throttle import log_throttle
from .fe_baselines import ksg_random_state
from types import SimpleNamespace as _SimpleNamespace

logger = logging.getLogger("mlframe.feature_selection.filters.hermite_fe")


def precompute_hermite_pair_basis(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    *,
    discrete_target: bool = True,
    basis: str = "chebyshev",
    mi_estimator: str = "plugin",
    plugin_n_bins: int = 20,
    n_neighbors: int | None = None,
) -> tuple:
    """Precompute the basis fit + identity baseline that ``optimise_hermite_pair`` would otherwise redo
    byte-for-byte on every ``fe_smart_polynom_iters`` restart of the SAME ``(x_a, x_b, y)`` (only ``seed``
    differs across restarts - mirrors the existing ``precomputed_trivial_baseline`` plumbing, which already
    hoists the trivial-baseline recompute out of that same restart loop).

    Call ONCE per pair before the restart loop and thread the five return values into every
    ``optimise_hermite_pair`` call via ``precomputed_z_a`` / ``precomputed_preprocess_a`` / ``precomputed_z_b``
    / ``precomputed_preprocess_b`` / ``precomputed_identity_baseline``.

    Returns ``(z_a, preprocess_a, z_b, preprocess_b, identity_baseline)``.
    """
    from mlframe.feature_selection.filters.hermite_fe.shared import POLY_BASES as _POLY_BASES

    from ._hermite_fe_optimise import _baseline_mi_pair
    if basis not in _POLY_BASES:
        raise ValueError(f"unknown basis {basis!r}; expected one of {list(_POLY_BASES)}")
    basis_info = _POLY_BASES[basis]
    n = len(y)
    if n_neighbors is None:
        if n >= 5000:
            n_neighbors = 3
        elif n >= 1000:
            n_neighbors = 5
        else:
            n_neighbors = 7
    z_a, preprocess_a = basis_info["fit"](x_a)
    z_b, preprocess_b = basis_info["fit"](x_b)
    z_a = np.ascontiguousarray(z_a, dtype=np.float64)
    z_b = np.ascontiguousarray(z_b, dtype=np.float64)
    identity_baseline = _baseline_mi_pair(
        z_a, z_b, y, discrete_target=discrete_target, n_neighbors=n_neighbors,
        mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins,
    )
    return z_a, preprocess_a, z_b, preprocess_b, identity_baseline


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
    # PRECOMPUTED BASIS FIT + IDENTITY BASELINE: mirrors precomputed_trivial_baseline above - a caller running
    # multiple fe_smart_polynom_iters restarts on the SAME (x_a, x_b, y) (only ``seed`` differs) can precompute
    # the basis fit (z_a/preprocess_a/z_b/preprocess_b via basis_info["fit"]) and the identity baseline
    # (_baseline_mi_pair) ONCE and pass them here to short-circuit the byte-identical per-restart recompute.
    # All four of precomputed_z_a/precomputed_preprocess_a/precomputed_z_b/precomputed_preprocess_b must be
    # supplied together to take effect; None (default) preserves the legacy per-call fit.
    precomputed_z_a: np.ndarray | None = None,
    precomputed_preprocess_a: Any | None = None,
    precomputed_z_b: np.ndarray | None = None,
    precomputed_preprocess_b: Any | None = None,
    precomputed_identity_baseline: float | None = None,
    noise_floor_perm_ratio: float = 1.50,
    noise_floor_n_perms: int = 50,
) -> HermiteResult | None:
    """Find polynomial coefficients c_a, c_b that maximise MI(bin_func(P(x_a, c_a), P(x_b, c_b)), y) over the requested
    Optuna/CMA budget. Standardises inputs, regularises coefficients, and only returns a result when the engineered MI
    strictly beats the identity baseline by baseline_uplift_threshold.

    Knob tuning notes
    -----------------
    * basis="chebyshev" (default) wins empirically across 12 regimes (synthetic + UCI California Housing + UCI Diabetes +
      bounded / heavy-tailed) - never finishes last, highest minimum MI. Pass basis="hermite" for synthetic Gaussian inputs
      or basis="laguerre" for skewed-positive. See _benchmarks/bench_polynomial_bases.py.
    * l2_penalty=0.05 weights a SCALE-SATURATING coefficient penalty (see ``hermite_fe._l2_penalty_value``): it rises toward a constant
      ``l2_penalty`` ceiling as ``||c||^2`` grows instead of growing without bound, so high-MI / high-coefficient solutions (e.g. a separable
      Chebyshev reconstruction of a non-monotone product, ``||c||^2`` ~ 86) are not crushed while pure-noise small-||c|| candidates still pay
      ~full ``l2_penalty``. ``l2_penalty_saturation`` (default ``hermite_fe._L2_PENALTY_SATURATION_DEFAULT`` = 1.0) sets the ||c||^2 scale at
      which the penalty reaches half its ceiling; pass ``l2_penalty_saturation<=0`` for the legacy raw ``l2_penalty * ||c||^2`` behaviour.
    * warm_start_als=True (default) seeds the optimiser with a per-operand rank-1 ALS fit of ``y ~ f(x_a)*g(x_b)`` in the basis (see
      ``hermite_fe.warm_start_als_seed``). This lands the search directly in the true (possibly large-coefficient) basin - without it cma_batch
      can be trapped on a deceptive atan2/div plateau for non-monotone inner distortions. Polynomial bases only; factory/KSG paths skip it.
    * n_neighbors (KSG): None auto-picks 3 for n>=5000, 5 for n in [1000,5000), 7 for n<1000.
    * max_degree=4 covers most smooth targets. For high-frequency targets raise to 6-8 (n_trials proportionally).
    * early_stop_no_improve: stop a study early if no improvement in the last N trials.
    * mi_estimator="plugin" (default) uses an njit plug-in estimator on quantile-binned values - ~50-100x faster than
      sklearn's KSG, rank-equivalent for optimization (constant entropy bias). Pass "ksg" for sklearn's KSG.
    * plugin_n_bins=20 (default): ~sqrt(n) rule-of-thumb; larger bins reduce bias, raise variance.
    * noise_floor_perm_ratio=1.50 (default): a permutation-null guard against the high-capacity optimiser fabricating an
      engineered feature on a target INDEPENDENT of the inputs. The plug-in MI estimator has a binning-bias floor that the
      optimiser can overfit on pure noise - on a noise target the best engineered MI barely clears both the trivial baseline
      and ``baseline_uplift_threshold``, so the uplift gate alone passes it through. The permutation null measures that floor
      directly: re-evaluate the winning engineered column's MI against ``noise_floor_n_perms`` shuffles of y (destroys any real
      dependence, keeps the binning bias) and reject (return None) when ``mi_real < perm_null_p95 * noise_floor_perm_ratio``.
      A genuine feature beats its own permutation-null p95 by 40x+; a noise feature by only ~1.2x. Set ``noise_floor_perm_ratio<=0``
      (or ``noise_floor_n_perms<=0``) to disable. Measured separation in the MRMR discrete path (n=4000, 20 restarts x 3 noise
      pairs): pure-noise max ratio 1.235 (reject), F-POLY 40.8x / F-OSC 61.7x (pass) - 1.50 has comfortable margin both ways.

    Returns HermiteResult or None if the search failed to beat the baseline.
    """
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)
    from mlframe.feature_selection.filters.hermite_fe.shared import DEFAULT_BIN_FUNCS as _DEFAULT_BIN_FUNCS, L2_PENALTY_SATURATION_DEFAULT as _L2_PENALTY_SATURATION_DEFAULT, POLY_BASES as _POLY_BASES
    # Sister-sibling import: ``_baseline_mi_pair``, ``_eval_coef_pair``,
    # ``_run_cma_search`` stayed in ``_hermite_fe_optimise``. Sister-to-sister
    # is cycle-free because the parent imports each sibling at its bottom
    # without either sibling importing the other at module-top.
    if mi_estimator not in ("plugin", "ksg"):
        raise ValueError(f"unknown mi_estimator={mi_estimator!r}; expected 'plugin' or 'ksg'")
    if optimizer not in ("optuna", "cma", "cma_batch", "random_batch", "numba_kernel", "cupy_kernel"):
        raise ValueError(f"unknown optimizer={optimizer!r}; expected one of " f"'optuna', 'cma', 'cma_batch', 'random_batch', 'numba_kernel', 'cupy_kernel'")
    if l2_penalty_saturation is None:
        l2_penalty_saturation = _L2_PENALTY_SATURATION_DEFAULT
    # Auto-pick n_neighbors based on n.
    n = len(y)
    n_neighbors = _optimise_hermite_pai_step1_auto_pick_neighbors(n_neighbors, n)
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
    st.basis_info = _POLY_BASES[basis]

    # Preprocess inputs to the basis's natural domain. A caller running multiple fe_smart_polynom_iters
    # restarts on the SAME (x_a, x_b) can precompute this fit ONCE (see precomputed_z_a et al. above) -
    # basis_info["fit"] is a pure deterministic function of (x_a/x_b, basis), so reusing it across restarts
    # is byte-identical to refitting every time.
    if precomputed_z_a is not None and precomputed_preprocess_a is not None and precomputed_z_b is not None and precomputed_preprocess_b is not None:
        st.z_a, st.preprocess_a = precomputed_z_a, precomputed_preprocess_a
        st.z_b, st.preprocess_b = precomputed_z_b, precomputed_preprocess_b
    else:
        st.z_a, st.preprocess_a = st.basis_info["fit"](x_a)
        st.z_b, st.preprocess_b = st.basis_info["fit"](x_b)
    st.z_a = np.ascontiguousarray(st.z_a, dtype=np.float64)
    st.z_b = np.ascontiguousarray(st.z_b, dtype=np.float64)
    # Hoist size-aware dispatch out of the hot trial loop: pick the backend ONCE per call (n is fixed across trials).
    # Saves ~4us/call closure overhead, ~5ms over 1000+ trials.
    st.n_eval = st.z_a.shape[0]
    st.factory_top = st.basis_info.get("eval_njit_factory")
    _optimise_hermite_pai_step3_saves_us_call(st, basis, precomputed_identity_baseline, y, discrete_target, n_neighbors, mi_estimator, plugin_n_bins, precomputed_trivial_name, use_trivial_baseline, precomputed_trivial_baseline, x_a, x_b, sweep_degrees, min_degree, max_degree, bin_funcs, seed)
    eval_func_b = _optimise_hermite_pai_step4_st_bf_callables(st, bin_funcs, multi_fidelity, seed, y, basis, max_degree)

    _optimise_hermite_pai_step5_degree_st_degree(st, eval_func_b, mi_estimator, plugin_n_bins, n_neighbors, discrete_target, l2_penalty, l2_penalty_saturation, warm_start_als, basis, coef_range, warm_start, cross_fit_prior_seeds, optimizer, early_stop_no_improve, n_trials, seed, direction_only, TPESampler, optuna, multi_fidelity, y, bin_funcs)

    if st.best is None or st.best.mi <= st.baseline * baseline_uplift_threshold:
        # Failed to beat baseline by enough - don't recommend an engineered feature.
        return None

    # Permutation-null noise floor. The high-capacity optimiser can overfit the plug-in MI estimator's binning-bias floor on a
    # target independent of (x_a, x_b): the winning engineered MI then sits just above the trivial baseline and clears the uplift
    # gate, fabricating a spurious feature. Re-evaluate the winning column's MI against shuffles of y (which destroy any real
    # dependence but preserve the binning bias) and reject when the real MI does not clear the null p95 by ``noise_floor_perm_ratio``.
    if noise_floor_perm_ratio > 0.0 and noise_floor_n_perms > 0 and mi_estimator == "plugin":
        try:
            from mlframe.feature_selection.filters.hermite_fe.shared import plugin_mi_classif_njit as _plugin_mi_classif_njit, plugin_mi_regression_njit as _plugin_mi_regression_njit
            # Run the noise-floor null on a STRIDED subsample of the operands (cap 30k). The permutation p95 is a COARSE
            # floor (compared against a 1.5x ratio), well-estimated on ~30k, while mi_real + the 50 shuffles on the FULL
            # n were the dominant per-pair cost at large n (measured: per-pair 12.5s@100k -> 68s@1M, ~all of it here) -
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
            comb = np.ascontiguousarray(st.best.transform(_xa_nf, _xb_nf), dtype=np.float64).reshape(-1)
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
                    # ``comb`` is FIXED across the shuffles, so its quantile binning (the argsort - ~3/4 of a plugin-MI
                    # call per the from-binned kernel's own bench) is identical every permutation: bin ONCE and reuse.
                    # Bit-identical to ``mi_fn(comb, yp)`` because ``_plugin_mi_from_binned_njit(_quantile_bin_njit(comb), y)``
                    # is byte-for-byte ``_plugin_mi_classif_njit(comb, y)`` (same histogram + plug-in MI, only the binning is hoisted).
                    from mlframe.feature_selection.filters.hermite_fe.shared import plugin_mi_from_binned_njit as _plugin_mi_from_binned_njit, quantile_bin_njit as _qbin
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
                        mi_real, null_p95, noise_floor_perm_ratio, st.best.bin_func_name,
                    )
                    return None
        except Exception as _nf_err:
            logger.debug("noise-floor permutation guard skipped: %s", _nf_err)

    return st.best  # type: ignore[no-any-return]  # read from the untyped state namespace / cache the stage helpers share


def _optimise_hermite_pai_step1_auto_pick_neighbors(n_neighbors, n):
    """Step 1 of optimise_hermite_pair: lines starting at ``if n_neighbors is None:``."""
    if n_neighbors is None:
        if n >= 5000:
            n_neighbors = 3
        elif n >= 1000:
            n_neighbors = 5
        else:
            n_neighbors = 7
    return n_neighbors


def _optimise_hermite_pai_step3_saves_us_call(st, basis, precomputed_identity_baseline, y, discrete_target, n_neighbors, mi_estimator, plugin_n_bins, precomputed_trivial_name, use_trivial_baseline, precomputed_trivial_baseline, x_a, x_b, sweep_degrees, min_degree, max_degree, bin_funcs, seed):
    """Step 3 of optimise_hermite_pair: lines starting at ``if st.factory_top is not None:``."""
    from mlframe.feature_selection.filters.hermite_fe.shared import CUDA_AVAILABLE as _CUDA_AVAILABLE, CUDA_THRESHOLD as _CUDA_THRESHOLD, NJIT_FUNCS as _NJIT_FUNCS, NJIT_PAR_FUNCS as _NJIT_PAR_FUNCS, PAR_THRESHOLD as _PAR_THRESHOLD
    from mlframe.feature_selection.filters._hermite_fe_optimise import _baseline_mi_pair

    if st.factory_top is not None:
        # Non-polynomial basis with data-dependent eval (RBF/Sigmoid). Factory is invoked below per-feature.
        st.eval_func = None
    elif basis in _NJIT_FUNCS:
        # Polynomial basis - size-aware ladder applies.
        if st.n_eval < _PAR_THRESHOLD:
            st.eval_func = st.basis_info["eval_njit"]
        elif st.n_eval >= _CUDA_THRESHOLD and _CUDA_AVAILABLE:
            st.eval_func = st.basis_info["eval_dispatch"]  # cuda path
        else:
            st.eval_func = _NJIT_PAR_FUNCS[basis]
    else:
        # Other non-polynomial basis with simple eval_njit (Fourier, Pade).
        st.eval_func = st.basis_info["eval_njit"]

    if precomputed_identity_baseline is not None:
        st.baseline = float(precomputed_identity_baseline)
    else:
        st.baseline = _baseline_mi_pair(st.z_a, st.z_b, y, discrete_target=discrete_target, n_neighbors=n_neighbors, mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins, random_state=seed)
    logger.debug("baseline MI(pair, y) = %.4f", st.baseline)

    # Stronger gate than the identity max(MI(x_a, y), MI(x_b, y)): try trivial pair-feature transforms
    # (mul, ratio, sum_sq, atan2, ...) and use BEST trivial MI as baseline. Often a simple mul(x_a, x_b)
    # captures most of the signal a polynomial would (verified on XOR / circle / saddle / UCI).
    #
    # 2026-05-20 NEW-A: callers running multiple ``fe_smart_polynom_iters``
    # restarts per pair can pre-compute the trivial baseline once and feed
    # it in via ``precomputed_trivial_baseline`` (+ ``precomputed_trivial_name``);
    # this elides ~5x duplicated 50-150ms ``best_trivial_pair`` calls per
    # pair on the n=200k production config.
    st.trivial_baseline_name = precomputed_trivial_name
    if use_trivial_baseline and precomputed_trivial_baseline is None:
        try:
            from mlframe.feature_selection.filters.fe_baselines import best_trivial_pair
            trivial = best_trivial_pair(
                np.asarray(x_a, dtype=np.float64),
                np.asarray(x_b, dtype=np.float64), y,
                discrete_target=discrete_target,
                mi_estimator=mi_estimator,
                plugin_n_bins=plugin_n_bins,
                n_neighbors=n_neighbors, random_state=seed,
            )
            if trivial is not None:
                st.trivial_baseline_name, st._, trivial_mi = trivial
                if trivial_mi > st.baseline:
                    logger.debug(
                        "trivial baseline %r raises baseline from %.4f to %.4f",
                        st.trivial_baseline_name, st.baseline, trivial_mi,
                    )
                    st.baseline = trivial_mi
        except Exception as e:
            logger.debug("trivial baseline check failed: %s", e)
    elif precomputed_trivial_baseline is not None:
        # Caller supplied the precomputed value - use it directly.
        if precomputed_trivial_baseline > st.baseline:
            logger.debug(
                "trivial baseline %r raises baseline from %.4f to %.4f (precomputed)",
                precomputed_trivial_name, st.baseline, precomputed_trivial_baseline,
            )
            st.baseline = float(precomputed_trivial_baseline)

    # Pre-cast y once for the njit fast path.
    if mi_estimator == "plugin":
        st.y_njit = np.asarray(y, dtype=np.int64) if discrete_target else np.asarray(y, dtype=np.float64)
    else:
        st.y_njit = None  # KSG path does not need it

    st.best = None

    st.degree_grid = list(range(min_degree, max_degree + 1)) if sweep_degrees else [max_degree]

    st.bf_names_global = list(bin_funcs.keys())


def _optimise_hermite_pai_step4_st_bf_callables(st, bin_funcs, multi_fidelity, seed, y, basis, max_degree):
    """Step 4 of optimise_hermite_pair: lines starting at ``st.bf_callables_global = [bin_funcs[n] for n in st.bf_names_global]``."""
    from mlframe.feature_selection.filters.hermite_fe.shared import build_basis_matrix, BASIS_BUILDERS as _BASIS_BUILDERS

    st.bf_callables_global = [bin_funcs[n] for n in st.bf_names_global]

    # Multi-fidelity subsample ladder: for large n, fit coefficients on a small subsample (saves O(n) MI work)
    # and refine on full data at the end. With 2*(d+1) <= 8 coefficients, 1500 samples is enough to estimate stably.
    st.n_full = st.z_a.shape[0]
    if multi_fidelity and st.n_full >= 4000:
        rng_mf = np.random.default_rng(seed if seed > 0 else 0)
        sub_idx = rng_mf.choice(st.n_full, size=1500, replace=False)
        st.z_a_search = np.ascontiguousarray(st.z_a[sub_idx], dtype=np.float64)
        st.z_b_search = np.ascontiguousarray(st.z_b[sub_idx], dtype=np.float64)
        st.y_search = st.y_njit[sub_idx] if st.y_njit is not None else None
        st.y_search_any = y[sub_idx] if isinstance(y, np.ndarray) else np.asarray(y)[sub_idx]
    else:
        st.z_a_search = st.z_a
        st.z_b_search = st.z_b
        st.y_search = st.y_njit
        st.y_search_any = y

    # Coef-size lookup: polynomial bases use degree + 1; non-poly bases (Fourier 2K, RBF up to 9, Pade 2p+1) override.
    st.coef_size_func = st.basis_info.get("coef_size_func", lambda d: d + 1)
    st.canonical_seeds_func = st.basis_info.get("canonical_seeds_func")

    # Factory-based bases (RBF, Sigmoid) eval depends on train-fold-fitted centres / thresholds. Build per-basis
    # eval once preprocess params are known. Wave 69: separate eval for x_a and x_b already implemented
    # below - factory is called twice with preprocess_a vs preprocess_b, producing distinct eval kernels per side.
    st.factory = st.basis_info.get("eval_njit_factory")
    if st.factory is not None:
        st.eval_func = st.factory(st.preprocess_a)
        eval_func_b = st.factory(st.preprocess_b)
    else:
        eval_func_b = st.eval_func

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
    st.B_a_search = None
    st.B_b_search = None
    _multi_fidelity_active = bool(multi_fidelity and st.n_full >= 4000)
    if st.factory is None and basis in _BASIS_BUILDERS:
        try:
            st.B_a_search = build_basis_matrix(basis, st.z_a_search, max_degree)
            st.B_b_search = build_basis_matrix(basis, st.z_b_search, max_degree)
        except Exception as _bm_err:
            logger.debug("build_basis_matrix failed for %r: %s", basis, _bm_err)
            st.B_a_search = None
            st.B_b_search = None
    return eval_func_b


def _optimise_hermite_pai_step5_degree_st_degree(st, eval_func_b, mi_estimator, plugin_n_bins, n_neighbors, discrete_target, l2_penalty, l2_penalty_saturation, warm_start_als, basis, coef_range, warm_start, cross_fit_prior_seeds, optimizer, early_stop_no_improve, n_trials, seed, direction_only, TPESampler, optuna, multi_fidelity, y, bin_funcs):
    """Step 5 of optimise_hermite_pair: lines starting at ``for degree in st.degree_grid:``."""
    from mlframe.feature_selection.filters.hermite_fe.shared import HermiteResult
    from mlframe.feature_selection.filters._hermite_fe_optimise import _eval_coef_pair

    for degree in st.degree_grid:
        ca_size = st.coef_size_func(degree)
        cb_size = st.coef_size_func(degree)
        # Truncate the basis matrices to THIS degree's coefficient length ONCE here instead of per-trial
        # inside _eval_coef_pair / _eval_coef_pair_batch (CMA-ES / random-batch trials only vary coefficient
        # VALUES, never ca_size/cb_size, within one degree) - tens of thousands of redundant slice+copies/pair
        # collapse to one per degree.
        B_a_deg = np.ascontiguousarray(st.B_a_search[:, :ca_size]) if st.B_a_search is not None else None
        B_b_deg = np.ascontiguousarray(st.B_b_search[:, :cb_size]) if st.B_b_search is not None else None

        # Shared kwargs for both Optuna and CMA paths. When eval_func differs per feature (factory-based bases
        # like RBF), wrap _eval_coef_pair to use both eval_func and eval_func_b.
        eval_kwargs, warm_seeds = _optimise_hermite_pai_step1_like_rbf_wrap(st, eval_func_b, mi_estimator, plugin_n_bins, n_neighbors, discrete_target, l2_penalty, l2_penalty_saturation, B_a_deg, B_b_deg, seed)
        # Per-operand ALS warm-start (data-fit, highest leverage). Fit the rank-1
        # separable model y ~ f(x_a) * g(x_b) in the basis via 3 alternating
        # lstsq solves and seed the joint optimiser with the resulting
        # coefficients. This lands the optimiser directly in the true
        # (potentially large-coefficient) basin - the canonical unit-magnitude
        # seeds below never reach it, which is why the deceptive atan2/div
        # plateau trapped cma_batch on the F-POLY pre-distortion case. Gated by
        # ``warm_start_als``; requires a polynomial basis with a precomputed
        # basis matrix (factory bases / KSG-only paths skip it).
        bf_idx_best, coef_a_best, coef_b_best, raw_mi_best = _optimise_hermite_pai_step2_basis_matrix_factory(warm_start_als, st, ca_size, cb_size, B_a_deg, B_b_deg, basis, degree, coef_range, warm_seeds, warm_start, cross_fit_prior_seeds)

        if optimizer in ("cma", "cma_batch", "random_batch", "numba_kernel", "cupy_kernel"):
            # 2026-05-20 NEW-D: translate the Optuna-trial-based
            # ``early_stop_no_improve`` knob into a CMA-generation count.
            _early_stop_gens = None
            if early_stop_no_improve and early_stop_no_improve < n_trials:
                _eff_popsize = max(8, min(20, n_trials // 8))
                _early_stop_gens = max(
                    2, int(early_stop_no_improve) // _eff_popsize + 1,
                )
            cma_result = None
            cma_result = _optimise_hermite_pai_step1_try(optimizer, ca_size, cb_size, coef_range, n_trials, seed, direction_only, warm_seeds, eval_kwargs, _early_stop_gens, degree, cma_result, st.eval_pair_fn)
            if cma_result is None:
                continue
            coef_a_best, coef_b_best, bf_idx_best, raw_mi_best, st._ = cma_result
        else:  # optuna

            study = _optimise_hermite_pai_step2_def_optuna_obj(degree, ca_size, cb_size, eval_kwargs, coef_range, direction_only, TPESampler, seed, optuna, warm_seeds, early_stop_no_improve, n_trials, st.eval_pair_fn)
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
        if multi_fidelity and st.n_full >= 4000:
            full_kwargs = dict(eval_kwargs)
            full_kwargs.update(z_a=st.z_a, z_b=st.z_b, y=y, y_njit=st.y_njit)
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
            st._, raw_mi_full, bf_idx_full = _eval_coef_pair(
                coef_a_best, coef_b_best, direction_only=direction_only,
                **full_kwargs,
            )
            if bf_idx_full >= 0 and raw_mi_full > 0:
                raw_mi_best = raw_mi_full
                bf_idx_best = bf_idx_full

        bf_name = st.bf_names_global[bf_idx_best]
        cand = HermiteResult(
            coef_a=coef_a_best, coef_b=coef_b_best,
            bin_func_name=bf_name, bin_func=bin_funcs[bf_name],  # bin_funcs values are the callables per bf_name; _DEFAULT_BIN_FUNCS' inferred dict-value type is imprecise
            mi=raw_mi_best, baseline_mi=st.baseline,
            uplift=raw_mi_best / max(st.baseline, 1e-12),
            degree_a=degree, degree_b=degree,
            basis=basis,
            preprocess_a=st.preprocess_a,
            preprocess_b=st.preprocess_b,
        )
        if st.best is None or cand.mi > st.best.mi:
            st.best = cand
        logger.debug(
            "degree=%s: best MI=%.4f (baseline %.4f, uplift %.2fx), bf=%s",
            degree, raw_mi_best, st.baseline, cand.uplift, bf_name,
        )


def _optimise_hermite_pai_step1_try(optimizer, ca_size, cb_size, coef_range, n_trials, seed, direction_only, warm_seeds, eval_kwargs, _early_stop_gens, degree, cma_result, eval_pair_fn):
    """Step 1 of _optimise_hermite_pai_step5_degree_st_degree: lines starting at ``try:``."""
    from mlframe.feature_selection.filters._hermite_fe_optimise import _run_cma_search

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
            # CMA-ES with batch eval - collects popsize
            # candidates per generation and runs ONE batched MI call
            # over all (cand, bf) columns. Removes the per-solution
            # Python GIL dance the plain CMA path paid. Does NOT
            # take eval_pair_fn (multi-fidelity is incompatible with
            # the batch eval signature today); falls back to
            # _eval_coef_pair_batch directly.
            from mlframe.feature_selection.filters._hermite_fe_optimise import _run_cma_search_batch
            cma_result = _run_cma_search_batch(
                ca_size=ca_size, cb_size=cb_size,
                coef_range=coef_range, n_trials=n_trials, seed=seed,
                direction_only=direction_only,
                warm_start_seeds=warm_seeds,
                eval_kwargs=eval_kwargs,
                early_stop_no_improve_gens=_early_stop_gens,
            )
        elif optimizer == "random_batch":
            # Pure batch random search + elitism. No
            # Optuna, no CMA dependency. One MI batch call per iter.
            from mlframe.feature_selection.filters._hermite_fe_optimise import _run_random_batch_search
            cma_result = _run_random_batch_search(
                ca_size=ca_size, cb_size=cb_size,
                coef_range=coef_range, n_trials=n_trials, seed=seed,
                direction_only=direction_only,
                warm_start_seeds=warm_seeds,
                eval_kwargs=eval_kwargs,
            )
        elif optimizer == "cupy_kernel":
            # GPU generation-batched twin of numba_kernel: one cuBLAS GEMM per generation for all
            # candidates' basis evaluation + batched device binning/MI. Same plugin-MI/polynomial-
            # basis limitations as numba_kernel. cupy_kernel is the DEFAULT since 2026-07-15
            # (production wellbore-100k validation: fit wall 854s -> 480s -> 538s under load,
            # selection identical). On a cupy-less host, fall back to random_batch - NOT cma_batch
            # - per the time-to-first-best comparison below (random_batch matches cma_batch's
            # bit-identical winners at equal-or-lower wall on every measured case; instead of
            # letting the generic except route to the much slower/worse-converging optuna path).
            #
            # OPTIMIZER COMPARISON (2026-07-15, protocol: 3 seeds x 3 restarts x 60 trials, plugin
            # MI, chebyshev basis, n=20000; "time to first best" = wall-clock offset at which the
            # EVENTUAL best MI was first reached, median across seeds; see
            # _benchmarks/bench_polynom_optimizer_bases_ab.py + results/bench_polynom_optimizer_bases.json):
            #
            #   variant       | cubic_inner MI / t-to-best | cross_cheb MI / t-to-best | quintic_mix MI / t-to-best
            #   optuna        | 0.4813 / ~4.2s   (restart0)| 0.3078 / ~4.0s   (restart0)| 0.14-0.15 / ~13s  (restart2, EVERY seed)
            #   cma_batch     | 0.4813 / ~1.6s   (restart0)| 0.3078 / ~1.3s   (restart0)| 0.15-0.17 / 1.3-4.2s (restart0-2)
            #   random_batch  | 0.4813 / ~1.1s   (restart0)| 0.3078 / ~1.2s   (restart0)| 0.16      / 1.1-3.3s (restart0-2)
            #   numba_kernel  | 0.4813 / ~3.4s   (restart0)| 0.3078 / ~3.4s   (restart0)| 0.14-0.15 / 6.8-11.0s(restart1-2)
            #   cupy_kernel   | 0.4813 / 0.3-1.2s(restart0)| 0.3078 / ~0.3s   (restart0)| 0.14-0.15 / 0.3-0.85s(restart0-2)
            #
            # optuna is 3-4x slower than cma_batch/random_batch at IDENTICAL quality on easy cases,
            # and on the hard case (quintic_mix) additionally needs its FULL budget (restart 2) on
            # every seed while landing WORSE MI (0.135-0.152) than cma_batch/random_batch (0.15-0.17)
            # - TPE's model-based sampling buys nothing here, only orchestration overhead.
            # numba_kernel is architecturally mismatched for single-pair search: its
            # @njit(parallel=True) outer kernel prange's over PAIRS, so a single-pair caller (every
            # caller today) pays full thread-pool dispatch/barrier overhead for a 1-iteration loop,
            # AND its per-candidate basis eval is a scalar Horner recompute (_polyeval_dispatch_njit
            # called per candidate) instead of the batched BLAS GEMM (B @ C.T) cma_batch/random_batch
            # use via _eval_coef_pair_batch - fixing this would duplicate random_batch's already-
            # faster BLAS path inside njit for no measured benefit, so numba_kernel is NOT
            # recommended for single-pair search and stays available only as an explicit opt-in.
            # This dispatch used to route to cupy any
            # time a bare `import cupy` succeeded, never checking gpu_globally_disabled() - a plain
            # MRMR() on a cupy-capable host launched real CUDA kernels for every hermite/orth
            # pair-FE search even under MLFRAME_DISABLE_GPU=1/CUDA_VISIBLE_DEVICES="", contradicting
            # the documented opt-out convention already honored by _fe_pure_form_retention_gpu_
            # resident.py and the STRICT-resident dispatch chain in the same cluster. (The estimator's
            # own use_gpu=False constructor default is not yet threaded through eval_kwargs to this
            # call site - that requires plumbing run_polynom_pair_fe's signature too and is tracked
            # as a separate follow-up; the env-var opt-out below is the safety-critical, universally-
            # applicable half of this fix and closes the actual reported bug.)
            from mlframe.feature_selection.filters._gpu_policy import gpu_globally_disabled

            try:
                if gpu_globally_disabled():
                    raise RuntimeError("GPU disabled (MLFRAME_DISABLE_GPU/CUDA_VISIBLE_DEVICES)")
                import cupy  # noqa: F401
                from mlframe.feature_selection.filters._cupy_polynom_optimizer import run_cupy_kernel_search
                _kernel_search = run_cupy_kernel_search
            except Exception:
                logger.debug("cupy unavailable or GPU disabled; cupy_kernel default falls back to random_batch")
                from mlframe.feature_selection.filters._hermite_fe_optimise import _run_random_batch_search
                _kernel_search = lambda **kw: _run_random_batch_search(**kw)  # noqa: E731
            cma_result = _kernel_search(
                ca_size=ca_size, cb_size=cb_size,
                coef_range=coef_range, n_trials=n_trials, seed=seed,
                direction_only=direction_only,
                warm_start_seeds=warm_seeds,
                eval_kwargs=eval_kwargs,
            )
        else:  # numba_kernel
            # all-numba single-pair entry point. Zero
            # joblib / Optuna / cma deps - one @njit(parallel=True)
            # kernel inlines polyeval / bf dispatch / plugin MI.
            # Limitations vs other optimizers: requires plugin MI
            # (no KSG), polynomial basis only (no RBF/Sigmoid factory
            # bases), no eval_pair_fn closures (multi_fidelity is
            # disabled inside the kernel).
            from mlframe.feature_selection.filters._numba_polynom_optimizer import run_numba_kernel_search
            cma_result = run_numba_kernel_search(
                ca_size=ca_size, cb_size=cb_size,
                coef_range=coef_range, n_trials=n_trials, seed=seed,
                direction_only=direction_only,
                warm_start_seeds=warm_seeds,
                eval_kwargs=eval_kwargs,
            )
    except Exception as e:
        log_throttle(logger, "hermite_fe_optimizer_failed_falling_back_optuna", logging.WARNING, "%s failed at degree %d (%s); " "falling back to Optuna", optimizer, degree, e)
        cma_result = None
    return cma_result


def _optimise_hermite_pai_step2_def_optuna_obj(degree, ca_size, cb_size, eval_kwargs, coef_range, direction_only, TPESampler, seed, optuna, warm_seeds, early_stop_no_improve, n_trials, eval_pair_fn):
    """Step 2 of _optimise_hermite_pai_step5_degree_st_degree: lines starting at ``def _optuna_obj(trial, _degree=degree, _ca_size=ca_size, _cb_size=cb_s``."""
    from mlframe.feature_selection.filters._hermite_fe_optimise import _eval_coef_pair

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
                logger.debug("suppressed: %s", e)
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
    return study


def _optimise_hermite_pai_step1_like_rbf_wrap(st, eval_func_b, mi_estimator, plugin_n_bins, n_neighbors, discrete_target, l2_penalty, l2_penalty_saturation, B_a_deg, B_b_deg, seed):
    """Step 1 of _optimise_hermite_pai_step5_degree_st_degree: lines starting at ``if st.factory is not None:``."""
    from mlframe.feature_selection.filters.hermite_fe.shared import l2_normalize_pair as _l2_normalize_pair, l2_penalty_value as _l2_penalty_value, plugin_mi_classif_batch_njit as _plugin_mi_classif_batch_njit, plugin_mi_regression_batch_njit as _plugin_mi_regression_batch_njit
    from mlframe.feature_selection.filters._hermite_fe_optimise import _eval_coef_pair

    if st.factory is not None:
        def _eval_dual(coef_a, coef_b, **kw):
            """Evaluate a candidate (coef_a, coef_b) pair against a factory-based basis (each operand using its own eval_func/eval_func_b); returns -inf score on any non-finite basis output."""
            from numpy import column_stack, ascontiguousarray, all as npall, isfinite
            z_a_loc = kw["z_a"]
            z_b_loc = kw["z_b"]
            bf_call = kw["bf_callables"]
            if kw.get("direction_only"):
                coef_a, coef_b = _l2_normalize_pair(coef_a, coef_b, 1.0)
            h_a = st.eval_func(z_a_loc, coef_a)
            h_b = eval_func_b(z_b_loc, coef_b)
            if not (npall(isfinite(h_a)) and npall(isfinite(h_b))):
                return -np.inf, 0.0, -1
            cols = []
            valid_idx = []
            for k, bf in enumerate(bf_call):
                try:
                    combined = bf(h_a, h_b)
                except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed: %s", e)
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
                    mi_arr = mutual_info_classif(X_batch, kw["y"], n_neighbors=kw["n_neighbors"], random_state=ksg_random_state(kw["random_state"]), discrete_features=False)
                else:
                    mi_arr = mutual_info_regression(X_batch, kw["y"], n_neighbors=kw["n_neighbors"], random_state=ksg_random_state(kw["random_state"]), discrete_features=False)
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
        st.eval_pair_fn = _eval_dual
    else:
        st.eval_pair_fn = _eval_coef_pair

    eval_kwargs = dict(
        z_a=st.z_a_search, z_b=st.z_b_search,
        eval_func=st.eval_func,
        bf_callables=st.bf_callables_global, bf_names=st.bf_names_global,
        y=st.y_search_any, y_njit=st.y_search,
        mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins,
        n_neighbors=n_neighbors, discrete_target=discrete_target,
        l2_penalty=l2_penalty, random_state=seed,
        l2_penalty_saturation=l2_penalty_saturation,
        # Precomputed basis matrices for BLAS GEMV fastpath, PRE-TRUNCATED to (ca_size, cb_size) above
        # (None when factory-based basis or polynomial basis not in registry). _eval_coef_pair(_batch) no
        # longer re-slices these per trial - they must already match coef_a.shape[0] / coef_b.shape[0].
        B_a=B_a_deg, B_b=B_b_deg,
    )

    # Canonical warm-start: low-degree polynomial identities matching common targets (XOR, saddle, radial).
    # Replicate across both feature slots, then concatenate.
    warm_seeds: list[Any] = []
    return eval_kwargs, warm_seeds


def _optimise_hermite_pai_step2_basis_matrix_factory(warm_start_als, st, ca_size, cb_size, B_a_deg, B_b_deg, basis, degree, coef_range, warm_seeds, warm_start, cross_fit_prior_seeds):
    """Step 2 of _optimise_hermite_pai_step5_degree_st_degree: lines starting at ``if warm_start_als and st.B_a_search is not None and st.B_b_search is n``."""
    from mlframe.feature_selection.filters.hermite_fe.shared import warm_start_als_seed, canonical_seeds as _canonical_seeds

    if warm_start_als and st.B_a_search is not None and st.B_b_search is not None and ca_size <= st.B_a_search.shape[1] and cb_size <= st.B_b_search.shape[1]:
        assert B_a_deg is not None and B_b_deg is not None  # derived from B_a_search/B_b_search, non-None per the guard above
        try:
            als_a, als_b = warm_start_als_seed(
                B_a_deg,
                B_b_deg,
                st.y_search_any,
                # DEVICE-BORN design (2026-06-30, H2D collapse): the standardised
                # columns + basis B_a_search/B_b_search were built from are in
                # scope, so route the resident GPU branch through
                # warm_start_als_seed_gpu_from_z - it rebuilds the (degree+1)
                # design ON DEVICE (max_degree = ca_size-1 / cb_size-1, the SAME
                # leading columns the [:ca_size]/[:cb_size] slice selects from an
                # orthogonal-poly basis) instead of uploading the prebuilt slices.
                # The CPU path ignores z/basis and stays byte-identical.
                z_a=st.z_a_search, z_b=st.z_b_search, basis=basis,
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
        if st.canonical_seeds_func is not None:
            # Non-polynomial basis ships its own canonical seeds via the registry.
            seeds_per_feature = st.canonical_seeds_func(degree)
        else:
            seeds_per_feature = _canonical_seeds(basis, degree)
        # Pair every seed with every other seed for c_b (limited to keep init pop small).
        for s_a in seeds_per_feature:
            warm_seeds.extend(np.concatenate([s_a, s_b]) for s_b in seeds_per_feature)
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
    # monotone-inner product target - the regime where CMA must actually
    # search). Median iters COLD=78 / WARM=79 (the prior seed adds ONE eval
    # and saves ZERO generations); wall -0.3%..-3.7% (slightly SLOWER).
    # ROOT CAUSE: the per-pair ALS warm-start (``warm_start_als``, the block
    # above) already re-derives the true basin from THIS fold's data each
    # call and lands x0 there, so a cross-fold coefficient prior is strictly
    # subsumed. WORSE, the extra seed perturbs the CMA population enough to
    # land on a DIFFERENT optimum on 8/12 folds (selection NOT byte-identical
    # - 6 higher-MI, 2 lower), which fails idea #20's identical-or-stabler
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
    return bf_idx_best, coef_a_best, coef_b_best, raw_mi_best
