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
from joblib._parallel_backends import LokyBackend

from ._joblib_safe import POLYNOM_LOKY_IDLE_WORKER_TIMEOUT, disable_cuda_in_worker, fit_constant_memmap, run_in_big_stack_thread
from .discretization import discretize_array
from .engineered_recipes import build_hermite_pair_recipe
from .hermite_fe import optimise_hermite_pair, precompute_hermite_pair_basis

logger = logging.getLogger(__name__)

# Cheap-first dispatch sentinel: a plain picklable string so it survives the
# loky process-pool round-trip (a module-level object() would not pickle by
# identity across processes). ``_eval_one_pair_impl`` returns it in the
# ``best_res`` slot for a pair whose expensive optimiser was skipped because the
# cheap trivial baseline already saturated the joint-MI ceiling; the serial
# reduce counts these for the summary log and treats them as no-injection.
_POLY_CHEAP_SKIP = "__poly_cheap_skip__"


from ._fe_family_timing import fe_timed


@fe_timed("smart_polynom")
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
    # STRATIFIED SUBSAMPLE (R1, 2026-06-18). When True the per-pair inner-search subsample below
    # draws a TARGET-STRATIFIED set of rows (per-class proportional for classification, y-quantile
    # for regression) instead of the plain uniform ``rng.choice``, so the CMA/Optuna inner MI is
    # evaluated on a slice that retains rare classes / target tails. False (default) keeps the
    # byte-identical legacy uniform draw. The caller resolves the MRMR ``fe_subsample_stratify``
    # tri-state knob (None=auto) to a concrete bool via ``_resolve_fe_subsample_stratify``.
    fe_subsample_stratify: bool = False,
    # ONE shared FE subsample (2026-06-25). When the caller passes the fit's single shared row-index
    # draw, the inner-search subsample REUSES it verbatim instead of drawing its own per-pair slice, so
    # the polynom path scores the SAME rows as the pair-search / sufficiency floor (one draw per fit).
    # ``None`` keeps the legacy per-pair uniform/stratified draw below.
    shared_subsample_idx: np.ndarray | None = None,
    # 2026-06-02 CHEAP-FIRST DISPATCH: the expensive CMA/Optuna orthogonal-poly
    # search only earns its cost on pairs whose signal a trivial library
    # unary/binary feature CANNOT already capture (non-monotone inners like
    # ``a**3-2a`` -> trivial MI << joint ceiling). When the cheap trivial
    # baseline already captures >= this fraction of the pair's joint-MI ceiling
    # (``pair_mi``, the SAME quantity the prevalence gate uses), a 1-D poly
    # feature cannot materially beat it (no function of (a,b) exceeds the joint
    # MI), so the optimiser is SKIPPED for that pair -- the trivial feature is
    # materialised by the always-on unary/binary path. Monotone-easy pairs skip;
    # non-monotone-hard pairs (F-POLY) fall through and optimise. Set to 1.0 to
    # disable (always optimise every prospective pair = legacy behaviour).
    poly_cheap_skip_ratio: float = 0.97,
    # LINEAR-USABILITY GUARD on the cheap-first skip. MI saturation alone is NOT
    # sufficient to skip: a trivial feature can capture the pair's joint MI while
    # being almost LINEARLY USELESS (e.g. ``atan2(a,b)`` reaches MI 0.37 on a
    # binarised bilinear target yet |corr| to y is ~0.10 -- it encodes the signal
    # as an angle). The orthogonal-poly optimiser then produces a far more
    # linearly-usable feature (|corr| ~0.80) that the MI-only skip would discard.
    # Skip therefore requires the trivial feature to ALSO be linearly useful:
    # |corr(trivial, y)| >= this floor. Monotone-easy pairs (trivial already
    # high-corr) still skip; MI-saturated-but-non-linear pairs fall through and
    # optimise. Set 0.0 to restore the MI-only skip.
    #
    # 0.90 (not 0.5): the orthogonal-poly optimiser reshapes a trivial feature for
    # LINEAR usability even when MI is saturated -- measured the "easy" monotone
    # pair exp(a)*log(b) lifts from trivial-``mul`` |corr| 0.71 to polynom |corr|
    # 0.92, and the bilinear xor pair from 0.59 to 0.80. So the skip is only truly
    # lossless when the trivial is ALREADY near-perfectly linear (|corr| >= 0.90);
    # below that the optimiser has real linear headroom and must run.
    poly_cheap_skip_min_corr: float = 0.90,
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
    pl: Any = None
    if is_polars_input:
        import polars as pl

    # ``prospective_pairs`` is keyed by ``(raw_vars_pair, _pair_mi)`` composite
    # tuples; the polynom-FE body only needs ``raw_vars_pair`` itself.
    _pair_keys = [k[0] for k in prospective_pairs.keys()]
    # Last-line numeric guard: the Hermite / polynomial basis (np.isfinite, z-score,
    # minmax) raises ``ufunc 'isfinite' not supported`` / ``unsupported operand`` on a
    # string column. Drop any pair whose operand column is non-numeric in X -- a string
    # categorical operand can slip through the upstream pool filter via a cached pair or
    # a synergy-kept operand. Indices are positional into X (``X_ndarr[:, idx]`` below).
    _numeric_pos = None
    try:
        _schema = getattr(X, "schema", None)
        if _schema is not None:  # polars
            _numeric_pos = {i for i, c in enumerate(X.columns) if _schema[c].is_numeric()}
        else:
            _dtypes = getattr(X, "dtypes", None)
            if _dtypes is not None:  # pandas
                import pandas as _pd
                _numeric_pos = {i for i, _dt in enumerate(_dtypes) if _pd.api.types.is_numeric_dtype(_dt)}
    except Exception:
        _numeric_pos = None
    if _numeric_pos is not None:
        _pair_keys = [p for p in _pair_keys if int(p[0]) in _numeric_pos and int(p[1]) in _numeric_pos]
    _n_pairs_to_eval = len(_pair_keys)
    if _n_pairs_to_eval == 0:
        return data, nbins, cols, X
    # Cheap-first dispatch: per-pair joint-MI ceiling (``pair_mi`` from the key)
    # so ``_eval_one_pair_impl`` can skip the expensive optimiser when the cheap
    # trivial baseline already captures >= ``poly_cheap_skip_ratio`` of it.
    _pair_mi_ceiling = {}
    for _k in prospective_pairs.keys():
        try:
            _pair_mi_ceiling[_k[0]] = float(_k[1])
        except (TypeError, ValueError, IndexError):  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
            pass

    _polynom_n_jobs = int(n_jobs) if n_jobs and n_jobs > 0 else 1
    logger.info(
        "Polynomial-pair FE starting: %d pair(s) x %d Optuna restart(s) x "
        "%d trial(s) per restart, n_jobs=%d backend=loky.",
        _n_pairs_to_eval, fe_smart_polynom_iters,
        fe_smart_polynom_optimization_steps, _polynom_n_jobs,
    )

    # 2026-05-22 loky-memmap fix: convert X to a single contiguous ndarray
    # ONCE so joblib can memmap it across loky workers. The previous
    # closure-captured X path would have cloudpickled the full 4M-row
    # frame INTO each worker (16 workers x ~432MB = ~7GB IPC traffic on
    # startup); passing X_ndarr as an explicit ``delayed()`` arg lets
    # joblib auto-detect it as a large ndarray, dump-to-temp ONCE, and
    # share via OS page cache (Windows: file-backed mmap; Linux: shm).
    # Per-pair tasks then index into the shared X_ndarr by column number
    # without pulling a fresh copy.
    if is_polars_input:
        X_ndarr = X.to_numpy()  # polars -> (N, F) contiguous ndarray
    else:
        X_ndarr = X.values if hasattr(X, "values") else np.asarray(X)
    # run_polynom_pair_fe is called once per FE round (up to fe_max_steps times per fit) with the SAME
    # X content each time -- a fresh Parallel(...) call below re-triggers joblib's memmapping reducer's
    # OWN dump of X_ndarr per call (it only dedups WITHIN one Parallel() invocation's tasks, not ACROSS
    # separate calls). fit_constant_memmap (shared with _step_pairmi.py's identical fix) dumps content
    # ONCE per process and hands back the read-only memmap on every subsequent round -- joblib passes an
    # existing np.memmap to loky workers by filename with no re-dump at all.
    X_ndarr = fit_constant_memmap(X_ndarr)

    def _eval_one_pair_impl(raw_vars_pair, X_arr, y_arr):
        """Search + fit the best polynomial-basis feature for one raw variable pair; runs inside a worker.

        Extracts the pair's two columns from the shared ``X_arr``, subsamples for the optimiser's MI loop (per
        ``shared_subsample_idx`` / ``subsample_n``), computes the trivial-feature baseline once, cheap-skips the
        expensive CMA/Optuna search when that baseline already saturates the pair's joint-MI ceiling (subject to the
        linear-usability guard), and otherwise runs ``optimise_hermite_pair`` for ``fe_smart_polynom_iters`` restarts,
        keeping the best-MI result. Returns ``(raw_vars_pair, best_res_or_sentinel, vals_a_full, vals_b_full)`` on the
        FULL (non-subsampled) columns so the injection step transforms every row."""
        # X_arr is X.to_numpy()/.values; a frame with ANY string column makes the WHOLE array object dtype, so even a
        # numeric operand extracts as an object slice that the Hermite basis (np.isfinite / z-score / minmax) rejects.
        # The pair already passed the numeric-position guard above, so coerce to float64; a genuinely non-numeric operand
        # that slipped the guard raises here and the pair is skipped (no polynomial FE for it).
        try:
            vals_a_full = np.ascontiguousarray(X_arr[:, raw_vars_pair[0]], dtype=np.float64)
            vals_b_full = np.ascontiguousarray(X_arr[:, raw_vars_pair[1]], dtype=np.float64)
        except (ValueError, TypeError):
            return None
        if np.std(vals_a_full) < 1e-12 or np.std(vals_b_full) < 1e-12:
            return None
        # 2026-05-18 subsample for CMA-ES inner search. Final transform on
        # FULL source array preserves precision; only the optimiser's MI
        # evaluation uses the slice.
        _poly_shared_idx = None
        if shared_subsample_idx is not None:
            try:
                _psi = np.asarray(shared_subsample_idx)
                if _psi.ndim == 1 and 0 < _psi.shape[0] < len(vals_a_full) and int(_psi.max()) < len(vals_a_full):
                    _poly_shared_idx = _psi.astype(np.int64, copy=False)
            except Exception:
                _poly_shared_idx = None
        if _poly_shared_idx is not None:
            # Reuse the fit's ONE shared draw (same rows as the pair-search / sufficiency floor).
            _ss_idx = _poly_shared_idx
            vals_a_sub = vals_a_full[_ss_idx]
            vals_b_sub = vals_b_full[_ss_idx]
            classes_y_sub = y_arr[_ss_idx] if hasattr(y_arr, "__getitem__") else y_arr
        elif subsample_n and 0 < subsample_n < len(vals_a_full):
            _ss_rng = np.random.default_rng(subsample_seed + int(raw_vars_pair[0]) * 1000 + int(raw_vars_pair[1]))
            if fe_subsample_stratify and hasattr(y_arr, "__getitem__"):
                # ``classes_y`` (y_arr) is the discrete target the inner MI scores against -> classification
                # stratification keeps the rare class in every per-pair subsample.
                from ._fe_subsample import stratified_subsample_idx
                _ss_idx = stratified_subsample_idx(_ss_rng, np.asarray(y_arr), int(subsample_n), is_clf=True)
            else:
                _ss_idx = _ss_rng.choice(len(vals_a_full), size=subsample_n, replace=False)
            vals_a_sub = vals_a_full[_ss_idx]
            vals_b_sub = vals_b_full[_ss_idx]
            classes_y_sub = y_arr[_ss_idx] if hasattr(y_arr, "__getitem__") else y_arr
        else:
            vals_a_sub = vals_a_full
            vals_b_sub = vals_b_full
            classes_y_sub = y_arr
        # 2026-05-20 NEW-A: compute trivial baseline ONCE per pair (was
        # silently re-computed once per ``fe_smart_polynom_iters`` restart
        # inside ``optimise_hermite_pair``). On the n=200k production
        # config ``best_trivial_pair`` is ~50-150ms; with the default
        # ``fe_smart_polynom_iters=5`` and 12 pairs that adds up to ~60
        # redundant calls, saving 3-9 seconds without changing the
        # numerical result (baseline is a deterministic fn of (x_a, x_b, y)).
        _trivial_baseline = None
        _trivial_name = None
        _trivial_feat = None
        try:
            from .fe_baselines import best_trivial_pair as _best_trivial_pair
            _t = _best_trivial_pair(
                np.asarray(vals_a_sub, dtype=np.float64),
                np.asarray(vals_b_sub, dtype=np.float64),
                classes_y_sub,
                discrete_target=True,
                mi_estimator=fe_mi_estimator,
                plugin_n_bins=20,
            )
            if _t is not None:
                _trivial_name, _trivial_feat, _trivial_baseline = _t
        except Exception as _e:
            logger.debug(
                "best_trivial_pair precompute failed for pair %s: %r; " "optimise_hermite_pair will recompute internally.",
                raw_vars_pair,
                _e,
            )

        # CHEAP-FIRST DISPATCH: skip the expensive CMA/Optuna search when the
        # cheap trivial baseline already captures >= ``poly_cheap_skip_ratio`` of
        # this pair's joint-MI ceiling. A 1-D engineered feature cannot exceed
        # the pair's joint MI, so once the trivial feature is that close to the
        # ceiling the optimiser has no headroom -- the trivial feature (which the
        # always-on unary/binary path materialises) is as good as it gets. Hard
        # pairs whose non-monotone inner the trivial set cannot express (trivial
        # MI << ceiling) fall through and DO optimise, so recovery on the cases
        # that need the orthogonal-poly basis is unaffected.
        if poly_cheap_skip_ratio < 1.0 and _trivial_baseline is not None and _trivial_baseline > 0.0:
            _ceiling = _pair_mi_ceiling.get(raw_vars_pair)
            if _ceiling is not None and _ceiling > 0.0 and _trivial_baseline >= _ceiling * poly_cheap_skip_ratio:
                # LINEAR-USABILITY GUARD: MI saturation is necessary but not
                # sufficient. A trivial feature can hit the MI ceiling while being
                # nearly orthogonal to y in LINEAR terms (atan2 on a bilinear
                # target: MI 0.37, |corr| 0.10), in which case the orthogonal-poly
                # optimiser still produces a far more linearly-usable feature.
                # Only skip when the trivial feature is ALSO linearly useful.
                _trivial_corr = 0.0
                if _trivial_feat is not None and poly_cheap_skip_min_corr > 0.0:
                    try:
                        _tf = np.asarray(_trivial_feat, dtype=np.float64).reshape(-1)
                        _yv = np.asarray(classes_y_sub, dtype=np.float64).reshape(-1)
                        if _tf.size == _yv.size and float(np.std(_tf)) > 1e-12 and float(np.std(_yv)) > 1e-12:
                            _trivial_corr = abs(float(np.corrcoef(_tf, _yv)[0, 1]))
                    except Exception:
                        _trivial_corr = 1.0  # corr unavailable -> fall back to MI-only skip
                if poly_cheap_skip_min_corr <= 0.0 or _trivial_corr >= poly_cheap_skip_min_corr:
                    return (raw_vars_pair, _POLY_CHEAP_SKIP, vals_a_full, vals_b_full)

        # Precompute the basis fit (z_a/preprocess_a/z_b/preprocess_b) + the identity baseline ONCE per pair --
        # the SAME bug class as the trivial-baseline hoist above (_trivial_baseline), just for the basis fit +
        # _baseline_mi_pair that optimise_hermite_pair would otherwise redo byte-for-byte on every one of the
        # fe_smart_polynom_iters restarts (only ``seed`` differs; vals_a_sub/vals_b_sub/classes_y_sub are identical).
        _pz_a = _pp_a = _pz_b = _pp_b = _pib = None
        try:
            _pz_a, _pp_a, _pz_b, _pp_b, _pib = precompute_hermite_pair_basis(
                vals_a_sub, vals_b_sub, classes_y_sub,
                discrete_target=True,
                basis=fe_polynomial_basis,
                mi_estimator=fe_mi_estimator,
            )
        except Exception as _e:
            logger.debug(
                "precompute_hermite_pair_basis failed for pair %s: %r; " "optimise_hermite_pair will recompute internally.",
                raw_vars_pair,
                _e,
            )

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
                precomputed_trivial_baseline=_trivial_baseline,
                precomputed_trivial_name=_trivial_name,
                precomputed_z_a=_pz_a,
                precomputed_preprocess_a=_pp_a,
                precomputed_z_b=_pz_b,
                precomputed_preprocess_b=_pp_b,
                precomputed_identity_baseline=_pib,
            )
            if res is not None and (best_res is None or res.mi > best_res.mi):
                best_res = res
        # Return FULL arrays so the injection step applies the polynomial
        # to all rows (subsampling was only for the optimiser's MI loop).
        return (raw_vars_pair, best_res, vals_a_full, vals_b_full)

    def _eval_one_pair(raw_vars_pair, X_arr, y_arr):
        """Worker entry point: runs :func:`_eval_one_pair_impl` on a big-stack sub-thread to dodge the Windows loky 1MB-stack numba crash."""
        # Windows loky workers have a 1MB main-thread stack -- numba's
        # JIT cache load runs an llvmlite finalize chain that needs
        # ~2-3MB and crashes the worker. Running the impl in a sub-thread
        # with 8MB stack avoids the overflow; pass-through no-op on Linux.
        return run_in_big_stack_thread(
            _eval_one_pair_impl, raw_vars_pair, X_arr, y_arr,
        )

    _poly_t0 = time.perf_counter()
    # 2026-05-18 threshold: at n=1M, 15 pairs, joblib worker spin-up
    # exceeded the per-pair work (11s parallel vs 5.75s serial). At 50+
    # pairs (typical 10-feature problem) parallel wins.
    #
    # 2026-05-22 backend FIX: prior ``backend="threading"`` left only one
    # CPU core busy in prod because the per-pair work spends most time
    # inside Optuna's TPE/RandomSampler decision loop -- pure-Python,
    # holds GIL. 16 threading workers serialised behind the GIL gave
    # the user-visible "16 workers but only 1 core busy" pathology
    # (231s wall-clock for 54 pairs that should have been ~15s).
    # Switched to ``backend="loky"`` (process pool): the per-pair work
    # carries its own NumPy arrays through pickle but the Optuna loop
    # then runs truly in parallel. Threading is still hard-coded for
    # the tail-loop where per-pair work is dominated by NumPy/Numba
    # (those release the GIL); the polynom-FE inner search is the
    # opposite regime.
    _PARALLEL_PAIR_THRESHOLD = 16
    if _polynom_n_jobs > 1 and _n_pairs_to_eval >= _PARALLEL_PAIR_THRESHOLD:
        # ``inner_max_num_threads=1`` caps the per-loky-worker BLAS / OpenMP /
        # Numba thread pool so 16 worker processes don't each spawn N numba
        # threads and oversubscribe the CPU. The inner polyeval_dispatch
        # kernel (njit_par at n >= 50_000) was sized to saturate cores via
        # ONE worker -- here we explicitly trade kernel-side parallelism for
        # sampler-side parallelism because Optuna's TPE/Random sampler
        # between trials is the actual bottleneck (~50% of per-trial time
        # in prod, holds GIL, can only be split across processes).
        #
        # ``initializer=disable_cuda_in_worker`` forces every loky worker CPU-ONLY
        # (CUDA_VISIBLE_DEVICES="") so it does NOT grab its own ~250 MB cupy CUDA
        # context -- 16 worker contexts filled a 4 GB card and stalled the search
        # ~2h (see the shared helper's docstring, 2026-07-05). We build a
        # ``LokyBackend`` instance instead of passing ``backend="loky"`` + kwargs to
        # ``Parallel`` because in joblib 1.5.x the ``initializer`` (and even
        # ``inner_max_num_threads``) are ONLY honoured when set on the backend
        # object; passed straight to ``Parallel(...)`` they are silently dropped.
        # SHARED initializer, not a local duplicate (2026-07-10 fix): this used to
        # be a separate ``_poly_worker_disable_cuda`` function with an identical
        # body -- loky's ``get_reusable_executor`` keys pool reuse on the
        # initializer FUNCTION REFERENCE, so two behaviourally-identical-but-
        # distinct callables meant this pool could never reuse the warm CPU-only
        # pool ``_step_pairmi.py``'s pair-MI sweep already spun up moments earlier
        # in the SAME fit, paying a full fresh 16-worker spawn (process create +
        # mlframe/numba re-import per worker) every time instead. Using the SAME
        # shared ``disable_cuda_in_worker`` lets the two call sites reuse one pool.
        # Memmapping of large ``X_ndarr`` is preserved (LokyBackend still uses the
        # memmapping executor), so this does not copy X per task.
        _loky_cpu_backend = LokyBackend(
            inner_max_num_threads=1,
            initializer=disable_cuda_in_worker,
            # Must match maybe_prewarm_polynom_loky_pool's idle_worker_timeout for get_reusable_executor's
            # reuse-key to hit the pre-warmed pool (2026-07-11) -- see that function's docstring for the
            # measured gap this timeout needs to survive and the production A/B history behind its value.
            idle_worker_timeout=POLYNOM_LOKY_IDLE_WORKER_TIMEOUT,
        )
        try:
            # Fallback to the exact serial path on ANY dispatch failure (2026-07-10 fix; reproduced
            # live at n=3M production scale on a small-VRAM card). A GPU OOM earlier in the SAME fit
            # can leave the main process's CUDA context in a poisoned state; cloudpickle serializing
            # the task closure then fails with ``_pickle.PicklingError`` (CUDADriverError as its
            # ``__cause__``) even though this pool's WORKERS are CPU-only -- the failure is in the
            # PARENT process's pickling step, before any task reaches a worker. Pre-fix this exception
            # propagated all the way up and crashed the whole training run. The per-pair work here is
            # independent of which backend runs it (same ``_eval_one_pair`` call either way), so
            # falling back to serial is safe and selection-equivalent, just slower for this one FE
            # stage on this one fit.
            _poly_pair_results = Parallel(
                n_jobs=_polynom_n_jobs, backend=_loky_cpu_backend,
                verbose=10 if verbose else 0,
            )(delayed(_eval_one_pair)(rv, X_ndarr, classes_y) for rv in _pair_keys)
        except Exception as _pool_exc:
            logger.warning(
                "Polynomial-pair FE: loky pool dispatch failed (%s: %s); falling back to the serial "
                "per-pair path for this round [n_pairs=%d].",
                type(_pool_exc).__name__, _pool_exc, _n_pairs_to_eval,
            )
            _poly_pair_results = [_eval_one_pair(rv, X_ndarr, classes_y) for rv in _pair_keys]
    else:
        if _polynom_n_jobs > 1 and verbose:
            logger.info(
                "Polynomial-pair FE: n_pairs=%d < threshold %d -- " "running serial to avoid joblib overhead.",
                _n_pairs_to_eval,
                _PARALLEL_PAIR_THRESHOLD,
            )
        _poly_pair_results = [_eval_one_pair(rv, X_ndarr, classes_y) for rv in _pair_keys]
    _eval_elapsed = time.perf_counter() - _poly_t0
    logger.info(
        "Polynomial-pair FE eval phase done in %.1fs (%d pairs, " "%.2fs/pair median).",
        _eval_elapsed,
        _n_pairs_to_eval,
        _eval_elapsed / max(_n_pairs_to_eval, 1),
    )

    # Serial reduce: log + uplift gate + inject into data/cols/X.
    _uplift_gate = float(fe_min_engineered_mi_prevalence)
    _n_cheap_skipped = 0
    # Accumulate surviving columns/names/nbins in lists and do ONE concat after the loop instead of a
    # np.append (= np.concatenate) per survivor: np.append reallocates + copies the ENTIRE existing `data`
    # matrix on every iteration, O(N x F x K) traffic for K survivors instead of O(N x F). `_existing_col_names`
    # substitutes for the incremental `cols` growth the dedup check used to read (nothing else in this loop
    # reads back from `data`/`cols`/`nbins` before they're rebuilt below -- `_src_a`/`_src_b` index into the
    # ORIGINAL `cols` by position, unaffected either way).
    _existing_col_names = set(cols)
    _new_data_cols: List[np.ndarray] = []
    _new_col_names: List[str] = []
    _new_col_nbins: List[int] = []
    for _pair_result in _poly_pair_results:
        if _pair_result is None:
            continue
        raw_vars_pair, best_res, vals_a, vals_b = _pair_result
        # Cheap-first dispatch skipped the optimiser for this pair (the trivial
        # baseline already saturated the joint-MI ceiling). Count it for the
        # summary log; the always-on unary/binary path materialises the trivial
        # feature, so nothing is lost -- only the expensive search was spared.
        if best_res is _POLY_CHEAP_SKIP or (isinstance(best_res, str) and best_res == _POLY_CHEAP_SKIP):
            _n_cheap_skipped += 1
            continue
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
            _src_a = cols[raw_vars_pair[0]] if 0 <= raw_vars_pair[0] < len(cols) else f"col{raw_vars_pair[0]}"
            _src_b = cols[raw_vars_pair[1]] if 0 <= raw_vars_pair[1] < len(cols) else f"col{raw_vars_pair[1]}"
            _new_col_name = f"_polynom_{best_res.basis}_{best_res.bin_func_name}" f"__{_src_a}__{_src_b}"
            if _new_col_name in _existing_col_names:
                continue
            _new_binned = discretize_array(
                arr=_t_vals,
                n_bins=quantization_nbins,
                method=quantization_method,
                dtype=quantization_dtype,
            ).reshape(-1, 1)
            _new_data_cols.append(_new_binned)
            _new_col_nbins.append(int(quantization_nbins))
            _new_col_names.append(_new_col_name)
            _existing_col_names.add(_new_col_name)
            if is_polars_input:
                X = X.with_columns(pl.Series(_new_col_name, _t_vals))
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
                    "Polynomial-pair FE injection failed for pair=%s: %s. " "Standard FE block below still runs.",
                    raw_vars_pair,
                    _inj_err,
                )
    if _new_data_cols:
        # ONE reallocation for all survivors instead of one per survivor (see the loop-entry comment).
        data = np.concatenate([data, *_new_data_cols], axis=1)
        nbins = np.concatenate([
            np.asarray(nbins),
            np.asarray(_new_col_nbins, dtype=nbins.dtype),
        ])
        cols = [*cols, *_new_col_names]
    if _n_cheap_skipped:
        logger.info(
            "Polynomial-pair FE cheap-first dispatch: skipped the optimiser for "
            "%d/%d pair(s) whose trivial baseline already reached >= %.0f%% of "
            "the joint-MI ceiling (the always-on unary/binary path materialises "
            "those); optimised %d hard pair(s).",
            _n_cheap_skipped, _n_pairs_to_eval, 100.0 * poly_cheap_skip_ratio,
            _n_pairs_to_eval - _n_cheap_skipped,
        )
    return data, nbins, cols, X
