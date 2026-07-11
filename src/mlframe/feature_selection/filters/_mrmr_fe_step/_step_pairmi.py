"""Operand-pool feed-forward cap + batch pair-MI computation + order-2 maxT floor stage of ``MRMR._run_fe_step``.

Carved verbatim from ``_step_core.py`` to keep that module under the 1k-LOC ceiling.
``compute_pair_mis_and_floor`` runs the engineered-operand feed-forward cap, the batch/per-pair joint-MI
computation (CUDA/CPU dispatch + legacy joblib fallback, populating ``cached_MIs`` in place), the order-2
Westfall-Young maxT permutation-null floor, and the "auto" prevalence MM-debias bias fill. The lazy
``..mrmr`` imports are done in-body exactly as in the inline block (the mrmr<->this-package import cycle
forbids a top-level import). Selection is byte-for-byte identical to the inline block.

Returns ``(numeric_vars_to_consider, _eng_cap, _pair_maxt_floor, _pair_mm_bias, _prevalence_debias_auto)``;
``cached_MIs`` is mutated in place.
"""
from __future__ import annotations

import logging
import os
from itertools import combinations
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed
from joblib._parallel_backends import LokyBackend

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from .._mrmr_fe_step_helpers import compute_pair_maxt_floor
from .._joblib_safe import disable_cuda_in_worker
from .._fe_family_timing import record_fe_family_wall


def compute_pair_mis_and_floor(
    self,
    *,
    data, cols, nbins, X,
    classes_y, classes_y_safe, freqs_y,
    target_indices,
    cached_MIs, cached_confident_MIs,
    numeric_vars_to_consider,
    _prevalence_debias_auto,
    n_jobs, prefetch_factor, parallel_kwargs,
    fe_min_nonzero_confidence, fe_npermutations,
    fe_min_pair_mi, fe_min_pair_mi_prevalence,
    verbose,
):
    """Compute pair MIs + the order-2 maxT floor for the FE candidate pool. ``cached_MIs`` mutated in place."""
    # Lazy import: ``.mrmr`` re-imports this package at its bottom for method binding -> a top-level
    # ``from ..mrmr import ...`` here would create a hard import cycle (see _step_core).
    from ..mrmr import (
        _lazy_chunks,
        _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS,
        compute_pairs_mis,
        tqdmu,
    )

    # ENGINEERED-OPERAND FEED-FORWARD CAP (2026-06-08). At FE step k>1 the operand
    # pool now also carries the engineered columns selected by the prior step(s)
    # (their cols-space indices are promoted into ``selected_vars`` at this
    # function's bottom and re-confirmed by the next screening pass). Feeding them
    # back lets the pair search build COMPOSITES of two engineered features -- e.g.
    # the additive ``add(div(sqr(a),abs(b)), mul(log(c),sin(d)))`` that captures ~the
    # entire deterministic signal. But engineered columns accumulate across steps,
    # so an uncapped feed-forward makes the O(k^2) pair count blow up. Keep only the
    # top-K engineered operands BY THEIR MARGINAL SCREENING MI (``cached_MIs[(idx,)]``,
    # populated by screen_predictors for every selected var); the rest still reach
    # ``support_`` -- they just don't seed further composites. ``fe_max_engineered_operands``:
    # 0 -> raw-only pool (legacy, no composites), <0 -> no cap, >0 -> top-K cap.
    # feature_names_in_ is an ndarray; "or []" would test truthiness and raise on a multi-element array.
    _raw_name_set = set(getattr(self, "feature_names_in_", []))
    _eng_cap = int(getattr(self, "fe_max_engineered_operands", 8))
    _engineered_in_pool = [v for v in numeric_vars_to_consider if cols[v] not in _raw_name_set]
    # ESCALATION FEATURES ARE TERMINAL -- never feed them forward as composite operands
    # (2026-06-12, F2 rescue). An ``esc_*`` escalation feature (orth-poly / adaptive-Fourier
    # pair warp) already captures a genuine richer-basis interaction the library could not
    # express; nesting it INTO a further pair composite (the feed-forward built
    # ``div(log(esc_poly_legendre_mul(a,b)),exp(mul(prewarp(c),prewarp(d))))`` on F2) fuses
    # two INDEPENDENT additive terms of the target into a single ratio whose joint MI tops
    # the greedy ranking, so MRMR then DROPS the clean raw predictors and the standalone
    # captures -- measured on F2: the standalone esc_poly(a,b) is a +0.05 downstream R^2 WIN
    # (raw c,d,f 0.943 -> +a**2/b 0.995), but the fed-forward nested form REGRESSES it to
    # 0.864 by restructuring the selection. The escalation feature stays selected (reaches
    # ``support_``); it just does not seed further composites. Gated on
    # ``fe_escalation_feedforward_enable`` (default OFF -- terminal is the safe default).
    if not bool(getattr(self, "fe_escalation_feedforward_enable", False)):
        _esc_idx = {
            v
            for v in _engineered_in_pool
            if str(cols[v]).startswith("esc_") or "esc_poly_" in str(cols[v]) or "esc_fourier_" in str(cols[v]) or "esc_chirp_" in str(cols[v])
        }
        if _esc_idx:
            numeric_vars_to_consider = set(numeric_vars_to_consider) - _esc_idx
            _engineered_in_pool = [v for v in _engineered_in_pool if v not in _esc_idx]
    if _engineered_in_pool and _eng_cap >= 0:
        if _eng_cap == 0:
            _keep_eng: set = set()
        else:
            # Rank by marginal MI desc; missing single-var MI sorts last. Tie-break on the var index so the
            # cap survivors are deterministic (the pool derives from a set, whose iteration order is not stable
            # across processes -- without the secondary key equal-MI operands kept an arbitrary, irreproducible subset).
            _ranked = sorted(_engineered_in_pool, key=lambda v: (-cached_MIs.get((v,), 0.0), v))
            _keep_eng = set(_ranked[:_eng_cap])
        _drop_eng = set(_engineered_in_pool) - _keep_eng
        if _drop_eng:
            numeric_vars_to_consider = set(numeric_vars_to_consider) - _drop_eng
            if verbose:
                logger.info(
                    "MRMR FE feed-forward: kept top %d engineered operand(s) by marginal MI for composite pairing, "
                    "dropped %d below fe_max_engineered_operands=%d to bound the pair count (they remain selected).",
                    len(_keep_eng), len(_drop_eng), _eng_cap,
                )

    # `combinations(...)` is consumed lazily by tqdmu (small path) or by
    # `_lazy_chunks` (large path). Pair count is closed-form, avoiding
    # `list(combinations(...))` materialisation (O(k^2) tuples, ~300 MB at
    # k=5000) before chunking even starts.
    _k = len(numeric_vars_to_consider)
    n_pairs = (_k * (_k - 1)) // 2

    if verbose:
        logger.info("Feature Engineering: Computing MIs of %d most prospective feature pairs...", n_pairs)

    # ---------------------------------------------------------------------------------------------------------------
    # Layer 3 pre-batch: compute pair MIs for every (a, b) in numeric_vars_to_consider via
    # dispatch_batch_pair_mi_chunked (CUDA / CPU njit prange by size, RAM-bounded row-block chunking). Pre-fills
    # cached_MIs[pair] so the per-pair compute_pairs_mis loop below skips the permutation-test branch entirely
    # (since "pair in cached_MIs" short-circuits at feature_engineering.py:394).
    #
    # Semantic change vs the legacy path: pairs no longer go through the permutation-test confidence filter
    # (min_nonzero_confidence). The raw original_mi is used as the FE-pair signal. Bench (commit 57f772c) shows
    # 10-30x speedup over the per-pair joblib loop; downstream MRMR FE pair selection is regression-validated by the
    # MRMR test suite. Disable by setting MLFRAME_MRMR_BATCH_PAIR_MI=0 (the env-var is the emergency rollback knob).
    #
    # NO POOL-SIZE CAP (removed 2026-07-09): the legacy ~35s/pair per-pair joblib fallback was previously forced
    # whenever the pool exceeded a flat 200-column ceiling (``_MRMR_BATCH_PRECOMPUTE_MAX_K``), which made a
    # realistic several-hundred-column production pool fall off a catastrophic-runtime cliff (observed: hours where
    # a few minutes was achievable). ``dispatch_batch_pair_mi_chunked`` enumerates the C(k,2) pair space in
    # RAM-bounded row-block chunks (never materialising the full pair-index arrays), so the fast batched path is
    # now ALWAYS used regardless of pool width -- there is no width at which this falls back to the slow path for
    # pool-size reasons. Note this does not (and cannot) make an EXHAUSTIVE pairwise sweep sub-quadratic: at true
    # extreme width (10^5-10^6 raw columns) C(k,2) itself is intractable, which is what ``sis_screen_threshold``
    # (Gate A, ``_mrmr_sis_screen.py``) exists to bound BEFORE this stage ever runs -- see that module's docstring.
    #
    # Guards:
    #   * n_pairs < _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS: pair count too small to amortise the dispatcher overhead.
    #   * Any backend failure (CUDA driver hiccup, dtype mismatch): logged WARN, fall through to legacy path.
    # Accept the common truthy/falsy spellings rather than require the operator
    # to remember the exact literals we sliced earlier. Empty / missing env
    # var defaults to ENABLED (the new behaviour).
    _BATCH_PRECOMPUTE_ENABLED = os.environ.get(
        "MLFRAME_MRMR_BATCH_PAIR_MI", "1",
    ).strip().lower() not in ("0", "false", "no", "off", "")
    _batch_prefill_count = 0
    # SECOND FUNNEL STAGE (2026-06-19): when the synergy bootstrap selected the GPU-exhaustive sweep it set
    # ``self._fe_synergy_exhaustive_active_`` -- FORCE the cuda backend so the full C(p,2) joint-MI sweep over ALL
    # raw numeric columns runs on the measured CUDA kernel (the only path that recovers a balanced L=0 interaction).
    # The exhaustive decision already verified GPU availability + budget. With the pool-size cap removed, exhaustive
    # mode no longer needs to "bypass" anything -- it only forces the backend choice.
    _exhaustive_active = bool(getattr(self, "_fe_synergy_exhaustive_active_", False))
    # When exhaustive is active, run the full C(p,2) sweep on the measured CUDA kernel where a GPU is
    # present, else on the CPU njit-prange backend (decide_exhaustive_sweep made the choice hardware-
    # independent, so a GPU-less host runs exhaustive on CPU rather than silently dropping to the lossy
    # pre-rank -- otherwise a balanced L=0 interaction feature would exist only on CUDA hosts).
    _exhaustive_backend = None
    if _exhaustive_active:
        try:
            from mlframe.feature_selection.filters.batch_pair_mi_gpu import _CUDA_AVAIL
        except Exception:
            _CUDA_AVAIL = False
        _exhaustive_backend = "cuda" if _CUDA_AVAIL else "njit_parallel"
    _batch_precompute_t0 = perf_counter()
    if _BATCH_PRECOMPUTE_ENABLED and n_pairs >= _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS:
        try:
            from mlframe.feature_selection.filters.batch_pair_mi_gpu import dispatch_batch_pair_mi_chunked

            _ids_arr = np.fromiter(numeric_vars_to_consider, dtype=np.int64, count=len(numeric_vars_to_consider))
            _pair_a_arr, _pair_b_arr, _pair_mi_batch, _backend_counts = dispatch_batch_pair_mi_chunked(
                factors_data=data,
                ids=_ids_arr,
                nbins=nbins,
                classes_y=classes_y,
                freqs_y=freqs_y,
                force_backend=_exhaustive_backend,
            )
            # Populate cached_MIs to short-circuit compute_pairs_mis's per-pair mi_direct call.
            # Skip pairs already in cached_confident_MIs (those had a confident permutation outcome).
            _n_pairs_batch = int(_pair_a_arr.shape[0])
            for _i in range(_n_pairs_batch):
                _p = (int(_pair_a_arr[_i]), int(_pair_b_arr[_i]))
                if _p not in cached_confident_MIs and _p not in cached_MIs:
                    cached_MIs[_p] = float(_pair_mi_batch[_i])
                    _batch_prefill_count += 1
            if verbose:
                _backend_summary = ", ".join(f"{k}={v}" for k, v in sorted(_backend_counts.items()))
                logger.info(
                    "MRMR FE: batch-prefilled %d/%d pair MIs via [%s] backend chunk(s) (permutation test skipped for these pairs)",
                    _batch_prefill_count, _n_pairs_batch, _backend_summary,
                )
        except Exception as _exc:
            # WARNING, not gated on ``verbose`` (2026-07-10 fix): this used to be ``if verbose:``, so on
            # a default ``verbose=0`` fit (the wellbore production config) a real batch-precompute
            # failure was completely SILENT -- live-reproduced at n=3M/p=423: this path failed with no
            # trace in the log, the downstream loky pool then also timed out, and the fit fell all the
            # way to the legacy per-pair sweep with zero diagnostic evidence of why. A failure this
            # consequential (it decides between a sub-minute batched pass and an hours-long serial one)
            # must always be visible, matching every other WARNING-level log in this module.
            logger.warning(
                "MRMR FE: dispatch_batch_pair_mi_chunked failed (%s: %s); falling back to legacy per-pair path "
                "[n_pairs=%d, n_rows=%d, n_classes_y=%d]",
                type(_exc).__name__, _exc,
                n_pairs, int(data.shape[0]) if hasattr(data, "shape") else -1,
                int(freqs_y.shape[0]) if hasattr(freqs_y, "shape") else -1,
            )
    record_fe_family_wall("pairwise_mi_batch_precompute", perf_counter() - _batch_precompute_t0)

    # SKIP THE LOKY POOL when the batch precompute above already covers every pair (2026-07-10 fix). With
    # the finding-#21 cap removed, the batch precompute now runs unconditionally at n_pairs>=8 and --
    # confirmed live (verbose repro, n=8000/p=249/n_pairs=31125): "batch-prefilled 31125/31125 pair MIs" --
    # routinely covers 100% of the pool, leaving NOTHING for the sweep below to compute. Spinning up a loky
    # PROCESS pool when there is zero real work is not just wasted overhead: it was the exact spot a
    # cold-cache multi-worker pool spawn was observed to take the full 300s watchdog bound (see the watchdog
    # comment below) before falling back to a serial pass that then finds nothing to do either. Routing a
    # fully-precomputed pool to the SERIAL branch is safe and cheap regardless of ``n_jobs``/``n_pairs``:
    # ``compute_pairs_mis`` already skips every pair already in ``cached_MIs``/``cached_confident_MIs`` (see
    # ``test_pair_mi_legacy_sweep_cache_starved.py``), so the serial pass is just a fast membership scan, not
    # a compute pass. Realistically thousands-to-low-hundred-thousands pairs at this stage (the operand-pool
    # PAIR stage, already bounded by the upstream operand caps -- not the raw-column count).
    _all_pairs_precomputed = n_pairs > 0 and all(
        (p in cached_MIs or p in cached_confident_MIs) for p in combinations(numeric_vars_to_consider, 2)
    )
    # Parallelise whenever (a) more than one worker is configured and
    # (b) we have at least n_jobs pairs to spread; per-pair MI compute is
    # ~35 s with default fe_npermutations on a wide frame, so parallel
    # overhead is amortised even at very small _k. Previously this took
    # the single-thread branch up to _k=50 (1225 pairs), serialising what
    # should be a 4-minute job into ~1 h on a 16-core box.
    _legacy_sweep_t0 = perf_counter()
    if n_jobs <= 1 or n_pairs < max(2, n_jobs) or _all_pairs_precomputed:
        if verbose and _all_pairs_precomputed:
            logger.info("MRMR FE: all %d prospective pair MIs already cached by the batch precompute; skipping the loky pool.", n_pairs)
        compute_pairs_mis(
            all_pairs=tqdmu(
                combinations(numeric_vars_to_consider, 2),
                total=n_pairs,
                desc="getting pairs MIs",
                leave=False,
                mininterval=5,
                disable=not verbose,
            ),
            data=data,
            target_indices=target_indices,
            nbins=nbins,
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            fe_min_nonzero_confidence=fe_min_nonzero_confidence,
            fe_npermutations=fe_npermutations,
            cached_confident_MIs=cached_confident_MIs,
            cached_MIs=cached_MIs,
            fe_min_pair_mi=fe_min_pair_mi,
            fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
        )
    else:
        chunk_size = max(1, n_pairs // (n_jobs * prefetch_factor))
        # BACKEND FIX (2026-07-06, wellbore diag): the per-chunk ``compute_pairs_mis``
        # body is GIL-bound CPU work -- it calls ``mi_direct`` per pair (joint plug-in
        # MI + the analytic/permutation null over ``fe_npermutations`` shuffles), all in
        # Python/numpy/njit-with-the-GIL-held-at-the-dispatch-boundary. Under the wellbore's
        # ``parallel_kwargs={'backend':'threading'}`` the whole chunk list serialised onto
        # ONE core (py-spy: MainThread stuck in joblib ``_retrieve`` sleep-poll at ~1.1
        # cores for ~42 min); GPU util was 0% because at the ~30k prospective-pair
        # subsample the dense-cell ANALYTIC MI null (n>=25k) is taken -- pure CPU, no CUDA.
        #
        # Route to a loky PROCESS pool for real multi-core parallelism, with every worker
        # forced CPU-ONLY via ``initializer=disable_cuda_in_worker`` (CUDA_VISIBLE_DEVICES="")
        # so no worker grabs a ~250 MB cupy context -- mirroring the ``run_polynom_pair_fe``
        # loky-CPU-only fix (commit 0476d8aa). This is selection-equivalent: ``compute_pairs_mis``
        # caches only the DETERMINISTIC ``original_mi`` (the confidence is discarded), and at
        # this n the analytic null path is CUDA-independent, so CPU-only workers compute the
        # identical pair-MI dict the threading/serial baseline does.
        #
        # We build a ``LokyBackend`` INSTANCE (not ``backend="loky"``) because in joblib 1.5.x
        # ``initializer`` / ``inner_max_num_threads`` are honoured ONLY when set on the backend
        # object -- passed to ``Parallel(...)`` directly they are silently dropped. Calling
        # ``joblib.Parallel`` here (not ``parallel_run``) because ``parallel_run`` does
        # ``"dask" in backend`` which raises on a backend instance. Memmapping of the large
        # ``data`` array is preserved (LokyBackend forwards to the memmapping executor).
        # ``inner_max_num_threads=1`` stops N worker processes each spawning N numba/BLAS
        # threads and oversubscribing the box.
        #
        # ``max_nbytes`` is stripped, NOT forwarded (2026-07-09 fix): ``parallel_kwargs``'s
        # ``max_nbytes=MAX_JOBLIB_NBYTES`` (1e3 bytes) is tuned for the ``backend="threading"``
        # branch, where the constructor's own comment (``_mrmr_class.py``) documents it as a
        # silently-ignored no-op. That no-op assumption does NOT hold here: this is a REAL loky
        # PROCESS backend, where joblib auto-memmaps every argument over ``max_nbytes`` bytes --
        # at a 1 KB bar that is ``nbins``/``classes_y``/``freqs_y`` on every dispatch, not just the
        # intentionally-memmapped multi-GB ``data`` matrix. Omitting the key lets joblib's own
        # built-in default (``'1M'``) govern instead, which still memmaps ``data`` (far past 1 MB)
        # but stops needlessly memmap-spilling every few-KB argument to a temp file.
        _extra_kwargs = {k: v for k, v in parallel_kwargs.items() if k not in ("backend", "max_nbytes")}
        _loky_cpu_backend = LokyBackend(
            inner_max_num_threads=1,
            initializer=disable_cuda_in_worker,
        )
        # OUTER WALL-CLOCK WATCHDOG, not just joblib's own ``timeout=`` (2026-07-10 fix). A fresh loky pool
        # spawns ``n_jobs`` brand-new Python processes, each re-importing mlframe/numba/its heavy deps and
        # JIT-compiling every njit kernel it touches from a COLD cache -- on Windows this has been observed
        # to hang INDEFINITELY (reproduced repeatedly: a 900s wait with zero progress, stuck in joblib's
        # ``_retrieve`` sleep-poll; likely a numba disk-cache write race, or a stuck process spawn, when
        # many freshly-spawned workers compile the SAME kernel concurrently for the first time). Passing
        # ``timeout=`` to ``Parallel(...)`` alone does NOT catch this: joblib's task-level timeout only
        # fires once a task has actually been DISPATCHED and registered in ``self._jobs`` -- a hang during
        # POOL SPAWN, before the first task is ever dispatched, is invisible to it (confirmed: the hang
        # recurred identically with ``timeout=300`` set on ``Parallel`` itself). Windows has no
        # ``signal.alarm``, so the only reliable bound is an OUTER watchdog: run the risky call on a daemon
        # thread and stop WAITING for it after ``_LOKY_POOL_WALL_CLOCK_TIMEOUT`` seconds, falling back to
        # the exact serial path immediately. The abandoned thread (and any loky worker processes it spawned)
        # may linger as orphans -- strictly better than blocking the whole fit indefinitely, which is the
        # exact "10h44m fit, weak CPU/GPU utilization" pathology the whole audit started from.
        _LOKY_POOL_WALL_CLOCK_TIMEOUT = 300

        def _run_loky_pair_mi_pool():
            return list(
                Parallel(
                    n_jobs=n_jobs,
                    backend=_loky_cpu_backend,
                    verbose=10 if verbose else 0,
                    timeout=_LOKY_POOL_WALL_CLOCK_TIMEOUT,
                    **_extra_kwargs,
                )(
                    delayed(compute_pairs_mis)(
                        all_pairs=chunk,
                        data=data,
                        target_indices=target_indices,
                        nbins=nbins,
                        classes_y=classes_y,
                        classes_y_safe=classes_y_safe,
                        freqs_y=freqs_y,
                        fe_min_nonzero_confidence=fe_min_nonzero_confidence,
                        fe_npermutations=fe_npermutations,
                        cached_confident_MIs=cached_confident_MIs,
                        cached_MIs=cached_MIs,
                        fe_min_pair_mi=fe_min_pair_mi,
                        fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
                    )
                    for chunk in _lazy_chunks(combinations(numeric_vars_to_consider, 2), chunk_size)
                )
            )

        try:
            import concurrent.futures

            _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            _future = _executor.submit(_run_loky_pair_mi_pool)
            try:
                dicts = _future.result(timeout=_LOKY_POOL_WALL_CLOCK_TIMEOUT)
            finally:
                # wait=False: never block here waiting for an already-abandoned (possibly hung-forever)
                # worker thread -- that would defeat the whole point of the watchdog.
                _executor.shutdown(wait=False)
            for next_dict in dicts:
                cached_MIs.update(next_dict)
        except Exception as _pool_exc:
            logger.warning(
                "MRMR FE: loky pair-MI pool failed or timed out (%s: %s); retrying via the batched "
                "CPU dispatcher before falling back to the slow per-pair path [n_pairs=%d, n_jobs=%d].",
                type(_pool_exc).__name__, _pool_exc, n_pairs, n_jobs,
            )
            # Retry via the batched dispatcher FORCED to CPU njit-parallel, not the legacy per-pair sweep
            # directly (2026-07-10 fix; reproduced live at n=3M/p=423, n_pairs=89,253: the loky pool
            # timing out and falling straight to ``compute_pairs_mis``'s serial per-pair ``mi_direct``
            # loop turned into a multi-hour, single-core slog -- the exact "weak CPU/GPU utilization"
            # pathology the whole audit traces back to). The batched kernel is 10-30x faster per pair
            # than the legacy permutation-test loop regardless of backend (see the finding-#21 fix
            # above); forcing ``njit_parallel`` here specifically avoids retrying on a GPU that may have
            # JUST failed (the loky pool failure and a GPU OOM earlier in the same fit are plausibly
            # correlated) while still using ALL CPU cores instead of one. Only pairs this ALSO can't
            # cover (a genuine kernel failure, vanishingly rare) fall through to the legacy per-pair path.
            try:
                from mlframe.feature_selection.filters.batch_pair_mi_gpu import dispatch_batch_pair_mi_chunked

                _retry_ids_arr = np.fromiter(numeric_vars_to_consider, dtype=np.int64, count=len(numeric_vars_to_consider))
                _retry_pair_a, _retry_pair_b, _retry_mi, _retry_backend_counts = dispatch_batch_pair_mi_chunked(
                    factors_data=data,
                    ids=_retry_ids_arr,
                    nbins=nbins,
                    classes_y=classes_y,
                    freqs_y=freqs_y,
                    force_backend="njit_parallel",
                )
                _retry_n = int(_retry_pair_a.shape[0])
                _retry_prefill_count = 0
                for _i in range(_retry_n):
                    _p = (int(_retry_pair_a[_i]), int(_retry_pair_b[_i]))
                    if _p not in cached_confident_MIs and _p not in cached_MIs:
                        cached_MIs[_p] = float(_retry_mi[_i])
                        _retry_prefill_count += 1
                logger.warning(
                    "MRMR FE: batched CPU retry covered %d/%d pair MIs via [%s] backend chunk(s) after "
                    "the loky pool failure.",
                    _retry_prefill_count, _retry_n,
                    ", ".join(f"{k}={v}" for k, v in sorted(_retry_backend_counts.items())),
                )
            except Exception as _retry_exc:
                logger.warning(
                    "MRMR FE: batched CPU retry ALSO failed (%s: %s); falling back to the slow serial "
                    "per-pair path for whatever pairs remain uncached [n_pairs=%d].",
                    type(_retry_exc).__name__, _retry_exc, n_pairs,
                )
            compute_pairs_mis(
                all_pairs=tqdmu(
                    combinations(numeric_vars_to_consider, 2),
                    total=n_pairs,
                    desc="getting pairs MIs (serial fallback)",
                    leave=False,
                    mininterval=5,
                    disable=not verbose,
                ),
                data=data,
                target_indices=target_indices,
                nbins=nbins,
                classes_y=classes_y,
                classes_y_safe=classes_y_safe,
                freqs_y=freqs_y,
                fe_min_nonzero_confidence=fe_min_nonzero_confidence,
                fe_npermutations=fe_npermutations,
                cached_confident_MIs=cached_confident_MIs,
                cached_MIs=cached_MIs,
                fe_min_pair_mi=fe_min_pair_mi,
                fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
            )
    record_fe_family_wall("pairwise_mi_legacy_sweep", perf_counter() - _legacy_sweep_t0)

    # ---------------------------------------------------------------------------------------------------------------
    # ORDER-2 Westfall-Young maxT permutation-null floor on the PROSPECTIVE-PAIR
    # JOINT MI (2026-06-03). The gating loop below ranks O(p^2) candidate pairs by
    # JOINT MI(x_i, x_j; y); at high p the MAX joint MI over PURE-NOISE pairs is a
    # positive order statistic that grows with the pool size -- the same best-of-p
    # selection bias the order-1 screening floor rejects, now at order 2. The
    # per-pair prevalence gates (``fe_min_pair_mi_prevalence`` /
    # ``fe_synergy_min_prevalence``) are PER-PAIR; they do NOT account for the
    # max-over-pool selection, so a wide noise matrix still surfaces
    # "synergistic-looking" noise pairs. Compute the floor ONCE here over the WHOLE
    # candidate pool: shuffle the discretised target K times, take the per-shuffle
    # MAX joint MI via the SAME batched plug-in estimator the screen scores
    # ``pair_mi`` with, floor at the q-th quantile. Applied IN ADDITION to the
    # prevalence gates in BOTH the zero-individual-MI (XOR) branch and the uplift
    # branch below. SELF-GATING: below ``fe_pair_maxt_min_pairs`` candidate pairs
    # the floor is 0.0 (no-op => byte-identical narrow pools), mirroring
    # ``screen_fdr_min_features``. ``fe_pair_maxt_null_permutations=0`` disables.
    _maxt_floor_t0 = perf_counter()
    _pair_maxt_floor, _pair_mm_bias = compute_pair_maxt_floor(
        self,
        numeric_vars_to_consider=numeric_vars_to_consider,
        n_pairs=n_pairs,
        data=data,
        nbins=nbins,
        classes_y=classes_y,
        freqs_y=freqs_y,
        verbose=verbose,
    )
    record_fe_family_wall("pair_maxt_null_floor", perf_counter() - _maxt_floor_t0)

    # "auto" prevalence debias needs the per-pair MM joint-MI bias; compute_pair_maxt_floor only
    # populates it when ``fe_mm_debias_prevalence`` is on, so fill it here (analytic, no shuffles)
    # when "auto" is active and the maxT path left it empty. On failure fall back to the fixed bar
    # (degrade to the proven default rather than risk a wrong gate).
    if _prevalence_debias_auto and not _pair_mm_bias:
        try:
            from .._permutation_null import pairwise_mm_joint_bias
            _auto_pairs = list(combinations(numeric_vars_to_consider, 2))
            if _auto_pairs:
                _auto_pa = np.fromiter((p[0] for p in _auto_pairs), dtype=np.int64, count=len(_auto_pairs))
                _auto_pb = np.fromiter((p[1] for p in _auto_pairs), dtype=np.int64, count=len(_auto_pairs))
                _auto_ky = int(np.asarray(freqs_y).shape[0])
                _auto_bias = pairwise_mm_joint_bias(data, _auto_pa, _auto_pb, nbins, _auto_ky)
                # UNDER-SAMPLE GUARD (bias-variance, 2026-06-13): the MM bias (k_joint-1)(k_y-1)/2n is
                # only a reliable correction when the joint table is adequately occupied. At tiny n the
                # bias is large/noisy and over-tightening the prevalence gate feeds the synergy-rescue
                # path, which can ADMIT worse features (measured: F2 n=2500 0.917 -> 1.079, but n=8000
                # unchanged). So skip the debias (bias -> 0, raw pair_mi) for any pair whose rows-per-
                # occupied-joint-cell falls below ``fe_confirm_undersample_rows_per_cell`` (default 5),
                # mirroring the existing CMI-fallback rule. This FORGOES the (unreliable) tiny-n win
                # rather than risk the tiny-n harm -- the large-n win (bilinear n=8000 0.195 -> 0.052)
                # is preserved because there rows-per-cell clears the floor. k_joint is recovered from
                # the bias: k_joint = 1 + bias*2n/(k_y-1).
                _auto_n = int(data.shape[0])
                _auto_min_rpc = float(getattr(self, "fe_confirm_undersample_rows_per_cell", 5.0))
                for _api, _apr in enumerate(_auto_pairs):
                    _b = float(_auto_bias[_api])
                    if _b > 0.0 and _auto_ky > 1:
                        _kj = 1.0 + (_b * 2.0 * _auto_n) / float(_auto_ky - 1)
                        _rpc = _auto_n / max(1.0, _kj * _auto_ky)
                        if _rpc < _auto_min_rpc:
                            _b = 0.0  # under-sampled joint -> unreliable bias -> use raw pair_mi
                    _pair_mm_bias[tuple(sorted(_apr))] = _b
        except Exception:
            _prevalence_debias_auto = False

    return numeric_vars_to_consider, _eng_cap, _pair_maxt_floor, _pair_mm_bias, _prevalence_debias_auto
