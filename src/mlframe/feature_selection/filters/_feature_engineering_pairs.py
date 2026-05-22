"""``check_prospective_fe_pairs`` carved out of
``mlframe.feature_selection.filters.feature_engineering``.

Re-imported at the parent's module bottom so historical
``from mlframe.feature_selection.filters.feature_engineering import check_prospective_fe_pairs``
resolves transparently.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from itertools import combinations
from timeit import default_timer as timer
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermval
from scipy import special as sp

from pyutilz.pythonlib import sort_dict_by_value
from pyutilz.system import tqdmu


# Wave 27 P1 (2026-05-20): ``check_prospective_fe_pairs`` is dispatched via
# ``parallel_run`` from mrmr.py with backend='threading'. The function
# accumulates per-binary-transform timings into a shared ``times_spent``
# defaultdict via ``+=``. Python's ``+=`` on a float is load-add-store and
# NOT atomic even under the GIL between threads; concurrent workers can
# drop updates silently, under-reporting the diagnostic at mrmr.py:1691.
# This module-level lock serialises the increment; threading workers
# synchronise correctly. Under loky/spawn each worker gets its own
# defaultdict copy (no shared state); the lock has no effect there but
# also doesn't break.
_TIMES_SPENT_LOCK = threading.Lock()

# CRITICAL #2 (2026-05-21): the hoisted shared buffer at
# ``check_prospective_fe_pairs`` allocates ``(n, max_n_combs * len(binary))``
# float32. With n=4M and the medium preset that's ~17.6 GiB -- production
# MRMR crashed with numpy.core._exceptions._ArrayMemoryError on a TVT run.
# The hoist landed in Wave Pack G (commit 068acdd) under small-n benchmarks
# and never measured peak RAM on million-row data.
#
# Two-strategy dispatch:
#   Fast path (current): if buffer < ``_FE_BUFFER_RAM_BUDGET_RATIO`` * available
#     RAM, allocate the shared buffer and use the hoist (cheapest if it fits).
#   Recompute fallback: drop the multi-column buffer, scratch into a fresh 1D
#     ``np.empty(n, float32)`` per inner iteration, and rebuild the ~10
#     survivor columns from their (transformations_pair, bin_func_name) metadata
#     after the inner loop. Extra recompute cost: ~K bin_func calls per pair
#     (K = num survivors, typically <= fe_max_pair_features + |leading|);
#     <= 1% of the ~max_combs*|binary| calls already done in the inner loop.
#
# Subsample path remains a separate opt-in (``subsample_n`` parameter); this
# memory dispatcher is the deterministic, accuracy-preserving fallback that
# auto-engages when the shared buffer would OOM.
_FE_BUFFER_RAM_BUDGET_RATIO: float = 0.4

# Shared subsample default across the two FE entry points. ``polynom_pair_fe``
# already uses 200_000 (validated 2026-05-18: 100k could lose a marginal hermite
# feature, 200k kept it). The accuracy bench for ``check_prospective_fe_pairs``
# at this n landed at jaccard=1.0 vs full -- see
# bench_fe_pair_subsample_accuracy.py. Keep both call sites pinned to ONE knob
# so a future re-tune lands consistently across the FE block.
FE_DEFAULT_SUBSAMPLE_N: int = 200_000




def check_prospective_fe_pairs(
    prospective_pairs,
    X,
    unary_transformations,
    binary_transformations,
    classes_y,
    classes_y_safe,
    freqs_y,
    num_fs_steps,
    cols,
    original_cols,
    fe_max_steps,
    fe_npermutations,
    fe_max_pair_features,
    fe_print_best_mis_only,
    fe_min_nonzero_confidence,
    fe_min_engineered_mi_prevalence,
    fe_good_to_best_feature_mi_threshold,
    fe_max_external_validation_factors,
    numeric_vars_to_consider,
    quantization_nbins,
    quantization_method,
    quantization_dtype,
    times_spent,
    verbose,
    # CRITICAL #2 follow-up (2026-05-21): subsample rows for the MI sweep so the
    # transformed_vars + shared_buffer allocations scale with subsample_n rather
    # than the (possibly multi-million-row) full X. Survivor columns are still
    # produced at full n via _rebuild_full_survivor_col so the caller contract
    # is preserved. Default 200_000 matches polynom_pair_fe's existing knob
    # (FE_DEFAULT_SUBSAMPLE_N); the standalone accuracy bench in
    # bench_fe_pair_subsample_accuracy.py shows jaccard=1.0 vs full at n_eff
    # >= 50k on synthetic 3-pair-competition data. 0 = use full data (legacy).
    subsample_n: int = FE_DEFAULT_SUBSAMPLE_N,
    subsample_seed: int = 42,
):
    # Starting from the most heavily connected pairs, create a big pool of original features + their unary transforms. Individual vars referenced more than once go
    # to the global pool, the rest to the local (not stored)?

    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .feature_engineering import FE_DEFAULT_SUBSAMPLE_N, _FE_BUFFER_RAM_BUDGET_RATIO, _can_hoist_shared_buffer, _estimate_fe_shared_buffer_bytes, _rebuild_full_survivor_col, discretize_array, get_new_feature_name, gpu_compatible_unary_names, logger, mi_direct
    res = {}

    # SUBSAMPLE-SETUP: when caller asks for subsample_n > 0 AND len(X) exceeds it,
    # build subsampled views of X / classes_y / classes_y_safe / freqs_y. The MI
    # sweep operates on these views; survivor packing always rebuilds at full n.
    # When subsample_n is 0 / negative / >= len(X) the legacy full-data path runs
    # unchanged (everything below uses ``X`` / ``classes_y`` / ... directly).
    _X_full = X
    _full_n_rows = len(_X_full)
    _use_subsample = isinstance(subsample_n, int) and 0 < subsample_n < _full_n_rows
    if _use_subsample:
        _rng_sub = np.random.default_rng(int(subsample_seed))
        _sample_idx = np.sort(_rng_sub.choice(_full_n_rows, size=int(subsample_n), replace=False))
        if isinstance(_X_full, pd.DataFrame):
            X = _X_full.iloc[_sample_idx].reset_index(drop=True)
        else:
            # Polars path -- row indexing returns a fresh frame; preserves zero-copy where possible.
            X = _X_full[_sample_idx]
        # Realign per-row target encodings; recompute freqs from the subsampled
        # class labels so MI estimates use the actual subsample distribution
        # rather than the full-n freq table (which would bias the MI estimator
        # toward classes that shrank under the random subset).
        _cy = np.asarray(classes_y)
        _cy_safe = np.asarray(classes_y_safe)
        classes_y = _cy[_sample_idx]
        classes_y_safe = _cy_safe[_sample_idx]
        # Recompute freqs from subsampled class labels. merge_vars returns
        # freqs_y as a FLOAT proportions array (sum=1.0), not raw counts; the
        # subsample needs the same shape. bincount gives counts -> divide by
        # total to get proportions matching the caller's expectation.
        if classes_y.size > 0 and classes_y.dtype.kind in ("i", "u"):
            _counts = np.bincount(classes_y.astype(np.int64))
            _total = _counts.sum()
            if _total > 0:
                freqs_y = (_counts.astype(np.float64) / float(_total))
        # else: leave the caller-supplied freqs_y; mi_direct handles its own
        # validation and would crash anyway on a non-integer class table.
        if verbose:
            logger.info(
                "check_prospective_fe_pairs: subsample_n=%d active (full_n=%d, %.1f%% sample); "
                "MI sweep runs on the subsample, survivor columns rebuilt at full n.",
                int(subsample_n), _full_n_rows, 100.0 * subsample_n / _full_n_rows,
            )

    # Exact preallocation. ``n_pairs * n_unary * 2`` over-counts because (var, tr_name) keys are de-duplicated in ``vars_transformations``; the unique-key set is the
    # true upper bound.
    unique_keys: set = set()
    for (raw_vars_pair, _), _ in prospective_pairs.items():
        for var in raw_vars_pair:
            for tr_name in unary_transformations.keys():
                unique_keys.add((var, tr_name))

    if verbose >= 2:
        logger.info(
            "Creating a pool of %d unary transformations for feature engineering "
            "(legacy upper bound was %d).",
            len(unique_keys),
            len(prospective_pairs) * len(unary_transformations) * 2,
        )

    transformed_vars = np.empty(shape=(len(X), len(unique_keys)), dtype=np.float32)

    # Hoist ``final_transformed_vals`` outside the per-pair loop: precompute each pair's ``combs``, find the max length, allocate one shared buffer. Each pair writes
    # then reads the same ``[:, i]`` slice so stale tail data is never observed.
    pair_combs: dict = {}
    max_n_combs = 0
    for (raw_vars_pair, _), _ in prospective_pairs.items():
        combs = list(
            combinations(
                [(raw_vars_pair[0], k) for k in unary_transformations.keys()]
                + [(raw_vars_pair[1], k) for k in unary_transformations.keys()],
                2,
            )
        )
        combs = [tp for tp in combs if tp[0][0] != tp[1][0]]
        pair_combs[raw_vars_pair] = combs
        if len(combs) > max_n_combs:
            max_n_combs = len(combs)

    # CRITICAL #2 (2026-05-21): memory-aware dispatch. The full hoisted buffer is the fast path
    # but on n=4M with medium preset this lands at ~17.6 GiB and crashes the suite. Estimate the
    # required buffer, check psutil.virtual_memory().available, and either keep the buffer (fast)
    # or set it to None and switch to recompute-from-metadata in the inner loop and survivor
    # rebuild stages (memory-safe; ~1% extra bin_func calls per pair).
    _n_binary = len(binary_transformations)
    final_transformed_vals_shared = None
    if max_n_combs > 0:
        _buf_bytes = _estimate_fe_shared_buffer_bytes(len(X), max_n_combs, _n_binary)
        _can_hoist, _bb, _avail = _can_hoist_shared_buffer(_buf_bytes)
        if _can_hoist:
            try:
                final_transformed_vals_shared = np.empty(
                    shape=(len(X), max_n_combs * _n_binary),
                    dtype=np.float32,
                )
            except MemoryError:
                # psutil over-reported available; falling back stays safe.
                final_transformed_vals_shared = None
                if verbose:
                    logger.warning(
                        "check_prospective_fe_pairs: shared buffer (%.1f GiB) allocation raised "
                        "MemoryError despite passing the available-RAM check (%.1f GiB available, "
                        "%.0f%% budget); switching to recompute-from-metadata fallback (~1%% extra "
                        "bin_func calls per pair).",
                        _bb / 2**30, _avail / 2**30 if _avail >= 0 else float("nan"),
                        _FE_BUFFER_RAM_BUDGET_RATIO * 100.0,
                    )
        else:
            if verbose:
                logger.warning(
                    "check_prospective_fe_pairs: shared buffer would need %.1f GiB but only %.1f GiB "
                    "RAM is available (%.0f%% budget = %.1f GiB cap); using recompute-from-metadata "
                    "fallback path (~1%% extra bin_func calls per pair, identical survivors). To force "
                    "the fast path either free RAM or raise _FE_BUFFER_RAM_BUDGET_RATIO; to bound "
                    "compute, pass subsample_n>0 from the MRMR config.",
                    _bb / 2**30, _avail / 2**30 if _avail >= 0 else float("nan"),
                    _FE_BUFFER_RAM_BUDGET_RATIO * 100.0,
                    (_avail * _FE_BUFFER_RAM_BUDGET_RATIO) / 2**30 if _avail >= 0 else float("nan"),
                )
    # In the recompute-fallback path we need to look up the
    # ``(transformations_pair, bin_func_name)`` for any index ``i`` that
    # was assigned in the inner loop, so the validation + survivor-packing
    # phases can rebuild the column on demand. The shared-buffer path
    # ignores this dict; either way it is bounded by ``max_n_combs *
    # _n_binary`` lightweight tuples per pair.
    _need_recompute_map = final_transformed_vals_shared is None

    vars_transformations = {}
    i = 0
    for (raw_vars_pair, _pair_mi), _uplift in prospective_pairs.items():
        for var in raw_vars_pair:
            # ``original_cols`` is built only for cols that survived the prior
            # selection pass; a temp / dropped column index may not be present.
            # Skip silently rather than KeyError out of the whole FE block.
            if var not in original_cols:
                continue
            # Polars vs pandas int-column indexing: ``X[:, idx].to_numpy()`` (polars, zero-copy for numerics) vs ``X.iloc[:, idx].values`` (pandas).
            if isinstance(X, pd.DataFrame):
                vals = X.iloc[:, original_cols[var]].values
            else:
                vals = X[:, original_cols[var]].to_numpy()
            for tr_name, tr_func in unary_transformations.items():
                key = (var, tr_name)
                if key not in vars_transformations:
                    try:
                        if "poly_" in tr_name:
                            transformed_vars[:, i] = hermval(vals, c=tr_func)
                        else:
                            # WAVE 5 (1/4): if CUDA is available, the
                            # transformation is GPU-compatible, AND the
                            # column is large enough to amortise the H2D
                            # + D2H round trip, run the elementwise op on
                            # GPU via cupy. Threshold matches
                            # ``discretize_2d_array_cuda``'s breakeven; on
                            # cc 6.1 ~500k cells. Wave 23 P1 fix (2026-05-20):
                            # consult kernel_tuning_cache so the crossover
                            # adapts to live HW; fall back to 500_000.
                            _gpu_used = False
                            try:
                                from pyutilz.system.kernel_tuning_cache import KernelTuningCache
                                _cache = KernelTuningCache.load_or_create()
                                _e = _cache.lookup(
                                    "unary_elementwise",
                                    n_samples=int(vals.size),
                                )
                                _min_cells = int(_e["min_cells"]) if _e and "min_cells" in _e else 500_000
                            except Exception:
                                _min_cells = 500_000
                            if (
                                vals.size >= _min_cells
                                and tr_name in gpu_compatible_unary_names()
                            ):
                                try:
                                    from pyutilz.core.pythonlib import is_cuda_available
                                    if is_cuda_available():
                                        import cupy as cp
                                        _cp_fn = getattr(cp, tr_name, None)
                                        if _cp_fn is not None:
                                            d_vals = cp.asarray(vals)
                                            d_res = _cp_fn(d_vals)
                                            transformed_vars[:, i] = cp.asnumpy(d_res)
                                            _gpu_used = True
                                except Exception:
                                    _gpu_used = False  # fall through to CPU
                            if not _gpu_used:
                                transformed_vars[:, i] = tr_func(vals)
                    except Exception as e:
                        # ``np.isnan`` / ``np.isinf`` / ``np.nanmin`` only work on float dtypes. When ``vals`` is object/string (e.g. a polars Utf8 cat column not encoded
                        # before reaching FE), calling them inside the error-log formatter itself raises -- masking the real transformation error and aborting MRMR
                        # entirely. Compute numeric-only diagnostics conditionally.
                        if np.issubdtype(vals.dtype, np.floating):
                            _diag = (
                                f", isnan={np.isnan(vals).sum()}, "
                                f"isinf={np.isinf(vals).sum()}, nanmin={np.nanmin(vals)}"
                            )
                        else:
                            _diag = f", dtype={vals.dtype} (numeric diagnostics skipped)"
                        logger.error(
                            f"Error when performing {tr_name} on array {vals[:5]}, "
                            f"var={cols[var]}: {str(e)}{_diag}"
                        )
                    else:
                        vars_transformations[key] = i
                        i += 1

    if verbose >= 2:
        logger.info("Created. For every pair from the pool, trying all known functions...")

    # For every pair from the pool, try all known functions of 2 variables (not storing results in persistent RAM). Record best pairs.
    for (
        raw_vars_pair,
        pair_mi,
    ), _uplift in tqdmu(
        prospective_pairs.items(), desc="pair", leave=False
    ):  # better to start considering form the most prospective pairs with highest mis ratio!

        messages = []

        combs = pair_combs[raw_vars_pair]

        best_config, best_mi = None, -1
        this_pair_features = set()
        var_pairs_perf = {}

        # CRITICAL #2 dispatch: hoist path uses the shared buffer (writes into
        # ``[:, i]``); recompute-fallback path uses a tiny 1D scratch + a
        # config-by-i map for on-demand survivor recomputation later.
        final_transformed_vals = final_transformed_vals_shared
        _col_buf_1d: np.ndarray | None = (
            np.empty(len(X), dtype=np.float32) if _need_recompute_map else None
        )
        _config_by_i: dict[int, tuple] = {} if _need_recompute_map else None

        i = 0
        for transformations_pair in combs:
            if (transformations_pair[0] not in vars_transformations) or (transformations_pair[1] not in vars_transformations):
                continue
            param_a = transformed_vars[:, vars_transformations[transformations_pair[0]]]
            param_b = transformed_vars[:, vars_transformations[transformations_pair[1]]]

            for bin_func_name, bin_func in binary_transformations.items():

                start = timer()
                try:
                    # with np.errstate(invalid='ignore'):
                    if final_transformed_vals is not None:
                        final_transformed_vals[:, i] = bin_func(param_a, param_b)
                        _col_view = final_transformed_vals[:, i]
                    else:
                        # Recompute fallback: write into the shared 1D scratch.
                        # bin_func returns a fresh ndarray; copy into the scratch
                        # so downstream nan_to_num + discretize see contiguous
                        # data. Avoids accumulating one alloc per inner iter.
                        _col_buf_1d[:] = bin_func(param_a, param_b)
                        _col_view = _col_buf_1d
                except Exception:
                    logger.error(f"Error when performing {bin_func}")
                else:
                    np.nan_to_num(_col_view, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                    # Wave 27 P1: serialise the increment under
                    # ``_TIMES_SPENT_LOCK``; pre-fix `+=` raced on the
                    # shared defaultdict under the parallel threading
                    # dispatch from mrmr.py.
                    with _TIMES_SPENT_LOCK:
                        times_spent[bin_func_name] += timer() - start

                    discretized_transformed_values = discretize_array(
                        arr=_col_view, n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype
                    )
                    fe_mi, fe_conf = mi_direct(
                        discretized_transformed_values.reshape(-1, 1),
                        x=np.array([0], dtype=np.int64),
                        y=None,
                        factors_nbins=np.array([quantization_nbins], dtype=np.int64),
                        classes_y=classes_y,
                        classes_y_safe=classes_y_safe,
                        freqs_y=freqs_y,
                        min_nonzero_confidence=fe_min_nonzero_confidence,
                        npermutations=fe_npermutations,
                    )

                    config = (transformations_pair, bin_func_name, i)
                    var_pairs_perf[config] = fe_mi
                    if _need_recompute_map:
                        # Map i -> (a_key, b_key, bin_func_name) for downstream
                        # rebuild; bin_func is looked up via the original dict.
                        _config_by_i[i] = (transformations_pair[0], transformations_pair[1], bin_func_name)

                    if fe_mi > best_mi:
                        best_mi = fe_mi
                        best_config = config
                    if fe_mi > best_mi * 0.85:
                        if not fe_print_best_mis_only or (fe_mi == best_mi):
                            if verbose > 2:
                                print(f"MI of transformed pair {bin_func_name}({transformations_pair})={fe_mi:.4f}, MI of the plain pair {pair_mi:.4f}")
                    i += 1

        if verbose > 2:
            print(f"For pair {raw_vars_pair}, best config is {best_config} with best mi= {best_mi}")

        if best_mi / pair_mi > fe_min_engineered_mi_prevalence * (1.0 if num_fs_steps < 1 else 1.025):  # Best transformation is good enough

            # If there is a group of leaders with almost the same performance, approve them through one of the other variables.
            # если будут возникать такие группы примерно одинаковых по силе лидеров, их придётся разрешать с помощью одного из других влияющих факторов
            leading_features = []
            for next_config, next_mi in sort_dict_by_value(var_pairs_perf).items():
                if next_mi > best_mi * fe_good_to_best_feature_mi_threshold:
                    leading_features.append(next_config)

            if len(leading_features) > 1:
                if len(numeric_vars_to_consider) > 2:

                    if verbose > 2:
                        print(f"Taking {len(leading_features)} new features for a separate validation step!")

                    # Test all candidates as-is against the rest of the approved factors (also as-is). Candidates significantly outstanding (in terms of MI with target)
                    # against any other approved factor are kept.
                    valid_pairs_perf = {}

                    for transformations_pair, bin_func_name, i in leading_features:
                        if final_transformed_vals is not None:
                            param_a = final_transformed_vals[:, i]
                        else:
                            # CRITICAL #2 recompute-fallback: rebuild the survivor column from its
                            # (a_key, b_key, bin_func_name) metadata. transformed_vars is small
                            # (deduped unary table); the bin_func call is cheap (one ufunc).
                            _a_key, _b_key, _bin_name = _config_by_i[i]
                            _pa = transformed_vars[:, vars_transformations[_a_key]]
                            _pb = transformed_vars[:, vars_transformations[_b_key]]
                            param_a = binary_transformations[_bin_name](_pa, _pb)
                            np.nan_to_num(param_a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                        best_valid_mi = -1
                        config = (transformations_pair, bin_func_name, i)

                        external_factors = list(set(numeric_vars_to_consider) - set(raw_vars_pair))
                        if fe_max_external_validation_factors and len(external_factors) > fe_max_external_validation_factors:
                            external_factors = np.random.choice(external_factors, fe_max_external_validation_factors)

                        for external_factor in tqdmu(external_factors, desc="external validation factor", leave=False):
                            if external_factor not in original_cols:
                                continue
                            if isinstance(X, pd.DataFrame):
                                param_b = X.iloc[:, original_cols[external_factor]].values
                            else:
                                param_b = X[:, original_cols[external_factor]].to_numpy()

                            for valid_bin_func_name, valid_bin_func in binary_transformations.items():

                                valid_vals = valid_bin_func(param_a, param_b)

                                discretized_transformed_values = discretize_array(
                                    arr=valid_vals, n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype
                                )
                                fe_mi, fe_conf = mi_direct(
                                    discretized_transformed_values.reshape(-1, 1),
                                    x=np.array([0], dtype=np.int64),
                                    y=None,
                                    factors_nbins=np.array([quantization_nbins], dtype=np.int64),
                                    classes_y=classes_y,
                                    classes_y_safe=classes_y_safe,
                                    freqs_y=freqs_y,
                                    min_nonzero_confidence=fe_min_nonzero_confidence,
                                    npermutations=fe_npermutations,
                                )

                                if fe_mi > best_valid_mi:
                                    best_valid_mi = fe_mi
                                    if verbose > 2:
                                        print(
                                            f"MI of transformed pair {valid_bin_func_name}({(transformations_pair,bin_func_name)} with ext factor {external_factor})={fe_mi:.4f}"
                                        )

                        valid_pairs_perf[config] = best_valid_mi

                    # Recommend proceeding with top N best transformations.
                    for j, (config, _valid_mi) in enumerate(sort_dict_by_value(valid_pairs_perf, reverse=True).items()):
                        if j < fe_max_pair_features:
                            new_feature_name = get_new_feature_name(fe_tuple=config, cols_names=cols)
                            if verbose:
                                messages.append(
                                    f"{new_feature_name} is recommended to use as a new feature! (won in validation with other factors) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
                                )
                            this_pair_features.add((config, j))
                        else:
                            break
                else:
                    if verbose:
                        messages.append(
                            f"{len(leading_features)} are recommended to use as new features! (can't narrow down the list by validation with other factors) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
                        )
                    for j, config in enumerate(leading_features):
                        if j < fe_max_pair_features:
                            this_pair_features.add((config, j))
            else:
                new_feature_name = get_new_feature_name(fe_tuple=best_config, cols_names=cols)
                if verbose:
                    messages.append(
                        f"{new_feature_name} is recommended to use as a new feature! (clear winner) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
                    )
                j = 0
                this_pair_features.add((best_config, j))

            transformed_vals, new_cols, new_nbins = None, None, None

            if this_pair_features:

                # Bulk add the found & checked best features.
                # ``this_pair_features`` is a SET of (config, j) tuples
                # with sparse, non-contiguous ``j`` indices into
                # ``final_transformed_vals``. The consumer (mrmr.py
                # ``_run_fe_step``) iterates
                # ``for k in range(len(this_pair_features)):
                # transformed_vals[:, k]``, so the buffer MUST have
                # exactly ``len(this_pair_features)`` columns packed
                # densely 0..N-1, not the sparse ``j``-indexed layout
                # with holes. Pre-fix code wrote to ``transformed_vals[:, j]``
                # then sliced to ``[:, :last_j + 1]`` -- this gives
                # either a too-short buffer (if last_j was small) and
                # IndexError downstream, or holes (if last_j was large).
                # Pack each (config, j) into a compact column index
                # ``idx = 0..len(this_pair_features)-1`` instead.
                if fe_max_steps > 1:
                    # Survivor buffer is sized to the FULL X regardless of subsample mode --
                    # mrmr.py at the consumer side allocates a full-n ``data`` append, so
                    # caller contract requires ``transformed_vals.shape[0] == len(_X_full)``.
                    transformed_vals = np.empty(shape=(_full_n_rows, len(this_pair_features)), dtype=quantization_dtype)
                new_nbins = []
                new_cols = []

                for idx, (config, j) in enumerate(this_pair_features):
                    new_feature_name = get_new_feature_name(fe_tuple=config, cols_names=cols)
                    transformations_pair, bin_func_name, i = config

                    if fe_max_steps > 1:
                        if _use_subsample:
                            # SUBSAMPLE path: rebuild from raw _X_full so the survivor column
                            # carries the FULL n rows the caller expects (mrmr.py appends it
                            # back to its full-n ``data`` array). The MI sweep used a 200k
                            # subset; the survivor IDENTITIES are correct (bench shows
                            # jaccard=1.0 vs full-n at n_eff>=50k), so we just need to
                            # rematerialise the values at full resolution.
                            transformed_vals[:, idx] = _rebuild_full_survivor_col(
                                config, _X_full, original_cols,
                                unary_transformations, binary_transformations,
                            )
                        elif final_transformed_vals is not None:
                            transformed_vals[:, idx] = final_transformed_vals[:, i]
                        else:
                            # CRITICAL #2 recompute-fallback (no subsample, tight RAM): rebuild
                            # the survivor column from its (a_key, b_key, bin_func_name)
                            # metadata via the cached unary table. transformed_vars is at
                            # full n in this path so the column lands at full n directly.
                            _a_key, _b_key, _bin_name = _config_by_i[i]
                            _pa = transformed_vars[:, vars_transformations[_a_key]]
                            _pb = transformed_vars[:, vars_transformations[_b_key]]
                            _col = binary_transformations[_bin_name](_pa, _pb)
                            np.nan_to_num(_col, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                            transformed_vals[:, idx] = _col
                        new_nbins += [quantization_nbins]
                    new_cols += [new_feature_name]

            res[raw_vars_pair] = (this_pair_features, transformed_vals, new_cols, new_nbins, messages)

    return res
