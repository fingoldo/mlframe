# fs_filters_mrmr

Files reviewed: 22 read in depth (14,621 LOC) + 6 structured greps across all ~250 files of `filters/`

## Summary

The headline documented-open bug -- `_binned_numeric_agg_fe.py::_derive_cell_stats` per-cell skew/kurt on the
raw-power binomial expansion -- is **already fixed** in this tree, verified on both the host and the GPU-resident
twin. A repo-wide sweep for the same shape found no remaining occurrence in this cluster.

The cluster's own P1 is elsewhere and is the same shape as a documented past regression: `_kernel_tuning.py`
latches a permanent process-wide downgrade of all 268 kernel-tuning-cache dispatch sites on a single transient
failure, logged only at `debug`.

## Findings

### FS_FILTERS_MRMR-1 [P1] silent-backend-downgrade

**File:** `src/mlframe/feature_selection/filters/_kernel_tuning.py` :93-99

**Summary:** A broad `except Exception` around `KernelTuningCache()` construction permanently latches the
process-wide singleton to `False` and logs only at `debug`, so one transient fault downgrades every KTC-gated
dispatcher in the cluster to its hand-tuned fallback for the rest of the process.

**Failure scenario:** `KernelTuningCache()` runs `_load` -> `_build_provenance` -> `gpu_capability_summary` -> an
`nvidia-smi` SUBPROCESS (this module's own docstring, :3-9). That spawn is exactly the operation this repo's
notes describe failing transiently under multi-process GPU contention, and on Windows a concurrently-written
tuning JSON can raise `PermissionError`/`OSError` on read. Neither is `ImportError` -- :89 already handles the
one genuine "pyutilz absent" case -- so both land in :93, set `_CACHE_SINGLETON = False` (:98) and return None.
Nothing clears it except `_reset_for_tests()`. Every later `get_kernel_tuning_cache()` returns None at :79
without retrying, and the 268 call sites that consult it each take their hardcoded fallback. A 600k-row fit
loses the KTC-gated parallel Fisher-Yates shuffle-gen (measured 2.7x on that stage, ~88% of the order-1 maxT
floor's wall) and the resident GPU permutation-null floor, with nothing above `debug` explaining it. Same shape
as the documented `_select_mi_backend` regression, where a transient device fault at import permanently pinned
the ~100x-slower sklearn MI path.

**Suggested fix:** Mirror the `_select_mi_backend` fix. Keep the `except ImportError` latch at :89. Change :93 to
`logger.warning(...)` naming the exception, and do NOT latch `_CACHE_SINGLETON = False` for a non-`ImportError`
-- leave it None with a bounded retry counter, so a transient subprocess or file fault does not cost the whole
fit.

**Evidence:** :78-100 is the entire resolution path; `_CACHE_SINGLETON` is module-global and reset only by
`_reset_for_tests` (:103-109). :89 proves the author already distinguished the genuinely-unavailable case, so
:93 is by construction the unexpected-failure branch. Cluster-wide count of KTC consumers: 268.

**Disposition:** RESOLVED in the P0 pass, same file and handler as XCUT_SWALLOWED_FAILURES-1: `_kernel_tuning.py` uses a bounded `_MAX_INIT_ATTEMPTS = 3` retry plus a `logger.warning`, so one corrupt read no longer latches the singleton off for the process.

### FS_FILTERS_MRMR-2 [P2] silent-backend-downgrade

**File:** `src/mlframe/feature_selection/filters/_kernel_tuning.py` :53, :62-65

**Summary:** `_register_default_tuning_cache` sets `_DEFAULTS_REGISTERED = True` BEFORE attempting registration,
commented "never re-attempt, even on failure", and logs a failure only at `debug`.

**Failure scenario:** Runs at import of `filters/__init__` (:117). If `register_default_cache` raises -- the 17KB
`default_kernel_tuning.json` being rewritten by a concurrent sweep, or a Windows `PermissionError` -- :65 logs at
`debug`, the flag stays True, and every KTC lookup for the rest of the process misses the shipped
measurement-derived defaults and falls through to hand heuristics. That is the exact regression the file exists
to prevent, with no operator-visible signal.

**Suggested fix:** Move `_DEFAULTS_REGISTERED = True` to AFTER a successful `register_default_cache(...)`, and
promote :64-65 to `logger.warning`. Keep the `except ImportError` at :59 silent.

**Evidence:** :53 `_DEFAULTS_REGISTERED = True  # never re-attempt, even on failure` precedes the `try` at :57;
:64-65 is `except Exception` + `logger.debug`.

**Disposition:** RESOLVED as suggested. `_DEFAULTS_REGISTERED` is now set on the SUCCESS path (an `else:` on the try) and on the two genuinely-permanent cases (no file, no package); a `register_default_cache` failure warns, names the consequence, and leaves the flag unset so a later call re-attempts. The failures it catches are transient by nature -- the 17KB JSON being rewritten by a concurrent sweep, a Windows PermissionError -- which is exactly why refusing to retry was the wrong response. `tests/feature_selection/test_fs_filters_target_cast_fingerprints_and_pads.py`.

### FS_FILTERS_MRMR-3 [P2] perf-primitive-not-wired

**File:** `src/mlframe/feature_selection/filters/_conditional_gate_fe.py` :340-341

**Summary:** `_perm_null_hi` still runs the unbatched per-permutation Python loop (12 separate `_mi()` calls)
while both sibling copies were ported to a single batched `_mi_classif_batch`.

**Failure scenario:** Not wrong output -- wall clock. `cheap_conditional_gate_scan._flush` calls it once per spec
clearing the operand margin (up to 800 specs at the shipped `k_gate=8`/`k_operand=10`), and
`cheap_row_argmax_scan` once per surviving triple (up to `max_triples=40`). Each call fires 12 independent
`_mi()` dispatches on the same fixed feature, each paying its own njit/GPU launch overhead.
`_integer_lattice_fe.py` :198-220 documents the identical fix at 1.95x on n=200k, bit-identical.

**Suggested fix:** Port `_integer_lattice_fe.py` :210-219 verbatim: keep `rng.permutation(n)` called `n_perm`
times in the same order so the RNG draw sequence is preserved, build the `(n, n_perm)` matrix of
`feat[argsort(perm_i)]` via the joint-reindex invariance `MI(feat; y[perm]) == MI(feat[inv_perm]; y)`, and score
it in one `_mi_classif_batch`. One extra complication versus the lattice copy: `feat` here may already be a
resident cupy handle (:333-336) -- keep the per-perm path for that branch, or materialise the reindex on device,
but do not lose the host-path win.

**Evidence:** :337-342 is a plain `for i in range(n_perm): vals[i] = _mi(...)`. `_integer_lattice_fe.py` :202-209
states it "Mirrors `_pairwise_modular_fe._perm_null_hi`" and describes the invariance.
`_lattice_gate_proto_shared.py` :16 has the same loop but is prototype-only, so this is the last production site.

**Disposition:** RESOLVED, with the resident-handle complication handled as the finding suggests. The host path builds the `(n, n_perm)` matrix of `feat[argsort(perm_i)]` and scores it in one `_mi_classif_batch`, with `rng.permutation(n)` still called `n_perm` times in the same order, so the result is bit-identical to the loop. The resident-cupy branch keeps the per-perm path: there the fixed candidate is already uploaded once and reused across all shuffles, so it never paid the re-upload cost the batching removes. `tests/feature_selection/test_fs_filters_target_cast_fingerprints_and_pads.py`.

### FS_FILTERS_MRMR-4 [P2] cache-identity-divergence

**File:** `src/mlframe/feature_selection/filters/_mrmr_fingerprints.py` :192 (versus :293-298)

**Summary:** The identity-cache X fingerprint samples 10 evenly-spaced cells per column, while
`_content_array_signature` in the same file was raised to 1024 strided samples for the explicitly documented
reason that 10 collided. One rule, two copies, fixed on one side.

**Failure scenario:** :293's own comment names it: "The prior 10-cell sample collided on any two frames whose ten
boundary cells happened to agree (e.g. column-wise outlier clip that preserves min/median/max rows)."
`_mrmr_compute_x_fingerprint` still takes exactly that sample (:192-195). Two frames differing only outside those
10 positions -- an outlier-clipped, winsorised or NaN-imputed variant of the same frame, the canonical case for a
suite sweeping preprocessing variants -- produce identical `cols`, `n_rows`, `dtypes_repr` and `cell_sample`.
With `mrmr_skip_when_prior_was_identity=True` (default) and a matching y, `_mrmr_class.py` :3605 then calls
`_fit_identity_shortcut(X)`, which sets `support_ = arange(n_cols)` and `mrmr_gains_ = []` -- the selector
silently returns "select everything" for a frame it never scored. `mrmr_identity_cache_include_y=True` does not
close this: a preprocessing sweep holds y fixed by construction.

**Suggested fix:** Raise `n_sample` at :192 from 10 to the same 1024, and factor the sampling rule into one
shared helper both functions call, so the next fix cannot land on one copy. Cost stays O(1) in `n_rows`.

**Evidence:** :140-142 docstring claims the cell sample "mirrors `_content_array_signature`" -- that mirror claim
is now false; the sibling moved to `_n_samples = 1024` at :298 and :293 documents why.

**Disposition:** RESOLVED, and the rule is now shared rather than duplicated: both fingerprints read one `_CELL_SAMPLE_POSITIONS = 1024` constant, so they cannot diverge again. Verified with the finding's own canonical case -- a frame and its column-wise-clipped variant now fingerprint differently, while a copy of the same frame still fingerprints identically. `tests/feature_selection/test_fs_filters_target_cast_fingerprints_and_pads.py`.

### FS_FILTERS_MRMR-5 [P2] contract-drift

**File:** `src/mlframe/feature_selection/filters/_conditional_gate_fe.py` :338, :451, :637

**Summary:** Three sites cast the target with `np.asarray(y).astype(np.int64)` -- precisely the continuous-target
truncation trap `_y_encoding.py` exists to prevent and which the sibling FE families all route through
`encode_y_for_classif_mi`.

**Failure scenario:** `hybrid_conditional_gate_fe_with_recipes`, `hybrid_row_argmax_fe_with_recipes`,
`cheap_conditional_gate_scan`, `cheap_row_argmax_scan` and `detect_*` are all in `__all__` and all take a raw y.
Called with a continuous regression target in [0, 1) -- log-returns, a normalised label, a probability -- every
`astype(np.int64)` collapses the target to class 0, so every MI in the module reads exactly 0.0, the gate never
fires, and the family silently emits zero features with no error and no warning. A target in [0, 10) collapses to
10 classes, which is worse: MI is non-zero but measures a truncated target, so tau and the accept decision are
made against a signal that is not the user's. The MRMR default path is safe by luck only -- the cascade pre-bins
before calling in, and its comment records that the "prior int64 cast turned continuous y into ~n distinct
classes" for exactly this module. The cast at the module boundary was never removed.

**Suggested fix:** Replace all three with `encode_y_for_classif_mi(y)`, as `_integer_lattice_fe.py` :213 does. It
is idempotent on already-dense integer codes, so the MRMR path stays bit-identical while the public API stops
silently destroying a continuous target.

**Evidence:** `_y_encoding.py` :3-11 names this exact defect. `_conditional_gate_fe.py` never imports it.

**Disposition:** RESOLVED. All three module-boundary casts route through `encode_y_for_classif_mi`, as the sibling families already do. Confirmed the trap reproduces exactly as described: on a target drawn from U(0, 1), `np.asarray(y).astype(np.int64)` leaves exactly ONE distinct class, so every MI in the module reads 0.0 and the family emits nothing; the encoder leaves it multi-class. The encoder is idempotent on dense integer codes, so the classification path is unchanged. `tests/feature_selection/test_fs_filters_target_cast_fingerprints_and_pads.py`.

### FS_FILTERS_MRMR-6 [P2] dead-guard

**File:** `src/mlframe/feature_selection/filters/polynom_pair_fe.py` :387 (guard at :447, pool at :496)

**Summary:** The `fe_deadline_passed()` check inside the multi-seed re-optimisation loop is unreachable in the
parallel configuration it was added for, because the work goes to a loky PROCESS pool and `_fe_deadline`'s state
is a `threading.local`.

**Failure scenario:** `_fe_deadline._state` is `threading.local()`, which does not cross a thread boundary and
certainly not a process boundary. :447 routes to the parallel path whenever `_polynom_n_jobs > 1` and
`_n_pairs_to_eval >= 16` -- trivially reached -- and :496 dispatches with `backend=_loky_cpu_backend`. Inside a
loky worker the deadline is always None, so `fe_deadline_passed()` returns False unconditionally and the loop
runs to completion regardless of `max_runtime_mins`.

**Suggested fix:** Pass the absolute deadline as an explicit `delayed()` kwarg and re-publish it in the worker
with `set_fe_deadline(...)` -- the pattern `_fe_deadline.py` :20-23 already prescribes. Alternatively check the
deadline in the DISPATCHING loop between `Parallel` batches, which does run on the main thread.

**Evidence:** `_fe_deadline.py` :17-20 flags this call site as the known exception but describes it as
`n_jobs=1`-only, i.e. live only in the serial configuration nobody runs by default. Mitigating:
`fe_smart_polynom_iters` defaults to 0, so the family is off unless opted into -- hence P2.

**Disposition:** RESOLVED, and the finding UNDERSTATES the scope. It reports the check as live in the serial configuration and dead only under `n_jobs > 1`; in fact `_eval_one_pair` runs the impl on a big-stack sub-thread (the Windows loky 1MB-stack numba workaround) in EVERY configuration, and a `threading.local` does not cross that boundary either -- so `fe_deadline_passed()` returned False unconditionally whether the work went to a process pool or ran serially. The deadline is now read once on the main thread, passed as an explicit argument through both wrappers, and re-published with `set_fe_deadline` inside the execution context, which is the pattern `_fe_deadline.py` prescribes. `tests/feature_selection/test_fs_filters_target_cast_fingerprints_and_pads.py`.

### FS_FILTERS_MRMR-7 [P3] wasted-per-call-work

**File:** `src/mlframe/feature_selection/filters/_binned_numeric_agg_fe.py` :36-59, consumed at :89

**Summary:** `_per_cell_raw_moments_njit` accumulates `s2`, `s3` and `s4` on every row, but its only caller
discards all three -- pass 1 of the two-pass stable scheme does 2.5x the arithmetic and 1.67x the allocation it
needs.

**Failure scenario:** Not wrong output -- overhead on a hot kernel. `_per_cell_moments_stable` (:89) destructures
`cnt, s1, _, _, _`, so `x2 = x*x`, `s2[c] += x2`, `s3[c] += x2*x`, `s4[c] += x2*x2` are computed and thrown away
for every row, plus three unused `np.zeros(n_cells)`. `fit_binned_numeric_agg` calls it once for the full pass
and once per fold for every `(group_col, agg_col)` pair -- at the shipped `n_folds=5` and 16 group columns that
is 6 x 16 x |agg_cols| full O(n) passes carrying the dead arithmetic.

**Suggested fix:** Add a `_per_cell_count_sum_njit(codes, v, n_cells) -> (cnt, s1)` pruned variant and call it
from `_per_cell_moments_stable`. Bit-identical by construction -- `cnt`/`s1` accumulate in the same row order.
The GPU twin already does this correctly (`_binned_numeric_agg_resident.py` :86-87 bincounts only `cnt` and
`s1`), so this is a host-only gap.

**Evidence:** :89 `cnt, s1, _, _, _ = _per_cell_raw_moments_njit(...)`; cluster-wide grep returns exactly two
hits -- the definition and that call.

**Disposition:** RESOLVED. `_per_cell_count_sum_njit` is a pruned twin of the full kernel called from `_per_cell_moments_stable`, mirroring it statement for statement so `cnt` and `s1` are bit-identical (asserted, not merely argued). One correction to the suggested fix: the full kernel has NO negative-code guard, so a `c < 0` skip in the pruned variant would have changed behaviour -- `codes` comes from `np.searchsorted`, which never returns a negative, so the case is unreachable either way and mirroring exactly is the right call for a bit-identity contract. `tests/feature_selection/test_fs_filters_target_cast_fingerprints_and_pads.py`.

### FS_FILTERS_MRMR-8 [P3] additive-epsilon-denominator

**File:** `src/mlframe/feature_selection/filters/hermite_fe/__init__.py` :858

**Summary:** `std = (s2 / n) ** 0.5 + 1e-12` pads a denominator additively, then that padded `std` is inverted
and cubed / fourth-powered to form the skew and excess kurtosis used as basis-routing decision variables.

**Failure scenario:** The moment kernel itself is correct (two-pass, centred, :852-857). The defect is
downstream: :859-862 compute `inv = 1.0/std`, `skew = (s3/n)*inv**3`, `kurt = (s4/n)*inv**4 - 3`. For a column
whose TRUE std lands in roughly [1e-12, 1e-10] -- a pre-scaled feature, or a tick-level price delta in units
where the spread is ~1e-11 -- the additive pad is the same order as the real denominator, so `skew` is scaled by
up to 0.125 and `kurt_excess` by up to 1/16. `basis_route_by_moments` then branches on `abs(skew) > 1.5` (:888)
and `abs(skew) < 0.5 and abs(kurt_excess) < 1.0` (:894), so an unpadded skew of 2.0 reads as 0.25 and a genuinely
heavy-tailed one-sided column is routed to Hermite instead of Laguerre. `spread_ratio = rng/std` (:885) is
deflated by the same factor, biasing toward Chebyshev.

**Suggested fix:** Drop the pad and guard multiplicatively as the two already-fixed siblings do: compute
`std = (s2/n) ** 0.5` and return `skew = 0.0, kurt_excess = 0.0` when `std <= 1e-12`, matching
`_global_stats_all`'s short-circuit. Guard `spread_ratio` on `std > 0` explicitly rather than relying on the pad.

**Evidence:** :831 docstring states the intent is to match `np.std(x) + 1e-12`, i.e. the pad was inherited from
the numpy body, not chosen. `_binned_numeric_agg_fe.py` :144-147 and `_target_encoding_fe.py` :137-141 both
document removing exactly this pad as a SECOND bug distinct from the cancellation one.

**Disposition:** RESOLVED as suggested -- the pad is gone and a `std <= 1e-12` column returns `skew = kurt_excess = 0.0`, matching `_global_stats_all`'s short-circuit. Verified on a gamma-shaped column rescaled by 1e-11: skew is now scale-invariant to 1e-6 relative, a genuinely heavy skew still reads above the 1.5 routing threshold at that scale, and `basis_route_by_moments` returns the same basis for the column and its rescaled copy. `tests/feature_selection/test_fs_filters_target_cast_fingerprints_and_pads.py`.

### FS_FILTERS_MRMR-9 [P3] permutation-null-uniformity

**File:** `src/mlframe/feature_selection/filters/_permutation_null_resident.py` :171-174

**Summary:** The device-born shuffle generator sorts FLOAT32 random keys, which have only ~2^24 distinct values,
so at the large n this path is gated to fire at, each row carries thousands of tied keys whose relative order is
decided by `cp.argsort`'s tie-breaking rather than by the RNG.

**Failure scenario:** `keys = _rng.random((nperm, n), dtype=cp.float32)` then `order = cp.argsort(keys, axis=1)`.
float32 uniforms occupy ~1.68e7 grid points, so expected tied pairs per row is n^2 / 2^25: at n=600k that is
~10,700, at n=2M ~1.2e5. `argsort` resolves ties by index, so tied positions keep their original relative order,
giving each row a small positive correlation with the identity permutation instead of a uniform draw. The error
direction is conservative -- the null MI band is slightly inflated, so the maxT floor is slightly too strict and
rejects marginal true candidates -- hence P3. But the GPU floor is then not the same estimator as the CPU
Fisher-Yates floor, in a way the docstring's "statistically equivalent, not byte-identical" claim does not cover.

**Suggested fix:** Generate the keys as `cp.float64` (keep it under the existing KTC/VRAM gate since the buffer
doubles), or use `cp.random.Generator.permuted` / a per-row device Fisher-Yates, which is tie-free by
construction and avoids materialising both `keys` and `order`.

**Evidence:** :160-162 assert "Each row is a uniform permutation of `y_codes` (argsort of i.i.d. keys)", which
holds only for continuous keys. The CPU counterparts are exact permutations (`_permutation_null.py` :148-152,
:182).

**Disposition:** RESOLVED with the float64 key option, the cheaper of the two suggested. The buffer doubles, but it stays under the existing KTC/VRAM gate that already decides whether this path runs at all, and float64 uniforms make the tie count per row negligible rather than the ~10,700 at n=600k that `argsort`'s index tie-break was resolving toward the identity permutation. `tests/feature_selection/test_fs_filters_target_cast_fingerprints_and_pads.py`.

### FS_FILTERS_MRMR-10 [P3] cache-identity-collision

**File:** `src/mlframe/feature_selection/filters/_mrmr_fingerprints.py` :219, :130, :284, :290

**Summary:** Four fingerprint failure paths fall back to an `id()`-derived key; CPython reuses object addresses
after garbage collection, so two genuinely different frames can receive the same identity-cache key.

**Failure scenario:** In a suite that builds a frame, fits, drops the reference, then builds a different frame,
CPython's allocator very commonly hands the second frame the freed address, so `id(X2) == id(X1)`. If the first
fit stored True under `fp_id<addr>`, the second hits it and takes `_fit_identity_shortcut`. Reaching :219
requires the whole content-fingerprint `try` to have raised, which is uncommon -- hence P3 -- but the fallback is
chosen precisely when the code has LEAST information about X, which is the worst moment to key a cache on an
address.

**Suggested fix:** Return a per-call unique, never-matching token (`f"fp_uncacheable_{uuid4().hex}"`) so a
fingerprint failure disables the cache for that call rather than risking a false hit. Same for the y side and
both `_content_array_signature` returns.

**Evidence:** :217-219 is `except Exception` -> `logger.debug` -> the `id()` key; `_mrmr_class.py` :3571 treats
any stored True as licence to short-circuit the entire fit.

**Disposition:** RESOLVED as suggested, on all four paths: a fingerprint failure now returns a `uuid4`-based token that cannot match anything, so the cache is disabled for that call instead of risking a false hit on a reused address. The failure is also warned rather than logged at debug, since it means the identity cache is off. `tests/feature_selection/test_fs_filters_target_cast_fingerprints_and_pads.py` asserts the property directly: two failed fingerprints never match.

### FS_FILTERS_MRMR-11 [P3] diagnosability

**File:** `src/mlframe/feature_selection/filters/_target_encoding_fe.py` :157, :160

**Summary:** These two `np.where` expressions evaluate both branches, so a zero-variance category produces
`0.0/0.0` and emits a `RuntimeWarning` on every call -- the sibling fixed at the same time wraps the identical
expression in `np.errstate`.

**Failure scenario:** `np.where(std > 1e-9, m3n / std**3, 0.0)` computes the division for EVERY category
including those where `std == 0` (any single-row or all-identical-y category, routine with high-cardinality
categoricals). The discarded values are correct-by-`where`, so this is noise rather than wrongness -- but it
fires per fold per column, floods logs, and would hard-fail any suite under `-W error` or
`np.seterr(all="raise")`.

**Suggested fix:** Wrap both in `with np.errstate(divide="ignore", invalid="ignore"):`, as
`_binned_numeric_agg_fe.py` :167 and :171 do -- those carry a comment explaining the suppression belongs in the
function rather than at every caller.

**Evidence:** `_binned_numeric_agg_fe.py` :163-172 is the same arithmetic with the guard;
`_target_encoding_fe.py` :155-160 is the same arithmetic without it.

**Disposition:** RESOLVED as suggested -- both expressions are wrapped in `np.errstate(divide="ignore", invalid="ignore")`, matching the guard `_binned_numeric_agg_fe` already carries on the same arithmetic. Values are unchanged; the `np.where` already selected correctly. `tests/feature_selection/test_fs_filters_target_cast_fingerprints_and_pads.py`.

## Verified, not a finding

- `_derive_cell_stats` per-cell skew/kurt (the documented OPEN item) is FIXED here: :137-176 takes
  `(cnt, mean, cm2, cm3, cm4)` from a genuine two-pass kernel, no binomial expansion, no `+1e-12` pad. The GPU
  twin matches, and the masked TRAIN variant correctly refuses the `full - test` additivity shortcut.
- Raw-power skew/kurt sweep: no other occurrence cluster-wide.
- JSON cache keys: every `json.dumps`/`orjson.dumps` feeding a hash uses sorted keys.
- Silent broad excepts: 12 hits cluster-wide, all 12 carrying a comment documenting a prior triage.
- `pl.Categorical` where `pl.Enum` is correct: no live instance -- the one cast returns codes together with the
  matching categories from the same Series, and its module is gated off and unwired.
- `min() == max()` on an all-null polars column: no instance.
- Gate-selection optimism is already documented with a FUTURE disposition; not re-reported.

## Coverage

Read in full or substantial part (22 files, 14,621 LOC): `_binned_numeric_agg_fe.py`,
`_binned_numeric_agg_resident.py`, `_conditional_gate_fe.py`, `_integer_lattice_fe.py`, `_mrmr_fingerprints.py`,
`_kernel_tuning.py`, `_fe_deadline.py`, `_y_encoding.py`, `_target_encoding_fe.py`, `_permutation_null.py`,
`_permutation_null_resident.py`, `_fe_matrix_io.py`, `_fe_subsample.py`, `polynom_pair_fe.py`,
`hermite_fe/__init__.py`, `mrmr/_mrmr_class.py` (identity-cache region), `mrmr/_mrmr_class_fit_helpers.py`,
`_mrmr_fit_impl/_fe_stage_cascade_mid_a.py`, `discretization/_discretization_dataset.py`,
`_lattice_gate_proto_shared.py`, `_wavelet_basis_fe.py` (null region), `_orthogonal_tail_dependence_fe.py`.

Swept by targeted grep across all ~250 cluster files: raw-power moment expansions; additive-epsilon denominators;
broad `except Exception` with no logging; JSON cache-key serialisation; `pl.Categorical`; `min()==max()`
constant checks; per-permutation Python loops; hardcoded parallelism thresholds; and
`@njit`/`prange`/`cuda.jit`/`cupy`/`kernel_tuning_cache` presence (268 KTC sites, 176 CUDA-kernel sites) as the
mandatory pre-check before any perf claim.

**Not reached -- recommend a follow-up pass.** The MRMR greedy core itself (`_mi_greedy_cmi_fe.py` 106KB,
`_mi_greedy_fe.py`, `evaluation.py` 66KB, `_usability_aware_selection.py` 58KB, `_screen_predictors.py` 60KB) --
`evaluation.py` was entered only via grep, its selection loop never read. The GPU-resident
materialisation/select subsystem (`_gpu_resident_materialise.py` 61KB, `_gpu_resident_select_kernels.py` 55KB,
`_gpu_resident_fe.py` 76KB, `_gpu_resident_pair_mi.py` 48KB) beyond the greps. `_fe_batched_mi.py` /
`_fe_batched_mi_cmi.py` (60KB + 52KB), `_mdlp_validated_split.py` (54KB), `_cat_confirm_permutation.py` (49KB)
and the cat-interaction families. The `_dynamic_cluster_discovery/`, `_feature_engineering_pairs/`,
`_gpu_strict_fe/`, `_orthogonal_univariate_fe/`, `info_theory/`, `mrmr/` and `_mrmr_fit_impl/` subpackages beyond
the files named above. The ~40 `_orthogonal_*_fe.py` FE-generation families were swept only for the four named
bug classes, not read for selection integrity or contract drift -- the single largest unexamined surface here.
