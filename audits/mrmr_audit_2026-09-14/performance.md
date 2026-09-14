# performance (cross-cutting) — mrmr_audit_2026-09-14

## Scope

Cross-cutting performance sweep of the whole MRMR surface (`find src/mlframe -ipath "*mrmr*" -name "*.py"`,
77 files / ~31 440 LOC) **plus the first-party primitives MRMR calls into that do not live on an `*mrmr*`
path** — `evaluation.py`, `_evaluation_driver.py`, `_relaxmrmr_3d.py`, `_pairwise_modular_fe.py`,
`_pairwise_modular_resident.py`, `_integer_lattice_fe.py`, `_conditional_gate_fe.py`,
`_orthogonal_univariate_fe/_orth_mi_backends.py`, `_eng_dedup_scan.py`, `_eng_dedup_batch_corr.py`.
`_benchmarks/` files were read as evidence of what has already been measured, not audited as production code.

Read-only. No benchmark was run; **every number below that is not a line count or an array-size arithmetic
is labelled "expected win, unmeasured"**.

Severity: P0 wrong results / P1 real live perf failure / P2 meaningful inefficiency / P3 quality-nit.

---

## Findings

### PERF-1 — `_pairwise_modular_fe._perm_null_hi`'s HOST path never got the batching its two siblings did; the batched path is gated behind cupy+STRICT-resident, so a CPU-only install always runs the 12-dispatch loop  [P1]

**Where:** `src/mlframe/feature_selection/filters/_pairwise_modular_fe.py:282-296`
(gate: `_pairwise_modular_resident.py:180-224`, which returns `None` on any non-cupy / non-STRICT host)

**What:**
```python
perms = [rng.permutation(yi.size) for _ in range(n_perm)]
_rb = fe_gpu_strict_resident_enabled()
vals = perm_null_residue_mis_resident(r, yi, perms, eff_nbins=eff_nbins, rank_binning=_rb)
if vals is None:
    vals = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        vals[i] = _mi(r, yi[perms[i]], nbins=eff_nbins)    # 12 separate dispatches
```
`perm_null_residue_mis_resident` is wrapped in `try: import cupy ... except -> return None`, so on **any**
host without cupy, or with `MLFRAME_FE_GPU_STRICT_RESIDENT` off (the default), `vals is None` and the
unbatched per-perm loop runs.

**Why it is wrong / costly:** CLAUDE.md's 2026-08-05 entry credits `_pairwise_modular_fe._perm_null_hi` as
"the sibling that already had this fix" and ports it *from* here *to* `_integer_lattice_fe`. The credit is
only half-true: the batching landed on the **GPU-resident** path. The two siblings
(`_integer_lattice_fe.py:212-220`, `_conditional_gate_fe.py:326-338`) batch on the **HOST**, via
`_mi_classif_batch` → `plugin_mi_classif_batch_dispatch` (njit `prange`, ~53× over the per-column loop per
`_orth_mi_backends.py:83`). `_pairwise_modular_fe` has no host-batched branch at all. So the module that was
the *source* of the fix is now the only one of the three still paying 12 Python→njit round-trips on the
default CPU configuration — the exact inversion of the sibling-divergence shape the repo has hit before.
This is also gate-shape #4: `fe_gpu_strict_resident_enabled()` is default-off, so the optimisation the
docstring advertises ("SF2 :311 collapse") is never taken in a default production fit.

**Fix:** add a host-batched `else` branch mirroring `_integer_lattice_fe._perm_null_hi` exactly: build the
`(n, n_perm)` matrix of `r[inv_perm_i]` from the **already-drawn** `perms` list (no new RNG draws — draw
order is unchanged, so bit-identity is by construction) and score all 12 columns against the unpermuted `yi`
in one `_mi_classif_batch(mat, yi, nbins=eff_nbins)` call. Keep the resident path as the first choice.

**Bench plan:** `bench_pairwise_modular_perm_null.py`. Compare the current host fallback (force
`perm_null_residue_mis_resident` → `None`) against the batched host branch on the same `r`/`yi`/`seed`, at
n ∈ {50k, 200k, 1M, 2M}, k (modulus) ∈ {3, 7, 12}, warm (one throwaway call to JIT), best-of-10, `n_perm=12`.
Report both the isolated kernel and a full `cheap_pairwise_modular_scan` wall. Expect the same order as the
measured 1.95× the `_integer_lattice_fe` port got at n=200k — **expected win, unmeasured**.

**Bit-identity risk:** LOW. The joint-reindex invariance `MI(feat; y[perm]) == MI(feat[inv_perm]; y)` is
already proven and pinned by `test_perm_null_hi_batched_matches_per_perm_loop`. The one real risk is the
estimator: `_mi()` routes through `_mi_classif_batch` with `rank_binning=_rb`, and a batched call must pass
the SAME `rank_binning` flag or edge-vs-rank binning will diverge on tied residues (`_mod` output is
heavily tied by construction — this is the tied-data case CLAUDE.md's "verify bit-identity on the UNSAFE
case explicitly" rule names). Gate the assertion on `==`, not a tolerance, on a tied-residue fixture.

**Test:** `test_pairwise_modular_perm_null_host_batched_matches_per_perm_loop` — asserts exact equality
(`==`, not `allclose`) between the batched host branch and the per-perm loop on a heavily-tied residue
column with cupy disabled, at two seeds.

---

### PERF-2 — the same `np.unique(y).size` + `pd.qcut(q=10)` y-discretisation block is copy-pasted ~16× across the FE cascade and re-sorts the identical full-length `_y_np` once per FE family  [P1]

**Where (all on the same `_y_np`):**
`_fe_stage_cascade_early_a.py:145, 266, 547, 621`;
`_fe_stage_cascade_mid_a.py:76, 143`;
`_fe_stage_temporal_agg.py:69`;
`_hybrid_orth_family_variants/_group1.py:37, 156, 241, 322, 402`;
`_hybrid_orth_family_variants/_group2.py:35, 134, 238, 326, 421`;
plus `_fit_impl_core.py:162, 691, 2406`.

**What:** each site is the identical block, e.g. `_fe_stage_cascade_early_a.py:137-158`:
```python
_y_for_hybrid = _y_np
if _y_for_hybrid.dtype.kind in "fc":
    _n_unique = int(np.unique(_y_for_hybrid).size)
    if _n_unique <= 32:
        _y_for_hybrid = _y_for_hybrid.astype(np.int64)
    else:
        _y_for_hybrid = pd.qcut(_y_for_hybrid, q=10, labels=False, duplicates="drop").astype(np.int64)
```
`_y_for_extra` (`:262`) even starts as `_y_for_hybrid` and then, on the `else` at `:264`, re-derives the whole
thing from `_y_np` again. Every `_y_for_*` in this family is derived from the SAME `_y_np` with the SAME
parameters, so every one of these ~17 sites computes a bit-identical result.

**Why it is wrong / costly:** `np.unique` on a float column is a full O(n log n) sort **plus** an n-element
allocation; `pd.qcut(q=10)` is another full sort + quantile pass + an n-element int64 output. On a
continuous regression target these run ~17 times per fit over the identical array. At n = 2 M that is ~17
sorts of a 2 M float64 array plus ~17 qcuts — pure, provably-duplicated work in the FE cascade's serial
orchestration frame, where it cannot be hidden by any kernel parallelism. At n = 1e7 it is ~17 × (80 MB sort
+ 80 MB qcut allocation). It scales with the NUMBER OF ENABLED FE FAMILIES, i.e. it gets worse exactly on
the wide, fully-enabled fits this module exists for.

**Fix:** one fit-scoped memo, e.g. `_discrete_y_for_fe(self, y_np)` in `_mrmr_fit_impl/_helpers.py`, caching
on `self` keyed by `(id/content-fingerprint of y_np, dtype)`. Every cascade site calls it. Additionally, the
`<= 32` test does not need `np.unique` at all — a capped-cardinality helper (`np.bincount` on a cast copy,
or a `set` built with an early break at 33 distinct values) is O(n) with no sort and no n-element
allocation, and returns the same boolean. Do both: the memo removes the repetition, the capped helper
removes the sort even on the first call.

**Bench plan:** `bench_fe_cascade_y_discretisation.py`. Time the 17-site sequence as written vs. the memoised
version on a continuous y at n ∈ {200k, 2M, 10M}, warm, best-of-5, with `line_profiler` on
`_fe_stage_cascade_early_a` to attribute the saving to the `np.unique`/`qcut` lines specifically (cProfile
will attribute qcut's cost to pandas). Also report the **full-fit wall** with all FE families enabled —
an isolated win that is flat end-to-end is a REJECT per CLAUDE.md. **Expected win, unmeasured.**

**Bit-identity risk:** NONE for the memo (same inputs → same output, returned object is only read). The
capped-cardinality helper needs care: it must count distinct values including NaN exactly as `np.unique`
does (`np.unique` treats each NaN bit-pattern as distinct in older numpy, collapses them in ≥1.21) — pin
the NaN case in the test rather than assuming.

**Test:** `test_fe_cascade_y_discretisation_memo_matches_per_site_recompute` — asserts the memoised
`_y_for_*` is `array_equal` to the per-site recomputation for float/int/NaN-bearing/all-tied targets, and
`test_y_cardinality_capped_matches_np_unique_size` for the capped counter incl. the NaN case.

---

### PERF-3 — `np.argsort(perm)` used to invert a permutation at 4 hot sites: O(n log n) where an O(n) scatter is exactly equal  [P1]

**Where:**
`_integer_lattice_fe.py:218` — `mat[:, i] = feat[np.argsort(perm)]`
`_conditional_gate_fe.py:335` — `mat[:, i] = feat_host[np.argsort(perm)]`
`_pairwise_modular_resident.py:208` — `inv_idx[:, i] = np.argsort(np.asarray(perm))`
(and `_split_helpers.py:186` outside MRMR, same shape — noted, not in scope)

**What:** `perm` is the output of `rng.permutation(n)` — a permutation of `0..n-1` with no ties. For such an
array `np.argsort(perm)` is **by definition** the inverse permutation, and the inverse can be built by a
single scatter:
```python
inv = np.empty(n, dtype=np.int64); inv[perm] = np.arange(n)
```
Better still, the gather-then-store can be collapsed into one scatter with no index array at all:
`mat[perm, i] = feat` is identically `mat[:, i] = feat[argsort(perm)]`.

**Why it is wrong / costly:** each of these is inside a `for i in range(n_perm)` loop with `n_perm=12`, i.e.
12 full O(n log n) sorts per `_perm_null_hi` call, where 12 O(n) scatters suffice. `np.argsort` on int64 at
n = 1e6 is roughly an order of magnitude more expensive per element than a linear scatter, and these calls
sit directly in the FE candidate scan (one `_perm_null_hi` per surviving candidate). This is un-taken
because the code *reads* minimal — the exact class CLAUDE.md's 2026-08-04 PROCESS FIX warns about: I greped
all three sites for `@njit`/`prange`/`cuda.jit`/`cupy`/`get_or_tune` and **none** covers these lines (the
njit kernel starts only at `_mi_classif_batch`, downstream of the matrix build).

**Fix:** replace all three with the scatter form. Prefer `mat[perm, i] = feat` (one pass, no index temp).
Keep `rng.permutation(n)` untouched so the draw order and therefore bit-identity are preserved.

**Bench plan:** `bench_perm_inverse_scatter.py` — microbench `feat[np.argsort(perm)]` vs `out[perm] = feat`
at n ∈ {10k, 200k, 1M, 10M}, warm, best-of-20, plus the enclosing `_perm_null_hi` wall at n ∈ {200k, 2M}
with `n_perm=12`, and a full `cheap_integer_lattice_scan` wall so the e2e effect is on record.
**Expected win, unmeasured.**

**Bit-identity risk:** NONE, and provably so: for a duplicate-free integer array, `argsort` (any kind) and
the scatter-inverse produce the identical index array; the resulting `mat` is bit-identical, not
tolerance-close. Assert with `==`.

**Test:** `test_perm_inverse_scatter_bit_identical_to_argsort` — for 200 random seeds and n ∈ {7, 1000},
asserts `np.array_equal(scatter_inverse(perm), np.argsort(perm))`.

---

### PERF-4 — `_su_normalize_relevance` re-factorises the whole target per candidate although `freqs_y` is already in the caller's scope  [P2]

**Where:** `src/mlframe/feature_selection/filters/evaluation.py:87-110`, called at `:447`, `:577`

**What:**
```python
_, _freqs_x_su, _ = merge_vars(factors_data=factors_data, vars_indices=_x_idx, ...)
_, _freqs_y_su, _ = merge_vars(factors_data=factors_data, vars_indices=_y_idx, ...)
_denom_su = entropy(freqs=_freqs_x_su) + entropy(freqs=_freqs_y_su)
```
The `y` half depends only on `(factors_data, y, factors_nbins, dtype)` — all fit-constant. `evaluate_candidate`
already receives `freqs_y` as a parameter (it forwards it to `mi_direct` at `:530`, `:565`, `:588`).

**Why it is wrong / costly:** `merge_vars` is a full O(n) factorisation + frequency count with an n-element
allocation. It runs **once per candidate** — on a wide fit that is p (up to ~1e5 after the SIS screen, ~2k
after it) × rounds times, for a value that never changes across the whole fit. Only active under
`mi_normalization='su'`, which is why it has survived: it is invisible on the default-config profile.

**Fix:** pass `freqs_y` into `_su_normalize_relevance` and use `entropy(freqs=freqs_y)` for the y term,
falling back to the current `merge_vars` call only when `freqs_y is None` (standalone/unit-test callers).
Confirm first that `freqs_y` is built by the same `merge_vars` call with the same `dtype` — if the
`var_is_nominal`/`dtype` arguments differ the frequencies could differ and this becomes a correctness change,
not a perf one. **Unverified**: settle it by asserting `np.array_equal` of the two frequency vectors on a
real fit before wiring.

**Bench plan:** `bench_su_normalize_y_hoist.py`. Full `MRMR.fit(mi_normalization="su")` wall at
n ∈ {100k, 2M}, p ∈ {50, 500}, warm, 3-rep median, with cProfile `cumtime` on `merge_vars` before/after —
the call-count drop is contention-immune evidence even if the wall is noisy. **Expected win, unmeasured.**

**Bit-identity risk:** LOW if the `array_equal` precondition above holds; otherwise the SU denominator moves
and selection can change. Gate the change on that assertion.

**Test:** `test_su_normalize_uses_precomputed_freqs_y_identically` — asserts the returned scaled gain is
bit-identical (`==`) with and without the hoisted `freqs_y`, and that `merge_vars` is called once (not twice)
per candidate via a spy.

---

### PERF-5 — RelaxMRMR got a per-round hoist of `y_col` / every selected column; BUR, PID, CMI-perm-stop and CPT did not, and each re-materialises them per candidate  [P2]

**Where:** the hoist that exists: `_evaluation_driver.py:370-385` + `evaluation.py:769-778`.
The four that did not get it:
- BUR — `evaluation.py:708-719` (`mi(...)` once per selected var per candidate)
- PID — `evaluation.py:794-801` (`_materialize_var(X)`, `_materialize_var(y)`, then `_materialize_var(_z)` per selected var)
- CMI-perm-stop — `evaluation.py:828-834` (same shape)
- CPT — `evaluation.py:857-865` (same shape, plus the `z_comp` composite rebuild)

**What:** `_evaluation_driver.py` computes `_relax_y_col` / `_relax_sel_cols` / `_relax_sel_nbins` once per
greedy iteration and threads them in; `evaluation.py:769` uses them when present. The four sibling
research-knob blocks immediately below it each open with their own
`x_col, k_x = _materialize_var(...)` / `y_col, k_y = _materialize_var(...)` and their own
`for _z in selected_vars: _materialize_var(_z)` loop — with no hoist parameter at all.

**Why it is wrong / costly:** this is CLAUDE.md's "already-optimised primitive, just not wired into every
call site that needs it" shape, verbatim — the same shape as the `_jackknife_ece` gap. `_materialize_var`
calls `merge_vars` (full O(n) factorisation + allocation). Per candidate, with |S| = k selected, each active
knob pays k+2 of them; over one greedy round of C candidates that is C·(k+2), and over a fit
O(k²·C) full-length factorisations of arrays that are constant within the round. Additionally, when two or
more knobs are active **in the same call**, `x_col` and `y_col` are each materialised 2–4 separate times for
the same candidate inside the same function body (lines 761, 794-795, 828-829, 857-858) — pure intra-call
duplication independent of the hoist question.

Secondary: these are all default-off knobs (`bur_lambda=0.0`, `pid_synergy_bonus=0.0`, `cmi_perm_stop=False`,
`cpt_test=False` — `_mrmr_class.py:403-432`), so this never shows on a default profile. That is the reason
it is still here, not a reason to leave it: any user who turns a knob on gets the O(k²·C) shape.

**Fix:** (a) materialise `x_col, k_x` ONCE at the top of the knob section and share it across all five
blocks; (b) extend the `_evaluation_driver.py:370-385` hoist to build `y_col`/`sel_cols`/`sel_nbins` whenever
ANY of the five knobs is active (not only RelaxMRMR) and thread the same `_relax_*` parameters into the four
other blocks — the parameters already exist, only the activation predicate and four call sites change.
(c) For BUR specifically, `mi(X, _z)` for a (candidate, selected) pair is recomputed every round the pair
survives — it belongs in the existing `cached_cond_MIs`-style memo keyed on the pair.

**Bench plan:** `bench_research_knob_hoist.py`. `MRMR.fit` wall with each knob individually enabled
(`bur_lambda=0.5`, `pid_synergy_bonus=0.5`, `cmi_perm_stop=True`, `cpt_test=True`) at n ∈ {100k, 1M},
p ∈ {100, 1000}, n_features ∈ {20, 200}, warm, 3-rep median; report `merge_vars` call count from cProfile
alongside the wall. **Expected win, unmeasured.**

**Bit-identity risk:** NONE — the hoisted values are byte-identical by construction (same function, same
fit-constant arguments). The only risk is a stale hoist if `selected_vars` changes mid-round; the existing
RelaxMRMR hoist's own comment establishes that it does not. Pin that with the test below.

**Test:** `test_research_knob_scores_identical_with_and_without_hoist` — for each of the four knobs, asserts
the returned `current_gain` and `expected_gains[cand_idx]` are bit-identical between the hoisted and
un-hoisted paths on a fixed fixture/seed; `test_materialize_var_called_once_per_round_not_per_candidate`
spies the call count.

---

### PERF-6 — `relax_mrmr_score` re-copies every selected column to int64 and re-runs its guards per candidate, and its O(|S|²) interaction term is a Python double loop over two njit kernels  [P2]

**Where:** `src/mlframe/feature_selection/filters/_relaxmrmr_3d.py:151-211`

**What:** three separate problems in one function, all per candidate:
1. `:153-154` — `for _j, _c in enumerate(selected_cols): _assert_codes_in_range(_c, ...)`: a full O(n) min/max
   scan of every selected column, for columns the caller already hoisted and which never change.
2. `:155-156, :179` — `x_int = x_cand.astype(np.int64)`, `y_int = y.astype(np.int64)`,
   `sel_int = [col.astype(np.int64) for col in selected_cols]`: |S|+2 full n-element **copies** per candidate.
   The caller's hoist (`_evaluation_driver.py:378-382`) already produced int64 arrays via `_materialize_var`
   (which returns `np.asarray(classes, dtype=np.int64)`), so `.astype(np.int64)` on them is a pure
   no-op-that-still-copies — `astype` copies by default.
3. `:199-209` — the interaction term:
```python
for i in range(n_S):
    for j in range(i + 1, n_S):
        cmi_ij  = _joint_cmi_xy_given_zw_njit(x_int, y_int, col_i, col_j, K_x, K_y, K_i, K_j)
        mi_x_zz = _mi_x_pair_njit(x_int, col_i, col_j, K_x, K_i, K_j)
```
   — the canonical "Python-level loop calling an already-`njit` kernel once per iteration" shape. Both
   kernels are `@njit(nogil=True, cache=True)` and **sequential**; there is no `prange` anywhere in this file
   (greped: `parallel=True`, `prange`, `cuda.jit`, `cupy`, `KernelTuningCache`, `get_or_tune` — zero hits).

**Why it is wrong / costly:** the function's own docstring says "Cost: O(|S|^2) 3-D plug-in MIs per
candidate". At |S| = 200 that is 19 900 iterations × 2 Python→njit dispatches × O(n) each, **per candidate**,
single-threaded, with the GIL held between iterations. Item 2 alone allocates (|S|+2)·n·8 bytes per
candidate: at |S| = 200, n = 1e6 that is 1.6 GB of transient copies per candidate.

**Fix:**
- Item 1/2: accept pre-validated int64 columns. Take the selected set as ONE contiguous `(n, |S|)` int64
  matrix built once per round by the caller (which already loops to build the list), and `np.asarray(...,
  dtype=np.int64)` — a view, not a copy — instead of `.astype`. Move `_assert_codes_in_range` and
  `check_joint_cardinality` for the selected set to the caller's once-per-round hoist.
- Item 3: one `@njit(parallel=True)` kernel, `prange` over the FLATTENED `i<j` pair index (not a nested
  prange — flatten to `p = 0 .. n_S*(n_S-1)/2 - 1` and decode `(i, j)`), calling the **sequential** variants
  of the two inner reductions from inside it, per CLAUDE.md's documented no-nested-`parallel=True` caveat.
  Accumulate per-pair contributions into a `(n_pairs,)` output and sum afterwards in the fixed `i<j` order
  the Python loop used, so the float accumulation order is preserved.

**Bench plan:** `bench_relax_mrmr_score.py`. Isolated `relax_mrmr_score` at n ∈ {50k, 200k, 1M},
|S| ∈ {5, 20, 50, 200}, K ∈ {10, 32}, warm, best-of-10, three variants (current / copy-elision only /
copy-elision + fused prange) so the two levers are attributed separately. Then a full
`MRMR.fit(relaxmrmr_alpha=1.0)` wall at n = 200k, p = 200, n_features = 50 — the isolated number alone is not
a verdict. Report peak RSS too (item 2 is a memory finding as much as a speed one).
**Expected win, unmeasured.**

**Bit-identity risk:** MEDIUM and it is the whole design constraint.
- Items 1/2 are bit-identical by construction (views of the same int64 data; the guards are pure predicates).
- Item 3: each `(i, j)` pair's own `cmi_ij` / `mi_x_zz` are computed by an unchanged sequential reduction, so
  per-pair values are bit-identical. The outer `inter += ...` sum MUST be done in the original `i<j` order
  after the prange (a cross-thread running sum would reorder float additions). With that, the result is
  bit-identical, not tolerance-close — assert `==`. Verify explicitly on a degenerate case (`n_S == 2`,
  all-constant selected column, `K_i == 1`) where the kernels hit their zero-guard branches.

**Test:** `test_relax_mrmr_score_fused_bit_identical_to_python_loop` (`==` across 40 synthetic scenarios
spanning |S| 2–50 and the degenerate cases) + `test_relax_mrmr_score_does_not_copy_selected_columns`
(asserts `np.shares_memory` between an input selected column and what the kernel receives).

---

### PERF-7 — `_one_vs_many_pearson_abs_masked_njit` recomputes each buffer row's mean and centred sum-of-squares on every candidate: 3 passes over `mat[j]` where 1 would do  [P2]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_eng_dedup_batch_corr.py:58-76`
(caller: `_eng_dedup_scan.py:95-110`)

**What:**
```python
for j in prange(k):
    sb = 0.0
    for i in range(n): sb += mat[j, i]          # pass 1: row mean
    mb = sb / n
    sbb = 0.0; sab = 0.0
    for i in range(n):                          # passes 2+3 (fused): row variance + cross term
        db = mat[j, i] - mb
        da = a[i] - ma
        sbb += db * db
        sab += da * db
```
`mat` is the module's own **append-only** buffer — its docstring states rows "never move once written". So
`mb` and `sbb` for row `j` are constants of that row, yet both are recomputed on every subsequent candidate.

**Why it is wrong / costly:** the dedup scan is O(K²) comparisons over up to ~200 engineered columns
(the module's own figure), so row `j`'s mean and variance are recomputed up to K times. Total redundant work
is O(K²·n) — the same order as the `np.corrcoef` calls this kernel was written to replace. Caching `(mb,
sbb)` at append time (O(K·n) once) collapses the inner work from **two reduction passes per comparison to
one** (only `sab` remains). Separately, `da = a[i] - ma` is recomputed inside the `j` loop, i.e. k·n extra
subtractions for a vector that is constant across the whole call — centring `a` once into a scratch before
the prange removes them.

This is exactly the case where a REJECT would be wrong for the right-looking reason: the kernel HAS
`@numba.njit(cache=True, parallel=True)` and a real `prange`, so it passes the decorator check — but the
lever here is not parallelism, it is redundant recomputation, which the decorator does nothing about.

**Fix:** extend `one_vs_many_abs_corr_masked` to take `row_mean` / `row_ss` arrays alongside `buf`
(maintained by `_eng_dedup_scan` at the same place it does `_eng_next_free_row += 1`), and pre-centre `a`
once. The near-constant guard `sbb <= 1e-24 * n` moves to the append-time computation unchanged.

**Bench plan:** `bench_eng_dedup_batch_corr.py` already exists for this kernel (it holds the
2026-07-13 0.88× vstack rejection) — add the cached-moment variant to it. Sweep K ∈ {20, 50, 100, 200},
n ∈ {10k, 100k, 1M}, warm, best-of-10, measuring the WHOLE dedup pass (not one kernel call — the win is
amortised across the K² comparisons, so a single-call microbench understates it). Also report the full
`_eng_dedup_scan` wall inside a real fit. **Expected win, unmeasured.**

**Bit-identity risk:** LOW-MEDIUM. The per-row `mb`/`sbb` values are computed by the *same* sequential
reduction, just earlier, so they are bit-identical. Pre-centring `a` changes `da` from
`a[i] - ma` computed inline to a value read from a pre-computed array — the *same* subtraction, same
rounding, so also bit-identical. The one thing to verify explicitly is the near-constant guard: it must fire
on exactly the same rows as before (assert on a synthetic all-equal rank row).

**Test:** `test_one_vs_many_abs_corr_cached_moments_bit_identical` — `==` (not `allclose`) against the
current kernel across 30 scenarios incl. a near-constant row, an inactive-masked row, and K=1.

---

### PERF-8 — `audit_degenerate_columns` holds two full p×n copies of the frame before its own width cap is consulted, then allocates two more p×n arrays in the standardise step; the cap counts COLUMNS, never BYTES  [P2]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_degenerate.py:176, 199-271`

**What:**
- `:213-217` — for every numeric column, `numeric_cols.append((name, values.astype(np.float64), finite))`
  retains a float64 copy **and** a bool mask of the full column. By the time the loop ends, the function
  holds p·n·8 + p·n·1 bytes of live copies.
- `:227` — only THEN is `len(live) > max_collinearity_cols` (default 4000) checked and the pass skipped. The
  memory has already been spent; the cap protects only the GEMM, not the retention.
- `:242-248` — `M = np.empty((len(live), n_rows))` is a THIRD p×n array, and `col = v.copy()` makes a full
  n-element copy per column even on the common `fin.all()` path where the copy is never modified.
- `:256` — `M = np.where(good[:, None], M / np.where(...)[:, None], 0.0)` materialises the division result
  AND the `np.where` result: two more p×n float64 temporaries, with the old `M` still live until rebinding.

**Why it is wrong / costly:** the docstring itself calls this "a PURELY DIAGNOSTIC scan that never influences
selection". At p = 4000 (right at the cap) and n = 1e6 a single p×n float64 array is 32 GB; the code's peak
holds four to five of them. At n = 1e7 even p = 100 is 8 GB per array. CLAUDE.md's memory rule is explicit:
*"Eager format conversion ... gate on byte size (~2 GB)"* and *"never `.copy()` a frame to work around a
bug"*. A column-count cap cannot express that gate at all — the same 4000 columns is 3.2 GB at n = 100k and
320 GB at n = 1e7.

**Fix:**
1. Replace `_COLLINEARITY_PASS_MAX_COLS` with a **byte budget** (`p * n * 8 <= _COLLINEARITY_PASS_MAX_BYTES`,
   default ~2 GB per the repo's own rule), keeping the column count as a secondary cheap cap. Check it
   **before** the retention loop starts appending (n is known up front).
2. Do not retain per-column float64 copies at all: stream. Decide participation in pass 1, then in pass 2
   write straight into `M[k, :]` from the source column — `M[k, :] = values` already casts, so `col =
   v.copy()` is unnecessary; take the fill branch only `if not fin.all()`.
3. Standardise **in place**: `np.divide(M, stds[:, None], out=M, where=good[:, None])` then
   `M[~good, :] = 0.0`. Removes two p×n temporaries.
4. Optionally chunk the Gram: `corr` itself is p×p (fine at p ≤ 4000: 128 MB), but `_gram_matrix`'s cupy
   branch (`:169`) uploads the whole p×n `M` to a 4 GB card — that upload needs the same byte gate.

**Bench plan:** `bench_degenerate_audit_memory.py`. Peak RSS (`tracemalloc` + process RSS) and wall for
`audit_degenerate_columns` at (p, n) ∈ {(200, 100k), (1000, 1M), (4000, 1M)}, current vs. streamed, plus a
guard run at (4000, 1e7) that must now SKIP rather than allocate. Report peak RSS as the headline metric —
this is primarily a memory finding. **Expected win, unmeasured.**

**Bit-identity risk:** LOW for the streaming/in-place changes (`np.divide(out=)` and the direct assignment
perform the identical arithmetic in the identical order). MEDIUM for the byte gate: it CHANGES WHICH FITS
report `collinear_with` reasons. Since the reasons are diagnostic-only (per the docstring) no selection
moves, but a test asserting the reason is present on a large fixture would flip — re-frame it to assert on a
fixture under the byte budget.

**Test:** `test_degenerate_audit_skips_collinearity_pass_on_byte_budget` (asserts skip + log at a shape over
budget, and that `all_nan`/`constant`/`duplicate_of` reasons are still produced) +
`test_degenerate_audit_inplace_standardise_matches_reference` (identical `degenerate` dict).

---

### PERF-9 — `_pairwise_modular_resident` builds and uploads an (n, n_perm) int64 host index matrix per call — 960 MB at n = 1e7 on a documented 4 GB card  [P2]

**Where:** `src/mlframe/feature_selection/filters/_pairwise_modular_resident.py:206-210`

**What:**
```python
inv_idx = np.empty((n, n_perm), dtype=np.int64)
for i, perm in enumerate(perms):
    inv_idx[:, i] = np.argsort(np.asarray(perm))
inv_g = cp.asarray(np.ascontiguousarray(inv_idx))
code_mat = cp.ascontiguousarray(codes_r[inv_g])
```

**Why it is wrong / costly:** three compounding costs on the path whose entire purpose is to avoid H2D
traffic ("the WIN is the codes-matrix replaces 12 residue uploads", `:198`):
- `n · n_perm · 8` bytes host-side **and** device-side: 96 MB at n = 1e6, 960 MB at n = 1e7 — on the GPU
  CLAUDE.md's 2026-08-05 note documents as a 4 GB card running with as little as **1.25 GB free during this
  exact workload**. The optimisation trades 12 residue uploads for one index upload that is 12× the size of
  a residue column and 8 bytes/element wide.
- `inv_idx[:, i] = ...` writes a COLUMN of a C-order array — every element strided by `n_perm·8` bytes.
  This is precisely the antipattern `_mrmr_degenerate.py:236-241` documents and fixed ("measured ~13 ms/column").
- The `np.argsort` is PERF-3 again.

**Fix:** (a) apply PERF-3's scatter; (b) build `inv_idx` as `(n_perm, n)` row-major and transpose the view
for the gather (each write becomes contiguous), or drop the matrix entirely and gather column-by-column on
device into a preallocated `(n, n_perm)` device array reusing one `(n,)` index buffer; (c) use `int32` when
`n < 2**31` — cupy fancy-indexing accepts int32 indices and it halves both the host allocation and the H2D
bytes; (d) best: upload the 12 `perms` (same size problem) — or better, invert on device
(`inv[perm] = arange(n)` as a cupy scatter), so nothing but the perms crosses H2D and the inverse never
exists on the host. Combined with (c) the H2D volume drops ~2× and the host allocation vanishes.

**Bench plan:** `bench_modular_resident_index_upload.py` — measure **three ways separately** per CLAUDE.md's
GPU rule: host-input, GPU-resident-input, and GPU-with-H2D, at n ∈ {200k, 1M, 5M}, `n_perm=12`, on a quiet
machine (`nvidia-smi` check first) with novel seeds (MRMR memoizes fits by content-hash). Report H2D bytes
and `cupy.cuda.Device().mem_info` high-water alongside wall. **Expected win, unmeasured.**

**Bit-identity risk:** LOW. The scatter-inverse is exactly equal (PERF-3). int32 indices index the identical
elements below 2³¹ — add an explicit `n < 2**31` gate, not an assumption. Device-side inversion produces the
same index array (integer scatter, no float arithmetic). Assert `==` on the resulting `code_mat`.

**Test:** `test_modular_resident_code_matrix_identical_across_index_dtype_and_layout`.

---

### PERF-10 — `_v_finite_mask = np.ones(n, bool)` reallocated once per outer iteration of the monotone-twin scan, for an all-True constant  [P3]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_friend_graph_and_redundancy/_group4.py:291`

**What:** `_v_finite_mask = np.ones(_mt_ranks[_v].shape[0], dtype=np.bool_)` sits inside
`for _v in _raw_sel_mt:` but its value never depends on `_v` — every rank vector in `_mt_ranks` was already
proven fully finite at `:265` (`np.all(np.isfinite(_cv))`).

**Why it is costly:** one n-byte allocation + fill per selected raw column. At n = 1e7 and 200 selected raws
that is 2 GB of transient allocation for a constant. Small next to the kernel work (the loop's own comment
correctly notes it is bounded by the final selected count), hence P3 — but it is free to fix.

**Fix:** hoist one `_mt_all_finite = np.ones(_mt_n, dtype=np.bool_)` above the loop.

**Bench plan:** not worth a dedicated bench; fold into PERF-8's RSS measurement of a wide fit.

**Bit-identity risk:** NONE — the kernel only reads the mask.

**Test:** covered by the existing monotone-twin tests; add `test_monotone_twin_mask_allocated_once` spying
`np.ones` call count if a regression pin is wanted.

---

### PERF-11 — three thread-local dispatch getters and one `np.array(selected_vars)` rebuild per candidate, inside the per-candidate scoring frame  [P3]

**Where:** `src/mlframe/feature_selection/filters/evaluation.py:668, 684-686`

**What:** `selected_vars=np.array(selected_vars, dtype=np.int64) if selected_vars else np.empty(0, ...)`,
plus `use_su=use_su_normalization()`, `use_jmim=use_jmim_aggregator()`,
`use_mm=(use_mi_miller_madow() and not use_su_normalization())` — note `use_su_normalization()` is called
**twice** in the same argument list. All four values are constant within a greedy round.

**Why it is costly:** CLAUDE.md: *"Hoist the dispatch decision out of hot loops (~4 µs/call overhead adds
up)."* Four thread-local lookups plus a list→ndarray conversion per candidate; with 1e5 screened candidates
per round that is ~1e5 × (4 getters + one O(k) allocation). Modest against the njit `evaluate_gain` body,
hence P3 — but `_evaluation_driver.py` is already the established place for exactly this kind of
once-per-round hoist (it does it for RelaxMRMR at `:370-385`), so the wiring cost is near zero.

**Fix:** resolve all four in `_evaluation_driver` once per round and pass them down; at minimum, bind
`_use_su = use_su_normalization()` to a local and reuse it for both `use_su=` and the `use_mm=` expression.

**Bench plan:** microbench the argument-construction block standalone (cProfile mis-attributes compiled
kernel time to this Python frame, so trust only the standalone number per CLAUDE.md's GPU/profiling rule),
then a full-fit wall at p = 5000 / n = 200k. **Expected win, unmeasured; likely small.**

**Bit-identity risk:** NONE provided the hoist is per-round (the getters are thread-locals that the fit
wrapper sets once per fit — `_mrmr_class.py:3903`).

**Test:** `test_dispatch_flags_resolved_once_per_round` — spies the getter call count across one round.

---

### PERF-12 — `_all_pairs_precomputed` re-enumerates the full C(k,2) pair space in Python purely as a membership check  [P3]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pairmi.py:243`

**What:** `all((p in cached_MIs or p in cached_confident_MIs) for p in combinations(numeric_vars_to_consider, 2))`

**Why it is costly:** allocates and hashes up to C(k,2) tuples. The comment at `:241-242` acknowledges
"thousands-to-low-hundred-thousands pairs" and argues it is "a fast membership scan, not a compute pass" —
true, but the batch precompute immediately above (`:198-207`) already knows exactly how many pairs it
prefilled (`_batch_prefill_count`) and how many exist (`_n_pairs_batch`). The equality
`_batch_prefill_count == n_pairs` answers the same question in O(1) when the batch path ran. Note the `all()`
short-circuits on the FIRST miss, so the worst case is the all-cached case — which is the case the comment
says is routine ("batch-prefilled 31125/31125").

**Fix:** short-circuit on the counters when the batch path succeeded; keep the enumeration as the fallback
for the exception branch at `:216-227`.

**Bench plan:** microbench the `all(...)` at k ∈ {250 (31k pairs), 1000 (500k pairs), 2000 (2M pairs)}.
**Expected win, unmeasured; small in absolute terms, grows quadratically in k.**

**Bit-identity risk:** NONE if the counter path is only taken when the batch dispatcher reported success and
covered the same pair set. Verify that `dispatch_batch_pair_mi_chunked` enumerates exactly
`combinations(numeric_vars_to_consider, 2)` — **unverified**; settle by asserting
`_n_pairs_batch == n_pairs` in the test.

**Test:** `test_all_pairs_precomputed_counter_matches_enumeration`.

---

### PERF-13 — the loky pair-MI pool branch (~160 LOC incl. its batched-CPU retry) is unreachable at the production default  [P3]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pairmi.py:39, 252-253, 282-440`

**What:** `_LOKY_POOL_MIN_FE_NPERMUTATIONS = 20`; `_below_perm_floor = fe_npermutations < 20`; the guard at
`:253` routes to serial when `_below_perm_floor`. The production default is `fe_npermutations=3`
(`_mrmr_class.py:841`), so `_below_perm_floor` is always True and the entire `else` branch — the loky pool,
its timeout machinery, and its batched-CPU-retry fallback at `:394-440` — never executes in a default fit.

**Why it is reported:** this is the gate-shape #4 the brief asks about, and the answer is "effectively never
taken — **correctly**". The constant's own comment is honest: the pool LOSES at every measured pair count
(0.03×–0.38×) and "20 is a conservative floor extrapolated well above the tested 3-permutation regime (not
itself measured to win)". So the dead branch is the SLOW path, and disabling it is right. Reporting it so the
next wave does not mistake it for a live path worth optimising, and so the maintenance cost is visible:
~160 LOC of untested-in-production concurrency code (loky spawn, `CUDA_VISIBLE_DEVICES` initializer,
wall-clock timeout, `wait=False` abandonment) whose only exercise is a test that must set
`fe_npermutations >= 20` artificially.

**Fix:** none required for performance. Recommend either (a) a comment at the branch head stating it is
unreachable at the default so nobody profiles it, or (b) deletion with the constant, the measurement and the
`bench` filename retained per the repo's REJECTED≠DELETED rule.

**Bench plan:** n/a (no change proposed). If the branch is kept, one re-bench at `fe_npermutations` ∈ {20,
50, 100} would at least establish whether the extrapolated floor is real — currently it is unmeasured.

**Bit-identity risk:** n/a.

**Test:** `test_loky_pair_mi_pool_not_entered_at_default_fe_npermutations` — spies pool construction and
asserts the serial path is taken with the default knob.

---

### PERF-14 — `np.isfinite(_arr_k)` recomputed per (candidate, kept) pair in the dedup slow path although `_eng_fully_finite` is already cached per column  [P3]

**Where:** `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_eng_dedup_scan.py:114`

**What:** `_mask = _fin_c & np.isfinite(_arr_k)` — a full O(n) `isfinite` pass over the kept column on every
comparison, while `_eng_fully_finite[_kc]` (used at `:95`) already records whether that column is fully
finite, and `_fin_c` is already cached for the candidate.

**Why it is costly:** O(K²·n) `isfinite` evaluations in the worst case. It is P3 only because the batched
fast path at `:95-110` already skips this loop entirely for fully-finite candidates — the slow path is
reached only when the candidate has NaN. On a NaN-heavy engineered pool (ratio/log/inverse families produce
NaN routinely) that is not rare.

**Fix:** cache `np.isfinite(arr)` per column alongside `_eng_fully_finite`, or short-circuit
`if _eng_fully_finite[_kept_col]: _mask = _fin_c`.

**Bench plan:** fold into PERF-7's whole-dedup-pass bench with a NaN-injected variant (30 % NaN in half the
engineered columns). **Expected win, unmeasured.**

**Bit-identity risk:** NONE — `_fin_c & all_true == _fin_c`.

**Test:** `test_eng_dedup_reuses_cached_finite_masks` — asserts identical `degenerate`/`keep` outcome on a
NaN-bearing pool and spies `np.isfinite` call count.

---

### PERF-15 — `mat[:, i] = ...` column-writes into a C-order `(n, n_perm)` array at all three `_perm_null_hi` sites: the strided-write antipattern this repo already documented and fixed elsewhere  [P3]

**Where:** `_integer_lattice_fe.py:216-218`, `_conditional_gate_fe.py:333-335`,
`_pairwise_modular_resident.py:206-208`

**What:** `mat = np.empty((n, n_perm), dtype=np.float64)` then `mat[:, i] = ...` in a loop — every element of
each write is strided by `n_perm · 8` bytes.

**Why it is costly:** `_mrmr_degenerate.py:236-241` documents this exact case: *"the previous `M[:, k] = col`
wrote a column of a C-order array — every element strided by K*8 bytes, the classic column-into-row-major
antipattern (measured ~13 ms/column on a 99401×~500 frame, 15 % of this whole scan's wall)"*. The same
diagnosis applies here with `K = n_perm = 12`. P3 rather than P2 only because `n_perm` is small (stride 96
bytes still straddles cache lines but is far better than the 4000-wide case that was measured).

**Fix:** allocate `(n_perm, n)` C-order and pass `mat_T = mat.T` to `_mi_classif_batch`. **Caveat, and the
reason this is not a trivial change:** `_mi_classif_batch` → `plugin_mi_classif_batch_dispatch` may require a
C-contiguous `(n, k)` input; a Fortran-ordered view would then trigger an internal `ascontiguousarray` copy
and net out worse (or, on the cupy branch, silently re-copy on upload). **Unverified** — settle it by reading
the dispatcher's input handling before implementing, and measure both layouts.

**Bench plan:** fold into PERF-3's `bench_perm_inverse_scatter.py`: four variants (argsort+C-order,
scatter+C-order, argsort+F-order, scatter+F-order) at n ∈ {200k, 2M}, so the layout and the inversion levers
are attributed separately and the F-order copy risk shows up if it exists.

**Bit-identity risk:** NONE for the layout change itself (identical values, identical order into the
kernel); the risk is purely performance (a hidden re-copy).

**Test:** covered by PERF-3's bit-identity test; add `test_perm_null_matrix_layout_is_contiguous_for_kernel`
asserting the array handed to `_mi_classif_batch` is C-contiguous.

---

## Verified-clean (already optimised)

Checked, with the marker actually confirmed as covering the path — so the next wave does not re-audit blind.

| Path | Marker confirmed | Note |
|---|---|---|
| `_mrmr_fe_step/_step_core.py::_run_fe_step_impl` body | n/a — third-party-free pure orchestration | Its own bench-note (`:13-24`) records `line_profiler` over the body: glue = 0.166 ms = 0.149 % of 11.16 s; 99.85 % is in the 9 carved-sibling call sites. Genuinely at floor. **Do not re-profile the body.** |
| The three `_perm_null_hi` siblings' *batching* (the 2026-08-05 open item) | `_mi_classif_batch` → `plugin_mi_classif_batch_dispatch` (njit `prange`) | **Prior-wave follow-up CLOSED**: `_conditional_gate_fe.py:326-338` HAS been ported (the 2026-08-05 note said it "has NOT yet been checked/ported"). All three now batch. Their remaining issues are PERF-1 (pairwise_modular's HOST path), PERF-3 and PERF-15 — not the batching itself. |
| `evaluate_gain` — the core per-candidate k-loop | `@njit` (called at `evaluation.py:651-686` with a `numba.typed.Dict` + counter array) | The redundancy/JMIM inner loop is fully inside the kernel; no Python round-trip per selected var. The findings around it (PERF-5, PERF-11) are in the *argument construction*, not the loop. |
| `_eng_dedup_batch_corr._one_vs_many_pearson_abs_masked_njit` | `@numba.njit(cache=True, parallel=True)` + `prange` over `k` | Parallelism is genuinely present and covers the comparison loop; it also carries a real `bench-attempt-rejected` note (2026-07-13, vstack variant, 0.88×) that tested a *different* lever. PERF-7 is the untested lever (redundant recompute), not parallelism. |
| `_step_pairmi.compute_pair_mis_and_floor` batch precompute | `dispatch_batch_pair_mi_chunked` (CUDA / njit-prange by size, RAM-bounded row-block chunking) | Runs unconditionally at `n_pairs >= 8`, routinely covers 100 % of the pool; the C(k,2) space is never fully materialised. |
| `_mrmr_sis_screen.py` | `_mi_classif_batch` njit + `second_moment_propensity` corr-matmul; chunked at `:299`; RAM-budgeted at `:84`, `:185` | Its docstring's claim that the glue holds nothing to optimise checks out; the two O(n log n) `np.unique` calls at `:261-280` are on `y` only and run once per screen. |
| `_step_score.py`'s pandas `X.copy()` sites (`:548`, `:898`, `:1008`) | n/a — memory, not kernel | Guarded by `_x_is_owned`, so three potential whole-frame copies collapse to at most ONE, taken only when a pair actually produced a column. Correct per the "fit must not mutate caller input" contract and consistent with the repo's memory rule. |
| MRMR's polars→pandas bridge | `get_pandas_view_of_polars_df` (zero-copy Arrow-backed) | Per CLAUDE.md's explicit "don't fix this again" rule. Not touched. |
| `_step_pairs_rank.py` batched usability / pair-scan | `batch_pair_usability_corr_gpu` njit(parallel=True), `prange` over the flattened pair space (`:165`); per-call memos at `:451`, `:470`; codes memoized per column at `:265` | Already batched with documented per-call memoization. |
| `_group4.py` monotone-twin O(k²) scan | `_abs_corr_finite_njit` (serial, deliberately) | The serial-not-batched choice is justified in-line (`:283-289`) by the loop being bounded by the FINAL selected count. Accepted. Only PERF-10 (the hoistable mask alloc) applies. |

## Skipped per the "<100× per fit or <1 % of wall" rule

- `_mrmr_fingerprints._mrmr_y_corr` (`:373-391`): `np.corrcoef` on a ≤`max_sample`-element strided sample,
  called once per fit for the identity-cache gate. Its own docstring says "never on a hot per-row path" —
  confirmed by the `arr[::step][:max_sample]` subsampling at `:366-367`.
- `_mrmr_artifacts.py`, `_mrmr_explain.py`, `_mrmr_stability_report.py`, `_mrmr_passthrough.py`,
  `_mrmr_setstate_defaults.py`, `_mrmr_config_dataclasses.py`, `_mrmr_param_constants.py` — config /
  reporting / serialisation surfaces, once per fit, no n-scaled loop.
- `mrmr/_mrmr_class.py:3266-3332` (`__getstate__` / setstate defaults `deepcopy`): once per pickle, and the
  `deepcopy` is confined to `(list, dict, set)` defaults, never an array or frame.

## Cross-references for the cluster agents

- PERF-1 / PERF-3 / PERF-15 all land in the FE-family files (`_pairwise_modular_fe`, `_integer_lattice_fe`,
  `_conditional_gate_fe`) — whoever owns that cluster should not treat `_perm_null_hi` as closed.
- PERF-4 / PERF-5 / PERF-11 all land in `evaluation.py` + `_evaluation_driver.py`, which are OUTSIDE the
  `*mrmr*` path glob and so may be in no cluster's file list. They are the per-candidate scoring hot path.
- PERF-2 spans six cascade files at once; it should be fixed as ONE shared helper, not six times.
