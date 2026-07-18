# Discretization bin-edge / quantization arithmetic audit (2026-07-16)

Scope: `src/mlframe/feature_selection/filters/discretization/_kernels.py` (primary), plus
`_discretization_edges.py`, `_discretization_dataset.py`, `discretization/__init__.py`,
`_gpu_resident_discretize.py`, `_adaptive_nbins.py`, `supervised_binning.py` (MDLP), and
`info_theory/_class_encoding.py::merge_vars`. This continues the prior pass
(`05_concurrency_and_statistics.md`) which explicitly deferred bin-edge arithmetic.

## Result: no correctness bugs found

Every candidate off-by-one / tie / NaN / degenerate-column / CPU-GPU-parity scenario tried
below reproduced bit-for-bit against its reference (numpy or the CPU sibling kernel). This is a
clean bill of health for the bin-edge arithmetic itself — not because it wasn't tried, but
because the module has already been through multiple prior audit waves (visible in the dense
`2026-05-xx Wave N fix` / `bench-attempt-rejected` comments throughout) that fixed exactly this
class of bug (NaN-poisoned percentiles, dtype-narrowing wraparound, degenerate min==max ranges,
NaN/real-bin collisions, CUDA K==1 percentile bug, etc.) before this pass started.

### What was empirically verified (all passed)

1. **`_quantile_edges_2d_njit` bit-identity to `np.percentile(axis=0)`** — swept
   float32/float64, n_rows∈{1,2,5,50,1000}, n_cols∈{1..20}, nbins∈{4..50}, including a
   constant column (min==max). `maxdiff == 0` in every case.
   ```
   PYTHONPATH=.../src python -c "... see /tmp/verify1.py logic ..."
   ```
   Output: no mismatches printed (the check only prints on `maxdiff>0`); constant-column
   `maxdiff: 0.0`.

2. **`_searchsorted_2d_right_njit` vs `_searchsorted_2d_right_njit_parallel` vs
   `np.searchsorted(..., side='right')`** — including a column with a **duplicate edge value**
   (tie) and a **NaN row**. Serial/parallel/numpy all matched exactly; the NaN row landed in
   the rightmost bin (`out=[3,0]`) matching `np.searchsorted`'s NaN-sorts-last convention on
   both kernels.

3. **`_quantile_codes_1d_njit`** (the fused 1-D kernel, currently unused in production per its
   own `bench-attempt-rejected` note but kept for re-bench) — verified against
   `np.searchsorted(np.nanpercentile(...)[1:-1], ..., side='right')` across n∈{2,3,5,50,1000},
   nbins∈{2,3,10,50}, and an explicit **values-exactly-on-a-bin-edge** case
   (`arr=[1..8]`, 4 quantile bins): codes `[0,0,1,1,2,2,3,3]` on both sides — the "value exactly
   on an edge goes to the higher bin" convention is applied consistently by both the edge
   computation and the edge application.

4. **MDLP (`supervised_binning.py`) Python-vs-njit backend agreement** — 30 random trials
   (n∈[50,500), n_classes∈[2,5), varying seeds) produced **identical edges** between the
   legacy pure-Python recursion and the production njit-fused recursion (0 mismatches).
   Also checked degenerate inputs: constant `x` → `[-inf, inf]` (no split attempted, no crash);
   all-same-class `y` → `[-inf, inf]`; a perfectly-separable two-cluster `x` → correctly split at
   the midpoint (`54.5`) between the clusters. The split-point-inclusion convention
   (`x[:best_idx+1]` = left, i.e. values `<=` the midpoint go left) is consistent between the
   two backends and matches the recursion's own downstream slicing.

5. **`categorize_dataset` end-to-end** (the actual `MRMR.fit` entry point), with
   `nbins_strategy='mdlp'` (the project default) and `missing_strategy='separate_bin'`:
   - A **low-cardinality integer column** (0..5) correctly gets one code per unique value
     (midpoint-edge branch in `_adaptive_nbins.py`), no NaN-bin collision.
   - A **continuous column with injected NaN** gets its own dedicated NaN bin (code 5, one
     past the max real code 4), never colliding with a real bin, per the Wave 9.1 fix
     documented at `_discretization_dataset.py:352-384`.
   - A **fully constant column** (all values equal) collapses cleanly to bin 0 for every row —
     no crash, no garbage codes, verified via `data.max(axis=0)`.
   - A **sparse-dominant column** (95% zeros, 5% scattered nonzero, the case the
     2026-05-31 "sparse-aware secondary fallback" in `_adaptive_nbins.py` targets) produced
     sorted, correctly-isolated edges: the dominant value (0.0) got its own bin (code 0)
     disjoint from the nonzero-range bins (codes 1-4).

6. **CPU/GPU parity for `_gpu_resident_discretize_codes` vs `discretize_2d_quantile_batch`**
   (CUDA is actually available in this environment, so this was run for real, not just read) —
   swept `(n, K, nbins)` ∈ {(1000,1,10), (1000,5,7), (50000,20,10), (37,3,5)} plus an explicit
   **K==1 constant column** (the case the code's own comment flags as a past CuPy
   `cp.percentile` bug for single-column input). All cases: `maxdiff == 0`, zero mismatched
   codes. The radix-select fast path, the `cp.percentile` fallback, and the fused
   `bin_codes` RawKernel binary search all stayed bit-identical to the CPU reference.

### Areas read but not independently re-verified

- `merge_vars` in `info_theory/_class_encoding.py`: on inspection this is joint-class melting
  (combining several already-discretized ordinal columns into one dense class id via a
  positional-radix encoding + empty-bin pruning), not bin-*edge* arithmetic — there are no
  edges or thresholds computed here, only integer remapping of existing codes. Its known past
  bug class (int8/int32 counter overflow on deep joints) was already fixed with an explicit
  int64 counter widening (`_class_encoding.py:54-64`) prior to this audit; no new issue found
  on inspection and it's out of this audit's stated scope (bin-edge arithmetic specifically).
- Knuth (`_knuth_best_M`) and Bayesian Blocks (`_bayesian_blocks_bin_edges`) degenerate-input
  guards (`n<1`, `a_max<=a_min`, all-uniform data forcing M>=2, BB's `_t_floor` tie-collapse
  guard) were read closely and are self-documenting with explicit prior-bug narration in their
  docstrings/comments; not independently re-run since they are not on the MRMR default path
  (`nbins_strategy='mdlp'`) and the prior wave already fixed and bench-validated them.

## Conclusion

No off-by-one, floating-point-boundary, NaN-corruption, degenerate-column, dtype-overflow, or
CPU/GPU-parity bug was found in the bin-edge / quantization arithmetic during this pass. The
module's density of `Wave N fix` comments suggests most of the low-hanging bugs in this class
were already caught and fixed in earlier audit rounds; this pass adds empirical (not just
static) confirmation that those fixes hold under the specific edge-case inputs this audit's
brief called out (ties on edges, NaN rows, constant/low-card/sparse-dominant columns, GPU K=1).
