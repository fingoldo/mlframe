# MRMR FE perf notes (agentic optimization /loop)

Profiling findings to guide the optimization loop. NOTE: prior memory says mlframe perf is mature
(metric kernels / preprocessing / io are bench-tuned; small-n grinding yields no wins) -- so treat
these as CANDIDATES to verify, not assumed wins. Always bench before/after; document no-wins.

## cProfile: MRMR().fit on CASE2 n=20000 (2026-06-13, tottime, mlframe-side)

| tottime | ncalls | function | note |
|--------:|-------:|----------|------|
| 3.88s | 18 | `discretization/_kernels.py:_searchsorted_2d_right_njit_parallel` | njit_parallel; **18 calls** -- one per fit/FE step? possible redundant re-discretisation |
| 3.44s | 18 | `discretization/_kernels.py:_quantile_edges_2d_njit` | same 18× cadence -- edges recomputed each step |
| 2.51s | 2 | `_feature_engineering_pairs/_pairs_materialise.py:_materialise_chunk` | FE pair materialisation |
| 2.34s(cum) | 34 | `_orthogonal_univariate_fe/_orth_extra_basis_fe.py:_detect_fourier` | Fourier-basis detection in orth FE |
| 0.57s | 1564 | `_orth_extra_basis_fe.py:_corr_sq_centered` | many small calls |

Total fit ~50s (n=20k, includes FE + escalation + orth basis).

### VERDICT (2026-06-13, investigated): both top hotspots are INTRINSIC -> NO perf win

- `discretize_2d_quantile_batch` (the 18x calls): each call discretises a FRESH chunk of MATERIALISED
  FE-candidate columns (`chunk_buffer[:, :col]` in `_pairs_chunks.py:252`), NOT a re-discretisation of
  the same matrix. Distinct data per call -> not redundant; the kernel is already njit_parallel. REJECTED.
- `_detect_fourier_freqs_for_col` (34 calls): per-candidate-column FFT frequency detection for the
  adaptive-Fourier FE generator (`_orth_extra_basis_fe.py:702/757`) -- intrinsic per-column work. REJECTED.

Consistent with the standing finding that mlframe FS/FE perf is MATURE (kernels njit-tuned; small-n
yields no wins). The optimization loop should focus on FE QUALITY / correctness, not perf re-grinding,
unless profiling an UNCONTENDED large-n run surfaces a genuinely redundant call.

### (superseded) original lead candidate
The 18x `_searchsorted_2d_right` + `_quantile_edges_2d` calls (7.3s combined, ~15% of fit) suggest the
discretised matrix / quantile edges are RECOMPUTED per FE step rather than cached+extended when FE
appends columns. If the existing columns' edges are stable across steps, caching them and only
discretising the NEWLY-appended engineered columns would cut this. VERIFY first: confirm the 18 calls
are on overlapping column sets (redundant) vs genuinely new data each time; bench the cache; ensure
bit-identical selection (edges must be frozen identically). Mature-perf caveat: may already be
intentional (FE changes nbins/strategy per step) -- check before assuming redundancy.

---

## 2026-06-16 -- 400k / fe_max_steps=3 cProfile (WALL=585s, clean composite selected)

CPU hotspots (cumtime / tottime):
* **conditional-gate FE = #1 lever (~31% of wall, ~180s+):** cheap_conditional_gate_scan
  (_conditional_gate_fe.py:351, 62s TOTTIME / 181s cumtime) + _gate_grid_mi (:201, 80s)
  + best_existing_op_mi (:143, 46s) + _baseline (:409, 37s). PRIME speed target at scale.
* orth-family MI batch _mi_classif_batch (945 calls, 166s cumtime) -- MI estimation, harder.
* binned_numeric_agg (93s; per_cell_stats_bincount 21s tottime).
* MDLP supervised binning _mdlp_best_split_njit (34s tottime).
TODO (/loop): optimize the conditional-gate scan -- the grid-MI sweep dominates; check whether
the per-(tau,op) MI grid recomputes redundant baselines / can prune the op x threshold grid early.

## 2026-06-16 -- large-n MEMORY: all FE candidates materialized at once

1M x fe_max_steps=3 OOMs a 16GB box: the ~46 FE families EACH append their candidate columns into one
growing `X` frame, so the discretised screening matrix holds (n, raw + EVERY family's candidates)
simultaneously. Mitigations landed: free dead per-family concat-frames before categorize_dataset
(28dc86d4); Haar-wavelet legs float32 (5e8ec8c6). FURTHER: orth-poly working arrays kept float64
(orthogonality); a genuinely chunked/streamed candidate screen (preserving zero-marginal SYNERGY
candidates -- can't pre-discard by marginal MI) is the deeper fix if 1M must fit.

## 2026-06-16 -- polars input gets ZERO feature engineering

All FE families guard `isinstance(X, pd.DataFrame)` and SKIP (with a warning) on polars input -- only
raw MI screening runs, no engineered features. TODO: auto-convert polars->pandas at fit entry when FE
is enabled (fit AND transform consistently) so polars users get full FE (one materialization; the FE
needs pandas anyway -- strictly better than silently skipping FE).

## Scaling profile 2026-06-16 (100k/400k/1M) -- the REAL large-n hotspot is the MI permutation null

bench_scaling (full MRMR.fit, cProfile, fe_max_steps=2) settled two things measured, not guessed:

* **FE candidate CONSTRUCTION is NOT the bottleneck and its share SHRINKS with n:** 12.2% (100k) ->
  3.9% (400k). Absolute construction grew ~linearly (30s->93s) while MI exploded ~17x (130s->2188s).
  Micro-bench (microbench_fe.py, 1M rows): njit/prange beats numpy only ~1.0-1.3x on arithmetic
  (memory-bound) and ~2.5x on transcendentals (log/sin) -- so even a big construction speedup buys a
  few % of wall. CONCLUSION: do NOT re-platform FE construction to numba FOR SPEED (polars/OOM are
  separate, legitimate reasons). This confirms the older `_conditional_gate_fe.py:420-424` bench
  (construction ~0ms of the MI-bound scan) holds AND strengthens at scale.

* **The dominant large-n cost is the MI PERMUTATION NULL:** at 400k the cupy `argsort` permutation
  generator (gpu.py:mi_direct_gpu_batched, and the CPU prange twin) was ~72% of the fit -- thousands
  of O(n) shuffles across the FE scan. (1M OOM'd: peak 9.87GB at 400k on a 15.9GB box; the argsort
  blowup is amplified by the dev GTX 1050 Ti -- ~30x faster on a strong GPU -- but it stays the top
  hotspot by rank regardless.) FIXED 2026-06-16 (commit cda55bcb): the analytic large-n null
  (Miller-Madow null mean + G-test p, `_analytic_mi_null.py`) replaces the shuffles at n>=50k,
  measured 24-35x on the null computation, identical MI/null + decision-equivalent p. NEXT large-n
  lever if more is needed: the joint-hist / binning path (`_binned_numeric_agg_fe`, MDLP), not
  construction.

## Analytic-null equivalence + minimum-permutations study (2026-06-16)

Two questions settled empirically (D:/Temp/equiv_study.py, minperm2.py; pinned by
test_analytic_mi_null.py::test_permutation_converges_to_analytic):

* **The analytic null IS the permutation null's nperm->infinity limit, not an approximation.**
  Both estimate the same independence null; the permutation is a Monte-Carlo estimate (error
  ~1/sqrt(nperm)). At n=80k the permutation null_mean converges 0.00049 (32 perms) -> 0.00051
  (= analytic) and p converges 0.25 -> 0.36 (= analytic, |dp| -> 0.0002); for genuine signal p=0 at
  every nperm. So the analytic value is MORE accurate than any finite-nperm run (zero MC noise).

* **Minimum permutations for a stable keep/reject DECISION: ~32.** Panel of 24 candidates
  (strong/weak/borderline/noise) at n=80k, keep-set vs the analytic ground truth: nperm<=8 admits
  1-3 Monte-Carlo noise false-positives; nperm=16 matches but only 2/3 seeds; **nperm=32 matches the
  reference on 3/3 seeds** and is rock-stable above. This independently validates the existing
  ``_NULL_MEAN_MIN_PERMS=32`` floor as the minimum safe permutation budget for the n<50k path (below
  the analytic threshold). Above 50k the analytic null (= the nperm->inf limit) supersedes it.
