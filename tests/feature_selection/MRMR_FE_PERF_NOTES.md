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
