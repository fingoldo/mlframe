# Fused pair / triple FE kernel: design (2026-11-03)

## Why
Utilization baseline, 100k x 100, every family, GPU fit (stage_sample5, `all_families_gpu_100k.md`): GPU busy mean 18.5 %, median 4 %, under 10 % for
61 % of the 101 s; CPU 44.8 % of 8 cores. The pair arithmetic stages that already run on the device are the busy ones (pair MI 74 %, pair emit 59 %);
the rest of the fit is many host stages of 1-3 s each. Each pair family today makes its own pass over the operands: materialise -> bin -> MI -> gate,
with host glue, small launches and syncs between them (1666 scalar D2H reads in a 25 s nsys window). The request: after the factors and the target(s)
are binned, try every pair / triple family inside one kernel.

## Scope
In (closed-form functions of 2-3 operand columns, scored by binned MI against the target):
- binary ops of the pair search: mul, add, sub, div, max, min, abs_diff, signed, ratio_abs (`_fe_materialise_kernel` op codes 0-8), over the unary
  transforms of each operand (the unary x binary grid of `_feature_engineering_pairs`);
- ratio / log-ratio, offset product (`_offset_product_fe`), pairwise modular (period residues of a - b, a / b);
- polynomial / orthogonal-basis cross terms of pairs and triples (He_i(a) He_j(b), triplet / quadruplet cross basis);
- the cat x num pair cells (`_cat_pair_fe`) when the categorical side is already integer-coded.
Out (structurally not one pass over 2-3 columns): group-by aggregates (grouped quantile, composite group agg, group distance), out-of-fold target
encodings, the adaptive Fourier / chirp detectors (iterative), bootstrap and permutation-null loops (they reuse the kernel's histogram step instead),
the LightGBM surrogate seeders. These go to the device separately.

## Kernel
One block per (pair or triple, op group): the operand columns (float32, n rows) are read once, coalesced, in tiles. Per tile, every op of the family
grid is evaluated in registers (unary transforms from a small switch, then the binary op), so a value never goes to global memory.
Binning needs each candidate's quantile edges. Two variants, both kept:
- EXACT (default): pass 1 writes the candidate values of a block of candidates to a scratch (k, n) buffer sized from the pool and runs the radix select
  (`radix_select_f32_v4`) for the edges; pass 2 recomputes the values (cheaper than reading them back) and bins with the f32-edge rule of `bin_codes_f32e`
  (exact vs the double compare). Same codes as today's pipeline -> same MI bit for bit.
- SAMPLED: edges from a fixed row subsample (the decision subsample the pair search already draws), one pass; selection-equivalence check needed, opt-in.
Histogram: per candidate a (nbins_x x n_classes) joint count in shared memory, one per target when several targets are scored (multi-target: the
operands and edges are target-independent, only the histograms are per target). MI from the counts in the same block (the `mi_from_codes` formula),
the analytic G-test floor / occupied bins in the epilogue (`_fused_observed_mi_and_bins`), so only (K,) MI, bins and pass flags leave the device.

## Multi-target
`multioutput_strategy` currently runs one full fit per target (`_fit_multioutput`: clone + fit per column). With the fused kernel the target-free part
(operand binning, candidate values, edges) is built once and every target's histogram is filled in the same pass; the per-target greedy / gates run on
the resulting (targets x K) MI table.

## Dispatch and fallbacks
Per family, KTC spec over (n, K, ops) choosing fused / current path; column blocks sized from the pool (`_binagg_candidate_block` style); CPU twin =
today's per-family path, which also stays the reference in the tests.

## Validation
For every family moved: codes and MI equal to the current path (exact variant) on frames with ties, constants, NaN-scrubbed values and
heavy tails; selection identity on the 50k and 100k all-families fits (same process, fit cache cleared); held-out AUC
(`D:/Temp/sel_quality.py`) unchanged; utilization (stage_sample5) and nsys scalar-read count reported before / after.

## Phases
1. Binary-op pair grid (ops 0-8 x unary grid) into the fused kernel, exact edges, single target. Replaces materialise -> bin -> analytic gate.
2. Multi-target histograms + `_fit_multioutput` sharing the target-free state.
3. Ratio / offset product / pairwise modular / cross-basis pairs and triples.
4. Sampled-edge variant (opt-in, selection-equivalence gated).
In parallel: the host-only families of the utilization table to the device (screen confirm / Fleuret recheck, grouped quantile scoring,
group distance, composite agg, CMIM / JMIM, k-fold OOF MI), and the non-mdlp `nbins_strategy` profiles at 100k.
