Written for: mlframe maintainers working on MRMR feature engineering.

# Binned numeric aggregation: what the pair cap and the CMI gate dropped

Question asked: does `max_pairs=64`, which drops low-ranked (group, agg) pairs before any out-of-fold work, lose good features?

Probe: `src/mlframe/feature_selection/_benchmarks/bench_binagg_cap_gate_probe.py` (20000 rows, the project's `sigma(cell)` scenario with linear-signal decoy groups).

## Findings

| # | Finding | Evidence | Disposition |
|---|---------|----------|-------------|
| 1 | The cap removes the `(aux \| g)` pair when 11 decoy groups outrank `g` | 11 decoys: 0 columns at cap 64 before any gate, 4 at cap 400 | Not changed. No layout tried kept a column that survives the gates, so the cap cost nothing measurable. A layout where a gated column exists without the cap is still to be found |
| 2 | The CMI redundancy gate rejected `binagg_std(aux\|qbin(g))` in every scenario tried, with or without the cap | Default gates keep 0 columns for regression and classification variants of the spread target, periods 1, 2, 3 and 5 of `sigma(g)`, and all decoy counts | Fixed: see 3 |
| 3 | A rejected column is useful to a linear model | Ridge held-out R^2 on the raw columns 0.000, with the column 0.857 / 0.852 / 0.858 (seeds 0, 1, 2); boosted trees 0.893 / 0.889 / 0.893 raw, 0.891 / 0.888 / 0.892 with it | Fixed: `linear_usability` (default on) in `binned_numeric_agg_with_recipes`, module `_binned_agg_linear_usability.py` |
| 5 | The CMI gate kept aggregates of noise columns of an informative group in 7 of 30 runs | permutation null shuffled `y` over all rows | Fixed: within-strata permutation and a Gamma tail |
| 4 | Pure noise and a linear target that the raw columns explain keep nothing under the fix | `tests/feature_selection/fe/test_binagg_linear_usability.py` (noise: 3 seeds, 10 columns; linear target) | Pinned by tests |

## Why the CMI gate rejects the column

The gate scores `CMI(col; y | qbin(g), qbin(a))`. The column is a function of the cell of `g`, the same partition the gate conditions on, so it is redundant for any model that can use the raw group. That is the right answer for trees and the wrong one for a linear model.

## The fix

For a column the CMI gate rejects, fit class indicators of `y` linearly on the raw sources with and without the column, out of fold, and keep the column when its gain is higher than a random per-cell column's. No fixed acceptance threshold. Rows are subsampled to 100000.

Three versions were tried, and the first two were wrong:

1. Null = the candidate shuffled across rows. It broke the link with the group's cell, so with an informative group a noise column's per-cell values passed: 130 noise aggregates kept on a 30-column, 200000-row frame.
2. Null = per-cell values permuted among the cells, decided with a z bound on the null's mean and standard deviation. With about ten cells the correlation of a random per-cell column is bounded by one, so a family-wise z of four standard deviations lies outside what any permutation can reach: 0 of 12 seeds found the real column.
3. Current: a permutation test. The p-value is `(hits + 1) / (draws + 1)`, with hits the permutations whose gain is at least the observed one, on about `2 / level` draws made in batches with an early stop. The mean, std, skew and kurt of one (group, agg) pair are tested together with a single-step max-T adjustment (every permutation of the cells is applied to all of them), and the level 0.05 is split over the pairs (Bonferroni). Splitting it over every column puts the level below what a genuine column reaches: with ten cells its p-value is near 1e-4 to 1e-3, the resolution of the test, and power fell to about half.

Measured, 30000 rows, informative group, 10 noise columns, 20 seeds: the real spread column found in 17 of 20 runs, 0 noise columns kept in any run.

## Cost

200000 rows, 30 numeric columns, informative group, minimum of three runs each: 5.8 s without the check and 9.5 s with it. The first version cost 24 s more; per-pair state (subsample, folds, base design, per-(fold, cell) sums, base inverse) is built once per pair, and the training normal equations are solved through the Schur complement of the base block, so a permutation needs a few small `einsum` products and no linear solve (the 4x4 solves had been 8 of the 12 s). The result is the same: 17 of 20 seeds, no noise column. The Schur form matches a direct solve to 2e-10.

## The CMI gate itself kept noise aggregates (found on the way, fixed)

Without the linear check, the CMI redundancy gate kept an aggregate of a pure-noise column in 7 of 30 runs (12000 rows, informative group, 6 noise columns; 9 columns, all aggregates over the cells of the group). The permutation null of `CMI(cand; y | joint(group bins, agg bins))` shuffled `y` over all rows, which removes `y`'s dependence on the conditioning bins; the plug-in estimate under that null had mean 0.0008, against 0.0032 when `y` is shuffled within the strata of the conditioning bins (one noise aggregate: observed 0.00517, ceiling 0.00439 under the old null and 0.00576 under the correct one). Two changes:

- `y` is permuted within the strata of the conditioning bins (`_strata_permuted_labels`: a random permutation followed by a stable sort on the stratum code, linear time), once per (group, agg) pair and shared by its four statistics.
- The tail is a Gamma fitted to the null by moments (`_null_tail`), at the same family-wise one-sided level, because the null is positive and right-skewed and `mean + z * std` sits below its tail. After the first change alone 4 of 30 runs still kept a noise aggregate; after both, 0 of 30.

Pinned by `test_the_cmi_gate_alone_does_not_keep_noise_aggregates_of_an_informative_group_above_the_nominal_rate`. With the previous behaviour that test would fail (7 of 30, 90 percent lower bound 0.11 against 0.05).

## Not verified

- Cost at 1M rows, and after the final per-pair rewrite: only 200000 rows and the earlier per-candidate version were timed.
- The probe covers one scenario family. A layout where a column survives the CMI gate on its own was not found, so the cap's effect on a gated column is unmeasured.
