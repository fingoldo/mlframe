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
| 4 | Pure noise and a linear target that the raw columns explain keep nothing under the fix | `tests/feature_selection/fe/test_binagg_linear_usability.py` (noise: 3 seeds, 10 columns; linear target) | Pinned by tests |

## Why the CMI gate rejects the column

The gate scores `CMI(col; y | qbin(g), qbin(a))`. The column is a function of the cell of `g`, the same partition the gate conditions on, so it is redundant for any model that can use the raw group. That is the right answer for trees and the wrong one for a linear model.

## The fix

For a column the CMI gate rejects, fit class indicators of `y` linearly on the raw sources with and without the column, out of fold, and keep the column when the gain clears a null (`mean + z * std`, `z` the family-wise z the CMI gate uses). No fixed acceptance threshold. Rows are subsampled to 100000.

The null is the candidate with its per-cell values permuted among the cells of the group, not the candidate shuffled across rows. A first version shuffled across rows and kept 130 aggregates of pure-noise columns on a 30-column frame with an informative group (a random value per cell correlates with a target that varies by cell). With the cell-permuted null the same frame keeps one column, `binagg_std(aux|qbin(g))`, and no noise aggregate. A candidate that does not beat the largest of the first 4 null gains is dropped without running the other 28.

## Cost

200000 rows, 30 numeric columns, informative group, 4 alternating runs: 5.3 to 5.9 s without the check, 13.7 to 13.8 s with it. The base model's per-fold statistics are computed once per call, so each null draw costs only cross sums.

## Not verified

- Cost at 1M rows: only 200000 was timed.
- The probe covers one scenario family. A layout where a column survives the CMI gate on its own was not found, so the cap's effect on a gated column is unmeasured.
