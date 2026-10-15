# Critique: exact per-pair family threshold for MRMR pair FE

Source: read-only critique agent, 2026-10-14. Scope: `src/mlframe/feature_selection/filters/`. Measured where stated; everything else is an estimate.

## Findings

### F01 - Per-pair exact threshold does not control FWER over all pairs
- **Disposition**: OPEN
- Each pair is gated independently. At alpha 0.01 over 28 pairs (8 columns) the chance of at least one false pass is about 0.25.
- Fix direction: a family of all pairs x configurations for the final FWER, not only per-pair.
- Evidence: `_pairs_emit.py:170` (K = `len(var_pairs_perf)`).

### F02 - Cost of the exact test per pair
- **Disposition**: OPEN
- Measured: `family_maxt_threshold`, n=2000, K=1734, 20 permutations: 0.70 s on a loaded machine (`_family_maxt_gate.py:60-105`).
- Estimated (not measured): 999 permutations per pair at n=2000 is about 35 s; 28 pairs is about 16 minutes. n=20000 is about 350 s per pair by linear extrapolation.

### F03 - Table docstring and table metadata disagree on the quantile
- **Disposition**: OPEN
- `_null_mi_threshold.py:1-5` says 95th percentile. The builder default is `QUANTILE = 0.95` (`_benchmarks/build_null_mi_table.py:34`). The checked-in table at the time of the critique had meta quantile 0.999, which came from a local rebuild and was not documented in the builder.
- Fix: one source of truth for the quantile, written into the table meta and the docstring.

### F04 - Nearest-neighbour lookup underestimates the threshold for K above the grid
- **Disposition**: OPEN
- `_null_mi_threshold.py:37-52` picks the nearest entry on log K. For K=1734 the entry is K=1000, which gives 0.0099 against a directly measured 99% family null of 0.01046 (`family_maxt_threshold`, 199 permutations, n=2000, same discretiser).
- Fix: extend the candidate grid to 2000-5000, or interpolate.

### F05 - Discretisation of the gate and of the pair search differ
- **Disposition**: OPEN
- `_family_maxt_gate.py:53-57` uses `np.quantile` with `np.unique` edges. The pair search uses `discretize_2d_quantile_batch` (`discretization/__init__.py:634`) with `np.percentile`.
- Comparing `best_mi` with a threshold computed on different codes is not a valid test. The gate must use the codes the pair search produces.

### F06 - GPU residency
- **Disposition**: OPEN
- `family_maxt_threshold` is numpy/numba only. Calling it from the device path forces a host copy of K x n codes: 3.5 MB at n=2000, 35 MB at n=20000.
- A cupy variant is needed, modelled on the table builder (`build_null_mi_table.py:44-70`). The device pair path is at `_pairs_emit.py:33-67` and `:748`; it must not be touched by the host variant.

### F07 - Sharing permutations across pairs
- **Disposition**: OPEN
- Sharing the y permutations between pairs of the same n saves only the generation of permutations. The cost is in K x n histograms, so the saving is small.

### F08 - Determinism
- **Disposition**: OPEN
- `default_rng(seed)` is reproducible for a fixed seed. Derive the seed from the pair index and n, so the result does not depend on processing order.

### F09 - Analytic bound as a cheap pre-filter
- **Disposition**: OPEN
- A Bonferroni chi-square bound (`2n*MI >= chi2_{1-alpha/K}((bx-1)(by-1))`) is O(K) but asymptotic and conservative. Must be checked against the permutation threshold on several realisations before use.

### F10 - Sequential stopping rule for borderline pairs
- **Disposition**: OPEN
- Accept early when the observed maximum exceeds all of the first 99 null maxima (p about 0.01). Reject early when it is below the median null after about 20 permutations. Only borderline pairs run the full count. Needs an explicit threshold rule, not a constant.

## Recommendation (from the critique)
Cascade: table floor as a cheap pre-filter, exact test only on survivors, sequential stopping for borderline cases, and a family over all pairs x configurations for the final FWER. Cupy variant for the device path.
