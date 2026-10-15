# Critique: single stricter global level vs. calibration/scheme changes

Source: read-only critique agent, 2026-10-14. Scope: `src/mlframe/feature_selection/filters/` and `tests/feature_selection/biz_val/`. Measured where stated; everything else is a hypothesis.

## Findings

### F01 - Single global level 0.999 is unstable and loses recall
- **Disposition**: OPEN
- Floor spread across seeds: 0.073-0.096 at q=0.999 (about +/-14%), against 0.058-0.065 at q=0.975 (measured, K=999).
- Recall on seed 7 (dense weak signal): 1 of 10 features kept at 0.999, against 4 of 10 at 0.975 (measured).
- One order statistic at 0.999 needs about 1000 draws to be stable; at K=999 it is almost the sample maximum.

### F02 - Dense weak-signal failure is power, not calibration
- **Disposition**: OPEN
- Empirical check (scripts `D:/Temp/mypyrepro/critic2/weak_calib.py`, `calib_rate.py`, not MRMR runs): the floor matches the family null. On fresh maxima for seed 1 the exceedance rate was 0.028 at q=0.975 (expected 0.025) and 0.015 at q=0.99 (expected 0.010), within noise (SE about 0.003-0.005).
- Per-feature permutation p-values of the strong features lie in 0.001-0.2. Their MI is 0.02-0.11.
- At n=330 with 16 target bins against 5 feature bins, the Miller-Madow term `(kx-1)(ky-1)/2n` is about 0.09, the same order as the floor (about 0.06). Weak features cannot clear it.

### F03 - Test checks a dead predicate
- **Disposition**: OPEN
- `TestDenseWeakSignalFloorStaysOff` (`tests/feature_selection/biz_val/test_biz_val_filters_screen_heavytail_floor.py:230`) exercises `target_oversplit_floor_applies` (`filters/_permutation_null.py:192`). Nothing in `src` calls it.
- The live gate is `filters/_screen_predictors_prescreen.py:130-134`, which turns the floor on for any narrow pool of two or more candidates. The end-to-end test is the only test that covers the live path.

### F04 - Live gate applies to every narrow pool
- **Disposition**: OPEN
- Turning the floor on for all narrow pools was a change made this session. It has not been benchmarked on the lognormal test or on signal recall beyond the three seeds checked.

### F05 - Uniform alpha across families is not principled
- **Disposition**: OPEN
- Screen, fallback and pair floors have different family sizes. One alpha does not give the same family-wise error across them. Set alpha per family and correct by the number of families (Bonferroni across families).

### F06 - Step-down maxT (Westfall-Young) as a power fix
- **Disposition**: OPEN
- Hypothesis, not run: compare each sorted observed MI with the null maximum over the remaining features. Should raise power on near-significant features without changing the level.

### F07 - Reliability predicate in the live gate
- **Disposition**: OPEN
- A rows-per-cell predicate at `filters/_permutation_null.py:226-228` (threshold 7.0) would have closed the dense weak-signal case without moving the level. Not measured; changes behaviour on narrow pools, so it needs the lognormal benchmark first.

### F08 - Unmeasured end-to-end checks
- **Disposition**: OPEN
- Not run: end-to-end MRMR on the weak-signal test (the 1-of-10 count was not re-verified); the pure-noise suite on the final scheme (15 of 18 was not re-verified on this scheme); stability at K >= 5000.

## Recommendation (from the critique)
Keep a fixed level (0.975 or 0.99) rather than moving to 0.999. Add step-down maxT for power, the reliability predicate after a lognormal benchmark, and per-family alpha with a family-count correction. Update the pair table and its docstring together.
