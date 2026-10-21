Written for: mlframe maintainers who run or extend the four-pillar bench.

# Four-pillar bench: review, and what was done with it

Two reviewers read `src/mlframe/feature_selection/_benchmarks/bench_four_pillars.py` and the pytest contracts it mirrors: `part1_f2_noise.md` (F2 and pure noise) and `part2_xor3_conjunction.md` (3-way XOR and conjunction). Their reports hold 30 findings, 20 + 16 proposed cases with generators, and harness proposals. This file records the disposition of each group. Nothing below has been calibrated against real MRMR runs yet: the gates in the registry are proposals, so every extended case starts as `report_only`.

## Findings that apply to the bench as it was

| Finding | Evidence | Disposition |
|---|---|---|
| F2 passes on name tokens only; a wrong-form column that mentions a, b, c, d passes | bench `:49-54,67` | `four_pillars_contracts.f2_verdict`: held-out Spearman of the selected columns against the noise-free signal, compared with the oracle, with exact column references (`refs`) for noise. Unit-tested on a right and a wrong column. The original pillar cases are unchanged |
| One seed per case | F2 seed 42, xor3 and conjunction seed 0 | `panel()` with one-sided Clopper-Pearson bounds; `--panel R` on the extended cases. A single lucky seed no longer counts |
| Noise pillar checks a flag, not the output | pure-noise test | `null_verdict`: the output must be empty |
| xor3 and conjunction fits pass no `random_state` / worker count | bench `:81` vs `:94`, `:125`, `:127` | extended cases always fit with `random_state=seed, n_workers=1`. The original pillar fits are untouched on purpose (their wall times are the tracker's baseline) |
| Conjunction checked in one column order only | bench `:127` vs the test `:116` | not done; the extended `conj_exact_partial_decoys` panel varies the seed instead |
| No null control of the same frame shape as the interaction cases | part 2 N1 | `null_control_parity_frame` |
| Contracts hard-code engineered names | test_xor3 `:48` | `cover_verdict` judges by operand coverage, with `normalize` for monotone twins |

## Cases implemented (`four_pillars_cases.CASES`, 37)

- Pillar 1 (F2): `f2_monotone_decoys`, `f2_leverage_tiny_b`, `f2_scale_and_range`, `f2_dupes_collinear`, `f2_discrete_mixed`, `f2_id_columns`.
- Pillar 2 (pure noise): factor blocks, wide p over n, tied targets, high-cardinality strings, NaN / duplicate / constant columns, heavy tails with ties, permuted F2, float32 offset, two report-only methodology cases (AR(1) and random walks, where an iid permutation null is expected to be anti-conservative), and the parity-frame null control.
- Pillar 3 (XOR): parity k=3 and 4, label noise 0.1 and 0.2, mixed operand types, unbalanced operands, NaN operands, a weak marginal on one operand, two triples (disjoint and sharing x1), gated parity, monotone twins.
- Pillar 4 (conjunction): exact and partial decoys, rare conjunction, AND-OR mixture, many-level categorical.

The generators for cases not in the registry (`f2_nan_mnar`, `f2_wide_small_n`, `f2_class_targets`, `f2_time_autocorr_walk`, `noise_tied_targets` kinds other than `zero_infl`, `g_regression_xor_conj`) are in the module but have no registry entry: each needs a contract the shared ones cannot express (complete-case Spearman, a chronological hold-out, an AUC bar), so they are deferred rather than given a wrong gate.

## How to run

```
python -m mlframe.feature_selection._benchmarks.bench_four_pillars --extra all --panel 20 --json out.json
python -m mlframe.feature_selection._benchmarks.bench_four_pillars --extra parity3_binary,null_control_parity_frame --panel 5
```

Each extended case prints `power k/R (CP95 lower)`, `forbidden k/R (CP95 upper)` and `PASS`, `INCONCLUSIVE` or `FAIL`. `--strict-extra` makes a gating case that does not PASS fail the run; none is gating yet.

## Not done, and why

- Calibration of the gates against real fits: the host has too little memory to run the 37 panels (hundreds of fits, several minutes each); the first full run decides which cases become gating.
- Subprocess isolation with a wall budget and peak-memory record per case, a double-fit determinism check, and a CPU/GPU selection diff: proposed in both reports, not implemented.
- A time-aware null for autocorrelated noise: a methodology change to the selector's permutation test, not a bench change.
- The reviewers' derived minimum row counts for k-way parity (160, 320, 640 rows practical floor for k=3, 4, 5) are derivations, not measurements.
