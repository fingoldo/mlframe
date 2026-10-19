# Tracker: 2026-10-14 agent critiques (family-wise gates for MRMR)

Each finding has one disposition line in its own file. Statuses: OPEN, COMPLETED, REJECTED (with reason).

| File | Findings | Status |
|---|---|---|
| `pair_family_threshold_critique.md` | F01-F10 | all OPEN |
| `level_strictness_critique.md` | F01-F08 | all OPEN |

## Open questions carried over
- Choice between an exact per-pair threshold and a cascade (table pre-filter, exact test on survivors). Recommended: cascade plus a family over all pairs x configurations.
- Whether the null-MI table is rebuilt at quantile 0.99 with candidate grid up to 5000. A rebuild was running on 2026-10-14 (`D:/Temp/mypyrepro/null_mi_table_q99_wide.json`); its output is not yet checked into the repo.
- The checked-in pair table currently holds quantile 0.999 (rebuilt locally); its docstring still says 95th percentile (F03 in the pair file).

## Measurements that back these findings (not MRMR end-to-end runs)
- K=1734, n=2000: direct 99% family null 0.01046 against the table value 0.0099 (K=1000 neighbour).
- Dense weak-signal floor at seed 1: exceedance 0.028 at q=0.975 (expected 0.025), 0.015 at q=0.99 (expected 0.010).

## Next
1. Implement the pair cascade (F01-F02, F05-F06 of the pair file) and re-run the pure-noise suite.
2. Decide on F01-F03 of the level file after the cascade is measured.

## Added during implementation (2026-10-14)
- **F11 (pair family scope, OPEN)**: the family gate must cover all configurations of one pair. Batch-level gating was too narrow (a 1-batch family has a lower threshold). Measured: on student_t2/binary the surviving pair `max(rint(c0),abs(c1))` (MI 0.0107) was never gated: the pair's 1734 configurations come from the chunk fast path (`_pairs_score_steps.py:210`, `_score_one_pair_step3_step1_pair_phase_produced` returns MI only), where the discretised codes are not exposed.
- **F12 (codes plumbing, OPEN)**: expose the step-1 discretised matrix (`_disc_2d`, `_pairs_score_steps.py:317-363`) or the chunk codes (`_pairs_chunks.py`) so the family gate can read per-configuration codes on the chunk path. Until then the gate covers only the non-chunk per-candidate path (`_pairs_score_steps.py:444`) and the batch extval path (`_pairs_emit.py`).
- **Status**: noise suite 15/18 with the pair gate on the non-chunk paths; no change for the student_t2 case.

## Status by finding

Generated from each report's `**Disposition**` line: OPEN is recorded as TODO and COMPLETED as RESOLVED, the vocabulary the countable-status check reads.

| Status | Finding | Report | Title |
|---|---|---|---|
| **TODO** | F01 | `level_strictness_critique.md` | Single global level 0.999 is unstable and loses recall |
| **TODO** | F02 | `level_strictness_critique.md` | Dense weak-signal failure is power, not calibration |
| **RESOLVED** | F03 | `level_strictness_critique.md` | Test checks a dead predicate |
| **TODO** | F04 | `level_strictness_critique.md` | Live gate applies to every narrow pool |
| **TODO** | F05 | `level_strictness_critique.md` | Uniform alpha across families is not principled |
| **TODO** | F06 | `level_strictness_critique.md` | Step-down maxT (Westfall-Young) as a power fix |
| **TODO** | F07 | `level_strictness_critique.md` | Reliability predicate in the live gate |
| **TODO** | F08 | `level_strictness_critique.md` | Unmeasured end-to-end checks |
| **TODO** | F01 | `pair_family_threshold_critique.md` | Per-pair exact threshold does not control FWER over all pairs |
| **TODO** | F02 | `pair_family_threshold_critique.md` | Cost of the exact test per pair |
| **TODO** | F03 | `pair_family_threshold_critique.md` | Table docstring and table metadata disagree on the quantile |
| **TODO** | F04 | `pair_family_threshold_critique.md` | Nearest-neighbour lookup underestimates the threshold for K above the grid |
| **TODO** | F05 | `pair_family_threshold_critique.md` | Discretisation of the gate and of the pair search differ |
| **TODO** | F06 | `pair_family_threshold_critique.md` | GPU residency |
| **TODO** | F07 | `pair_family_threshold_critique.md` | Sharing permutations across pairs |
| **TODO** | F08 | `pair_family_threshold_critique.md` | Determinism |
| **TODO** | F09 | `pair_family_threshold_critique.md` | Analytic bound as a cheap pre-filter |
| **TODO** | F10 | `pair_family_threshold_critique.md` | Sequential stopping rule for borderline pairs |
