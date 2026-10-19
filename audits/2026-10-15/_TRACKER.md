# Tracker: 2026-10-15 critiques of the in-fit family-wise gate

Each finding has one disposition line in its own file. Statuses: OPEN, COMPLETED, REJECTED (with reason).

| File | Findings | Status |
|---|---|---|
| `family_gate_statistics_critique.md` | F01 COMPLETED; F02-F08 OPEN | 1 of 8 completed (no change in this session) |
| `family_gate_engineering_critique.md` | COMPLETED: F01, F04, F09, F11, F13, F15, F18, F19, F20, F21; REJECTED: F08; OPEN: F02, F03, F05, F06, F07, F10, F12, F14, F16, F17 | 10 completed, 1 rejected, 10 open |

## Done in code (2026-10-15)
- Exact permutation p-value replaces the quantile rule in the family gate and in the empty-support fallback (`_family_maxt_gate.py`: `exact_pvalue_pass`; `_fallback_maxt.py`; `_mrmr_fit_impl/_finalise.py`).
- Block kernel over permutations, verified equal to the per-permutation loop (maximum absolute difference 0.0).

## Open questions for the next step
- F01 (engineering): the noise fit that selected `f24` must be reproduced and traced before any default is claimed. Until then "nothing on noise in every fit" is not established.
- Sequential stopping (statistics F03, engineering F05) is the main cost lever and is not implemented yet.
- The strict level must be a constructor parameter in every fit (engineering F12); the raw screen and fallback need the same parameter, not only the pair path.

## Verification
- Pure-noise suite, null-FWER, parity and periodic tests: run after the exact-rule change (results recorded below when available).

## Verification result (2026-10-15, after the exact p-value change)
- Pure-noise suite, null-FWER, parity, periodic identity: 45 passed, 3 failed. The failures are the same three noise cases as before (binary/uniform, continuous/uniform, binary/student_t2). The exact rule did not change the noise outcome, so the cause is elsewhere (see F01 in the engineering file).

## Root cause of the f24 noise selection (2026-10-15)
- Traced: the first raw confirmation step accepts a noise feature because the screen floor is at quantile 0.975 (family-wise 2.5%). Recorded as engineering F14 (OPEN). The fix is the strict-level decision applied to the screen, which needs sequential stopping or an analytic tail (statistics F03, engineering F05, F06).

## Strict-screen experiment (2026-10-15)
- F15 (OPEN, decision needed): strict screen level improves noise (16/18 at 1e-4 and 1e-3 versus 15/18 at 0.975) but breaks the dense weak-signal test on seed 1. Two directions: strict default with a reframed weak test, or a documented compromise level. Measurements are in the engineering file.

## Decision (owner, 2026-10-15): option B
- Screen default restored to quantile 0.975 with 999 permutations (family-wise about 2.5% per fit). The strict level stays a parameter.
- The pure-noise contract changes from "flag set on every seed" to a rate contract: the fraction of fits without the flag over many seeds must not exceed the level plus Monte Carlo error. Support is non-empty by policy (`best_feature`), so emptiness is not the criterion; the flag is.
- Next: rewrite the pure-noise test as a rate test over seeds (F16), and record the change of contract in the audit.

## MDLP track (2026-10-15)
- F18 (OPEN): root cause: MDLP fitted on y, null not refitted.
- F19 (OPEN): cost and options, decision A accepted by owner.
- F20 (COMPLETED module, OPEN integration): honest refit null implemented and measured; integration into the screen floor for supervised strategies is next, using the production call for the observed value too.
- Test: `test_pure_noise_selects_nothing` now parametrized over `nbins_strategy` in (None, quantile, uniform, mdlp, mdlp_validated): 90 cases; mdlp cases expected to fail until integration is done.

## Session 2026-10-15 (continued): verified changes and dispositions

Verified in this session (evidence recorded in the code and the transcript, numbers as measured):
- Honest supervised null integrated for `mdlp` and `mdlp_validated` (held-out-validated cut logic in `_mdlp_oos_honest_njit.py`). Cut sets equal the production recursion on 120 identical splits (0 mismatches). The holdout split uses numba's generator, so the null is compared to the Python reference in distribution (quantiles 0.5/0.9/0.975 within about 2%), not bit for bit.
- Pure-noise suite, all binning strategies (None, quantile, uniform, mdlp, mdlp_validated), 90 cases: 90 passed after the oversplit gate was removed.
- The target over-split gate (`target_oversplit_floor_applies`) was removed together with its tests. Evidence: with the gate restored, five pure-noise cases failed; with the floor open on narrow pools, all five passed. The gate's only measured effect was switching the floor off on binary noise. No test showed a power loss from the open floor.
- Dead global `get/set_no_significant_policy` removed; the per-instance parameter path is unchanged.
- Meta test `test_no_new_function_without_a_caller` passes. Reference kernels kept by the keep-all-kernels rule are listed in `_RETAINED_KERNELS` with their parity tests.

Still open and not verified in this session:
- F14 / F16 / F17: the raw screen level 0.975 admits noise on the first step by design; the binary-target rate contract (F16) is still the flag-on-every-seed test. Not rewritten yet.
- F02-F07, F10, F12: not touched.
- Heavy-tail floor file: 9 tests after the decorator fix; the run is in progress and its result is recorded below when available.
- STRICT GPU mode (`MLFRAME_FE_GPU_STRICT=1`): not run.

## Status by finding

Generated from each report's `**Disposition**` line: OPEN is recorded as TODO and COMPLETED as RESOLVED, the vocabulary the countable-status check reads.

| Status | Finding | Report | Title |
|---|---|---|---|
| **RESOLVED** | F01 | `family_gate_engineering_critique.md` | Noise fit selected a feature in the profiling run |
| **TODO** | F02 | `family_gate_engineering_critique.md` | Family gate is a small share of fit time on that run |
| **TODO** | F03 | `family_gate_engineering_critique.md` | Per-permutation kernel cost |
| **RESOLVED** | F04 | `family_gate_engineering_critique.md` | Resolution of the quantile threshold at 999 permutations |
| **TODO** | F05 | `family_gate_engineering_critique.md` | Sequential stopping cost estimate |
| **TODO** | F06 | `family_gate_engineering_critique.md` | Reaching alpha = 1e-4 needs 1e4 permutations or an extreme-value approximation |
| **TODO** | F07 | `family_gate_engineering_critique.md` | Narrow the family to configurations above the table floor |
| **REJECTED** | F08 | `family_gate_engineering_critique.md` | Shared permutations across pairs of the same n |
| **RESOLVED** | F09 | `family_gate_engineering_critique.md` | Code reuse |
| **TODO** | F10 | `family_gate_engineering_critique.md` | cupy path for the family kernel |
| **RESOLVED** | F11 | `family_gate_engineering_critique.md` | Bit-identity across thread counts |
| **TODO** | F12 | `family_gate_engineering_critique.md` | Strict default hook |
| **RESOLVED** | F13 | `family_gate_engineering_critique.md` | Claims in the brief that did not match the repository |
| **TODO** | F14 | `family_gate_engineering_critique.md` | Raw screen floor at the 0.975 level admits noise (root cause of F01) |
| **RESOLVED** | F15 | `family_gate_engineering_critique.md` | Strict screen level trades off against the dense weak-signal test |
| **TODO** | F16 | `family_gate_engineering_critique.md` | False-selection rate is far above the designed level for a binary target |
| **TODO** | F17 | `family_gate_engineering_critique.md` | Screen gain and its null floor are on different statistics (binary target) |
| **RESOLVED** | F18 | `family_gate_engineering_critique.md` | Supervised MDLP binning is fitted on y, and the permutation null never refits it (root cause of F16/F17) |
| **RESOLVED** | F19 | `family_gate_engineering_critique.md` | Honest MDLP null and its cost (follow-up of F18) |
| **RESOLVED** | F20 | `family_gate_engineering_critique.md` | Honest MDLP null implemented (decision A: accepted cost) |
| **RESOLVED** | F21 | `family_gate_engineering_critique.md` | Honest supervised null integrated (screen floor and fallback) |
| **RESOLVED** | F01 | `family_gate_statistics_critique.md` | Quantile threshold does not give the stated level |
| **TODO** | F02 | `family_gate_statistics_critique.md` | 1e-4 is not reachable with 999 permutations |
| **TODO** | F03 | `family_gate_statistics_critique.md` | Sequential stopping (Besag-Clifford style) |
| **TODO** | F04 | `family_gate_statistics_critique.md` | Early rejection changes the level only if its premise fails |
| **TODO** | F05 | `family_gate_statistics_critique.md` | Table floor is a nearest-neighbour lookup, not a test |
| **TODO** | F06 | `family_gate_statistics_critique.md` | Possible leakage through configuration construction |
| **TODO** | F07 | `family_gate_statistics_critique.md` | Overall level is a union over gates |
| **TODO** | F08 | `family_gate_statistics_critique.md` | Regression test for the level |
