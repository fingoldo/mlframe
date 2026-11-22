# MRMR module audit (2026-11-23)

Read-only audit of the MRMR modules by Haiku agents, one file per audit. Code reading only; nothing executed. Each finding is confirmed (traced in the code path) or a hypothesis; the disposition for each lives in `_TRACKER.md` (rows AUD23-01 to AUD23-50).

| File | Area | Wave | Tracker rows |
|---|---|---|---|
| 01_selection_core.md | greedy selection, support assignment, index mapping | 1 | AUD23-37, 03 |
| 02_discretization.md | edge builders, code dtypes, NaN | 1 | AUD23-12, 13 |
| 03_gpu_paths.md | device memory, circuit breakers, copies, dtypes | 2 | AUD23-02, 18-21 |
| 04_state_caches.md | fitted state, pickling, caches | 2 | AUD23-01, 03, 49, 50 |
| 05_api_tests.md | constructor, docs, fs_params, weak tests | 2 | AUD23-28 to 36 |
| 06_fe_stages.md | FE stages, recipes, dedup, gates | 2 | AUD23-38 to 43, 46 |
| 07_concurrency.md | concurrent FE stages, merge, pools | 2 | AUD23-04, 05, 06, 47, 48 |
| 08_it_estimators.md | MI, entropy, Chao-Shen, Miller-Madow | 2 | AUD23-22 to 27 |
| 09_transform.md | transform and recipe replay | 3 | AUD23-07 to 11 |
| 10_categorical.md | categorical encoding, target_stat, NaN | 3 | AUD23-07, 08, 44, 45, 46 |
| 11_significance.md | maxT / FDR gates, nulls, p-values | 3 | AUD23-14 to 17 |

Coverage gaps the auditors reported: greedy core `_screen_predictors*.py` and `_numba_utils*.py` (wave 1); `supervised_binning.py`, `_mdlp_validated_split.py` (wave 1); GPU-only kernel bodies (wave 2); the all-NaN fit paths of `_missingness_fe` and `_grouped_*` (wave 3). Those are the next places to read.

Process: each row needs a reproducing run or test before it is called a defect; hypotheses stay hypotheses until then. Fixes follow the tracker rule (status moves to RESOLVED only with a test that fails before and passes after).
