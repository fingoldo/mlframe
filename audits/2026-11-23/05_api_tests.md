# Audit 05: MRMR public API and tests (wave 2)

Scope: constructor signature and docstring, `_mrmr_param_constants.py`, `_mrmr_validate_strings.py`, generated `training/fs_params/mrmr.py`, and the tests that pin them. Read-only, Haiku model, nothing executed.
Deviation disclosed by the auditor: its first shell call wrote scratch files `/tmp/ctor.txt` and `/tmp/fs.txt`, breaking the read-only rule. They were deleted; neither exists now.

Tracker rows: AUD23-28 to AUD23-36.

1. Docstring says cv defaults to None; code says 3. `mrmr/_mrmr_class.py:208` vs `:1198` and `fs_params/mrmr.py:185`. Confirmed. Medium (documentation).
2. fe_fast_search default stated two ways. `_mrmr_class.py:552` says "default ON"; `:576` says "DEFAULT FALSE"; constructor `:580` is `False`. Code is consistent, the comment at 552 is wrong. Confirmed. Medium.
3. Accepted redundancy values not implemented. `_mrmr_param_constants.py:20` allows `pld_max` and `pld_mean`; `fs_params/mrmr.py:80` exposes them; `_mrmr_validate_strings.py:19-22` and `:127-135` say they run the Fleuret path and only log a warning. The warning is confirmed; the redundancy path was not traced. Low to medium.
4. categorical_fold_strategy not enforced by the generated model. `fs_params/mrmr.py:49` types it as plain `str`; live allow-list at `_mrmr_validate_strings.py:150` and `_mrmr_param_constants.py:18`. A config accepts `"auto"` that the constructor rejects only at fit. Confirmed. Low to medium.
5. Unvalidated string parameters: `fe_mi_estimator` (read at `_step_core_parts.py:678`) and `fe_acceptance` (`:146`) are not in `_validate_string_params`. Unknown values fall through. Hypothesis, not traced. Low.
6. Drift test ignores annotations. `tests/training/test_fs_params_in_sync.py:27` passes `check_annotations=False`; `:36-50` compares a fresh render with the committed module, not the live signature, so it cannot catch a retyped live parameter. Confirmed. Medium.
7. Name drift is clean: 553 names in both, no missing or extra. Defaults match except number formatting (`1e-6` vs `1e-06`), `dtype` / `quantization_dtype` (skipped by design), and `UNIFIED_FE_SUBSAMPLE_N` (`30_000` at `_feature_engineering_mem_budget.py:278`). Confirmed.
8. Weak test: fold-strategy test checks an argument, not behaviour. `tests/feature_selection/mrmr/test_categorical_fold_param.py:32-54` spies on `categorize_dataset` and asserts the support value it receives; it would pass if the fold were a no-op. Confirmed. Medium.
9. Weak test: unmatched exception check. Same file `:28-29`: `pytest.raises(ValueError)` without `match=`. Confirmed. Low.
10. Weak test: documentation token check. `tests/test_meta/test_default_scorer_choice_is_deliberate.py:42-43` checks the docstring contains `"bench-attempt-rejected"`, not behaviour. The default and the recommendation currently differ, so the branch runs. Confirmed. Low.
