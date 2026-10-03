# tests/test_meta: failures, root causes, fixes (2026-10-03)

Start: 8 failed / 782 passed. Final full run (meta_run_final.log): 5 failed / 785 passed; one of those (annotations) was fixed
right after and re-verified alone, so 4 remain open (below). No baseline was refreshed wholesale; only stale/over-generous entries were removed or lowered.

## Fixed
| Test | Rule / root cause | Fix | Files |
|---|---|---|---|
| test_c901_ceiling_is_not_stale | C901 debt drained 194 -> 148, ratchet wants the ceiling lowered | C901_CEILING = 148 | tests/test_meta/test_c901_debt_ratchet.py |
| test_long_functions_do_not_grow (part) | new 260-line `_apply_min_features_fallback` | split into `_rank_raw_candidates`, `_accept_rescue_candidates`, `_commit_fallback_support` (verbatim bodies) | src/.../filters/_mrmr_fit_impl/_finalise.py (MRMR core; meta failure pointed here) |
| same (baseline shrink) | 4 entries now under 150 lines removed; 4 ceilings lowered (train_postcalibrators 193, _phase_train_val_test_split 239, select_target 174, select_best_scorer_per_column 172) | shrink only | tests/test_meta/_function_length_baseline.json |
| test_no_new_fail_open_handlers | two DEBUG-only substitutions after carve-outs | now `log_throttle(..., logging.WARNING, ...)`; stale baseline scopes (`#2`, `#4`) removed | filters/_mrmr_fe_step/_step_pairmi.py, feature_selection/pre_screen.py, _fail_open_handlers_baseline.json, _code_audit_baseline.json (stale pre_screen entry) |
| test_no_new_drifted_duplicate_functions | 17 PEP 562 `__getattr__` facades form a new group (each must close over its own globals); also the first test falsely failed on the interpreter-sensitive `__dir__` allow entry (gate false positive: 3.12 vs 3.14) | `__getattr__` added to KNOWN_DUPLICATE_GROUPS with reason and INTERPRETER_SENSITIVE; test 1 now drops unreported sensitive names from `allow` | tests/test_meta/test_no_drifted_duplicate_functions.py |
| test_no_mlframe_file_exceeds_1k_loc (part) | `_conditional_gate_fe.py` 1005, `_hermite_fe_optimise.py` 1006 | carved `_conditional_gate_naming.py` (names, recipe builders, constants) and `_hermite_fe_diverse.py` (`_select_diverse_topm`), re-exported | filters/_conditional_gate_fe.py, _conditional_gate_naming.py, _hermite_fe_optimise.py, _hermite_fe_diverse.py |
| test_no_new_code_audit_findings | two copy-pasted scorer dispatchers; two wrapper variants with inlined body | shared `_dispatch_scorer` in `_orth_auto_scorer_fe.py` used by both; `_permnull_variant` factory | filters/_orth_auto_scorer_fe.py, _orthogonal_scorer_auto_fe.py, _permutation_null_resident_ktc.py |
| test_no_new_test_quality_findings_in_the_test_suite | tautological `is not None` test; 4 stale baseline entries | test now asserts panel type, series labels, categories, rates; stale entries removed | tests/reporting/test_charts_error_analysis.py, _code_audit_tests_baseline.json |
| test_no_new_unannotated_public_functions | my carved recipe builders lacked return types | `-> EngineeredRecipe` (TYPE_CHECKING import) | filters/_conditional_gate_naming.py |

Not run through black: these files are not black-formatted (project says use py_ci_shared.black_filtered_apply, never raw black). mypy and ruff F-rules clean on all touched files.

## Open (not fixed)
1. test_the_installed_pyutilz_includes_the_pin: sibling pyutilz checkout is clean but 15 commits behind origin/master (pin 635d0c5a34c3). Needs `git pull --ff-only` in C:\Users\Admin\Machine learning\pyutilz. Not done: it changes scanners for every concurrent session.
2. _pairs_score.py (owned by the pairs agent, uncommitted +67 lines): `_score_one_pair` 1255 lines vs ceiling 1213, and file 1355 LOC vs 1283+50 slack. Needs the new branch moved into a helper/sibling by its owner. Baselines deliberately not raised.
3. (2 and the LOC test are the same file.)
4. test_no_new_nondiscriminating_assert: tests/feature_selection/shap_proxied/test_biz_val_shap_proxied_faith_interaction.py::test_biz_val_faith_interaction_beats_additive_on_xor has a `late-skip` (pytest.skip after computing add_recall). File was edited by another session at 16:13. Its owner should assert the premise or pick a seed where it holds.
