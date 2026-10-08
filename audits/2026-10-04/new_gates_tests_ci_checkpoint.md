# Checkpoint: new gates wiring (tests/CI), paused

## Files changed so far
- 27 test files under tests/: `from tests._known_gap import known_gap` replaced by `from py_ci_shared.pytest_known_gap import known_gap` (ruff clean).
- tests/_known_gap.py deleted (no importer left).
- 9 files under tests/feature_selection (biz_val imbalanced_rare_class, multicollinear_pollution, weak_family_adversarial; fe/provenance target_aware_leak_contract;
  mrmr/biz_val test_anchor_refinement, underselection; mrmr/fe sample_weight_fe; stability selector_contract_protocol_extra, selector_contract_shared):
  17 xfail reasons prefixed `(KG-n)` (the 17 untracked-reason hits). KG ids: 1 RFECV imbalance, 2 multicollinear x3, 3 weak family, 4 target-aware leak,
  5 anchor refinement, 6 underselection, 7 sample_weight FE, 8 ndarray/DataFrame-only/set_output parity, 9 transform width + duplicate-column guard, 10 column-order dependence.
- tests/test_meta/test_shared_gates_adopted.py: xfail gate call now passes `known_gap_modules=()`.

## Done
- Item 2 wiring + reasons. Verification run in background: log `%TEMP%/xf2.log` (grep `untracked|^FAILED|passed`); result not yet read.
- Item 1 hits enumerated: 48 (46 single-shot, 2 tight races; script scratchpad/find_ss.py). Markers `hang_guard` and `perf` already registered in pyproject.

## Remaining
1. Add a KG-1..KG-10 register (exact gap per id) to audits/2026-10-04/new_gates_tests_ci.md (reasons point at it).
2. Item 1: review the 48 hits; hang ceilings (`time.time()-t0 < 300`, conditional_gate x4, mrmr 300s/180s fits, wide_data_scalability, etc.) get `@pytest.mark.hang_guard`;
   ratio tests become best-of-N (perf_speedup_floor/perf_time_budget in tests/conftest.py); 2 tight races get `perf` or wider slack. Then replace
   tests/test_meta/test_no_single_shot_timing_assertion.py + _single_shot_timing_baseline.json with the shared gate (check test_baseline_registry_consistency / BASELINES_README refs).
3. Item 3: ci_default_branch_never_cancelled over .github/workflows (7 hits; sklearn-matrix-ci and mypy-full get `cancel-in-progress: ${{ github.event_name == 'pull_request' }}`,
   utility workflows `# cancel-ok: <reason>`), validate with actionlint/zizmor/yamllint.
4. Item 4: ci_install_covers_entry_imports + assert_entry_imports_without_extras replacing the hand-written subprocess test in test_workflow_supply_chain.py.
5. Item 5: optional_imports_guarded with reachable_from=["mlframe"] replacing test_optional_third_party_imports_are_guarded.py.
6. black_filtered_apply on edited files; run touched tests, the known_gap importers, and the final `-k "xfail or timing or optional or workflow or adopted or nondiscriminating or source_text or floorless"` run; write the report.

## Next exact step
Read `%TEMP%/xf2.log`, then start item 1 with the hang_guard markers (first file: tests/feature_selection/biz_val/test_biz_val_filters_conditional_gate.py lines 533-589).

## Resumed session update
- tests/_known_gap.py deletion is intended (no importer left). The tree flapped mid-session (another session's pre-commit stash/restore reverted edits for minutes); re-verify edits persist before commit.
- KG register written: audits/2026-10-04/new_gates_tests_ci.md.
- Item 1 done: 48 hits drained (hang_guard / perf markers, best-of-N); tests/test_meta/test_no_single_shot_timing_assertion.py now calls the shared gate (no baseline, json removed, README + fail_closed list entries removed).
- Item 3 done: 5 workflows cancel on pull_request only, black-filtered and docs carry `# cancel-ok`; actionlint and zizmor clean.
- Item 4/5 wired in test_workflow_supply_chain.py and test_optional_third_party_imports_are_guarded.py; verification run log %TEMP%/m2.log.
- Remaining: read m2.log, targeted runs of touched tests, commit with explicit paths, write report audits/2026-10-04/new_gates_tests_ci_report.md.

- Committing all of the above by explicit paths; remaining: re-run the dynamic core-only import test on a quiet tree.

## Commit status
Not committed: six commit attempts lost to other sessions' index.lock and mixed-line-ending autofix rounds. Path list: %TEMP%/ngtc_paths.txt (80 paths), message: %TEMP%/msg_ngtc.txt.
Retry: git commit -o -F <msg> -- <paths>, verify HEAD moved and the subject. Then re-run test_core_only_install_can_import_the_public_subpackages on a quiet tree.
