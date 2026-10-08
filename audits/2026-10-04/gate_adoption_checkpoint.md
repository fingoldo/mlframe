# gate_adoption checkpoint

State (resumed 2026-10-08): nothing committed yet. Scratch drivers: scratchpad/drv2.py (candidate gates with scoping), drv3.py (unresolved attrs, ~16 min, writes scratchpad/f_unres.txt), drv4.py (second batch).

Edited so far (uncommitted, all mine):
- tests/test_meta/test_shared_gates_adopted_pass2.py (new): wires ~25 shared gates + population canary tests.
- 12 gate files in tests/test_meta gained `_candidate_files()` (population canary): fe_family_noop_no_copy, logger_lazy_formatting, module_cache_mutated_without_its_lock,
  no_audit_metadata_in_comments, no_fe_family_enabled_without_budget, no_module_level_logging_disable, no_nondiscriminating_assert, no_numba_config_env_restore_footgun,
  no_stale_not_wired_docstrings, no_tick_isinstance_offset_check, no_unlocked_module_cache, no_unprotected_shap_treeexplainer, readonly_to_numpy_mutation.
- src fixes: calibration/post.py (path in docstring), training/crash_diagnostics.py (threading.__excepthook__ guard for 3.9), 3 `# vendored-ok` markers on joblib.externals.loky imports,
  stdlib json -> orjson in ~14 src files, core/proportion_stats.norm_ppf shared with reporting/charts/slice_finder, NS5_COEFFS shared in neural/_muon_*.
- .github/workflows/*.yml: runner labels pinned via `python -m py_ci_shared.workflow_runner_labels --fix .` (14 files).

Still to do:
- Wait for drv3 (unresolved module attributes, 103 findings pre-fix): fix stale readers, baseline the rest in tests/test_meta/_unresolved_module_attributes_baseline.json.
- Create baselines _committed_line_endings_baseline.json (115 legacy CRLF blobs) and _drifted_duplicate_literals_baseline.json (RULE_SET only, hand-written reasons);
  register both in BASELINES_README.md and test_baseline_registry_consistency.FLAG_EXEMPT (BASELINES_README.md is modified by another session: append only).
- Run pass2 tests, ruff, mypy; commit with explicit paths.
- Write audits/2026-10-04/gate_adoption_report.md with a disposition per gate.

Dispositions decided so far: commit_metadata not adoptable (history carries Co-Authored-By trailers that cannot be rewritten); tracked_secret_shapes N/A (detect-secrets in pre-commit,
no vendor shape); unsafe_deserialization only fires in _benchmarks; hardcoded_seed/cuda_width/unread_params/single_shot/bare-pickle covered by local equivalents.
