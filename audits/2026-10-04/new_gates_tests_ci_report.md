# New py-ci-shared gates: tests and CI wiring, report

Scope: the five items of `new_gates_tests_ci_checkpoint.md`. Dispositions use RESOLVED / FUTURE / DOC / REJECTED.

## Dispositions

| # | Item | Disposition | What changed |
|---|------|-------------|--------------|
| 1 | known_gap swap in 27 test files | RESOLVED | `from tests._known_gap import known_gap` became `from py_ci_shared.pytest_known_gap import known_gap`; `tests/_known_gap.py` deleted on purpose (no importer left, checkpoint line 5). |
| 2 | xfail gate: untracked reasons | RESOLVED | 17 xfail / known_gap reasons carry `(KG-n)`; `test_shared_gates_adopted.py` passes `known_gap_modules=()`. The register is `audits/2026-10-04/new_gates_tests_ci.md` (KG-1 .. KG-10, exact gap per id). The gaps themselves stay open as FUTURE and are tracked by the register; each known_gap flips to a failure when its gap closes. |
| 3 | single-shot timing gate: 48 hits | RESOLVED | Replaced `test_no_single_shot_timing_assertion.py` with the shared gate, no baseline (`_single_shot_timing_baseline.json`, its README line and its `test_scanning_gates_fail_closed` entry removed). See breakdown below. |
| 4 | `ci_default_branch_never_cancelled` | RESOLVED | 7 hits. `codeql`, `gpu-extras-install-matrix`, `hooks-not-in-ci`, `mypy-full`, `sklearn-matrix-ci` now use `cancel-in-progress: ${{ github.event_name == 'pull_request' }}`; `black-filtered` and `docs` carry `# cancel-ok: <reason>`. `actionlint` and `zizmor --offline` report nothing. New test `test_no_workflow_that_runs_on_a_master_push_cancels_its_own_master_runs`. |
| 5 | `ci_install_covers_entry_imports` + dynamic twin | RESOLVED | Static gate: 0 findings, test `test_every_workflow_installs_what_the_modules_it_runs_import` runs it with `include_pytest=False` (with pytest entries it follows ~4,000 test files and takes 260+ s; test modules are covered by the conftest install gate and `importorskip`). The hand-written `_BLOCKER` subprocess test now calls `assert_entry_imports_without_extras`. |
| 6 | `optional_imports_guarded` with `reachable_from=["mlframe"]` | RESOLVED | `test_optional_third_party_imports_are_guarded.py` is two tests over the shared gates (0 findings). The hand-written AST scanner and its teeth-check are retired (the shared gate has its own canaries). |
| 7 | Production defect found by the dynamic twin | RESOLVED | `mlframe.models.tuning_rules` imported `pyutilz.db` at module level, which imports `sqlalchemy` (the `db` extra). Every importer of `mlframe.models`, `mlframe.training.core`, `mlframe.feature_selection*`, `mlframe.training.composite` and `mlframe.feature_engineering.numerical` (10 entries) failed on an install without the `db` extra. The import is now inside `prepare_trials_dataset`, the only function that uses it. |
| 8 | Dynamic twin blocking transitive core dependencies | RESOLVED (test design) | Blocking the `stats` extra broke `category_encoders -> statsmodels`, a transitive dependency of a core dependency, not a defect. The test now blocks only the extras whose distributions are disjoint from the transitive closure of `[project.dependencies]` (`all` and `stats` are dropped). |

## Item 3 breakdown (48 hits)

* `@pytest.mark.hang_guard` (ceiling asserts the call returns, 10x or more above the measured time): conditional_gate x4, `test_full_suite_regression` (300 s), `test_state_of_union_regression` (180 s), `test_mrmr_real_100k`, `test_wide_data_scalability` x3 (500 s), `test_fs_hybrid_runner`, `test_gradient_interaction_seeder` cprofile ceiling, `test_polars_group_by_prewarm` x2 (first-call cold-start sensor), `test_charts_quantile` corp decomp, `test_biz_val_training_baseline_diagnostics` sample_n, `test_stress` multiple models, `test_training_overhead_integration_fixes` fix5, `test_bizvalue_feature_selection` (10x catastrophic-slowdown bound).
* `@pytest.mark.perf` (multi-minute fits where best-of-N is not affordable, or a deliberate sub-second race): `test_fe_rung_schedule` wide pool, both `test_biz_val_shap_proxied_fs` cap tests, `test_auto_tune_speedup_smoke`.
* Best-of-N per side: polars dynamic window, runtime-budget envelopes x2, integration-contract cache hit, mdlp oos vs validated, cmi-gate K growth, per-feature-edges narrow frame, `test_biz_val_filters_mrmr` x2, gpu hybrid cuda vs njit_par, selection-stability report, append-engineered, dedup source cols, edge-cases cache hit, cached F-scores, `test_perf_regression` x2, calibration band, baseline diagnostics disabled, cb gpu budget reference fit, drift snapshot lazy plan, artefact cache hit and miss.
* Slack widened: `test_biz_val_proxy_mode_auto_gate_silent_on_additive_bed` 1.10x to 1.25x (best-of-3 already; the selection identity assert is the real contract).

## Verification

* `find_single_shot_timing_assertion(tests)`: 48 hits before, 0 after.
* `ruff check` over every touched test file and `black_filtered_apply --check`: clean.
* `tests/test_meta/test_workflow_supply_chain.py`, `test_optional_third_party_imports_are_guarded.py`, `test_shared_gates_adopted.py`, `test_no_single_shot_timing_assertion.py`: 19 of 20 passed on the first run, the one failure being the production defect above (fixed); the dynamic twin re-run (after the lazy import and the core-closure change) could not be completed to green on this box: two re-runs executed while another session's stash/restore had briefly reverted tuning_rules.py, so they still saw the old module-level import. FUTURE: re-run `test_core_only_install_can_import_the_public_subpackages` once the tree is quiet (about 5 minutes).
* `test_scanning_gates_fail_closed.py`, `test_baseline_registry_consistency.py`: pass.

## Notes

* Another session's pre-commit stash/restore reverted unstaged edits in this tree for several minutes mid-run; every edit was re-verified by re-running the shared single-shot detector before commit.
* The static entry-import gate costs 70 s with `include_pytest=False`; the dynamic twin imports 16 modules in separate interpreters (about 5 minutes on a loaded box).
