# py-ci-shared gate adoption, mlframe

Wired in `tests/test_meta/test_shared_gates_adopted_pass2.py` (commit "Wire 25 more py-ci-shared gates into test_meta and fix what they found").

## Adopted

| gate | result | disposition |
|---|---|---|
| atomic_write_staging | clean | zero-tolerance |
| clock_day_boundary | clean | zero-tolerance |
| hash_key_determinism | clean | zero-tolerance |
| lf_file_writes | clean | zero-tolerance |
| machine_specific_paths | 1 in src (docstring) | fixed; `_benchmarks`, `benchmarks`, `profiling`, `scripts` skipped (60 `D:/Temp` scratch paths in developer scripts) |
| pickle_state_completeness | clean | zero-tolerance |
| plotly_annotation_loop | 1 in a `_benchmarks` script | `_benchmarks` skipped |
| polars_null_equality | clean | zero-tolerance |
| reiterated_iterable_params | clean | zero-tolerance |
| stdlib_json_ban | 21 sites in 17 src files | 12 files moved to orjson; 5 kept via `allow` with reasons (digest-bound spec, cache key, composite id, canonical fallback, raw_decode) |
| stub_signature_parity | clean | zero-tolerance |
| vendored_internal_imports | 3 `joblib.externals.loky` | marked `# vendored-ok` (must be the copy joblib itself runs) |
| wrapper_protocol_parity | clean | zero-tolerance |
| workflow_runner_labels | 45 | fixed with `--fix` in 14 workflows |
| api_floor | `threading.__excepthook__` (3.10) | fixed with `getattr`; test marked `slow` (about 10 min) |
| coverage_config_parity | clean | zero-tolerance |
| ci_install_covers_conftest | gpu-matrix run-time extras | acknowledged with reason |
| ci_install_covers_entry_imports | clean | zero-tolerance (about 4 min) |
| ci_default_branch_never_cancelled | clean | zero-tolerance |
| pytest_addopts_path_runs | needed repo root as package root | wired |
| id_keyed_cache_validates_identity | clean | zero-tolerance |
| persisted_negative_probe | clean | zero-tolerance |
| committed_line_endings | 115 CRLF blobs | ratchet baseline |
| drifted_duplicate_literals | 109 (26 literal sets, 70 named constants, 12 thresholds) | only the literal-set rule adopted; two real copies removed (normal quantile coefficients shared via `norm_ppf`, Newton-Schulz coefficients shared via `NS5_COEFFS`); remaining 26 groups baselined with reasons. Named-constant and threshold rules are coincident epsilons and sweep points, not adopted |
| unresolved_module_attributes | 103 | 80 are `MRMR` read as the package directory `mrmr` on a case-insensitive file system; 23 are same-named-module false positives (`mlflow.py`, `matplotlib.py`, `io.py`) or `_benchmarks`. Baselined; tests tree is scanned only on a case-sensitive file system |
| gate_population_canary | 13 gates without a declared population | `_candidate_files()` added to each; canary tests added |

## Not adopted

| gate | disposition |
|---|---|
| commit_metadata | 29 recent commits carry a Co-Authored-By trailer and history cannot be rewritten. Needs a forward-only range |
| tracked_secret_shapes | no vendor-specific token shape to declare; detect-secrets already runs in pre-commit |
| unsafe_deserialization | only `_benchmarks` findings; a separate session wired its own gate |
| hardcoded_seed_in_library, cuda_kernel_integer_width, unread_function_params, single_shot_timing_assertion, bare pickle | covered by local or concurrently added equivalents |
| hook_hygiene | passes against `.git/hooks`; hooks are managed by pre-commit, not wired |
| mutation_teeth, teeth_sweep | mutation runners, too expensive for the meta suite |
| alembic_concurrently, arb_checks, dart_scanners, sql_*, db_transaction_completeness, connection_liveness_kwargs, connect_error_echo, edge_function_hygiene, embedded_postgres, schema_snapshot_parity, llm_call_archive_gate, prompt_field_parity, hardcoded_token_ceilings, external_fact_tables, statement_compilation, rollback_then_continue, index_coverage, destructive_tests_throwaway_only, env_example_round_trip, config_call_site_parity | not applicable: no database, Dart, Alembic, edge functions or LLM call archive in mlframe |
| plugins (resource_leak_guard, stub_signature_guard, randomly_seed_guard, offline_suite_without_credentials) | pytest plugins that change every run; out of scope for a meta-test pass |
| ci_health, config_drift_check, adoption_matrix, baseline_trend, worktree_hygiene, sibling_floor_skew, hook_attestation, closed_audit_rounds, audit_path_references, version_tag_currency | cross-repo reports or history checks, not per-test gates |
| import_layering, constant_relations, protocol_attributes, dataclass_case_completeness, save_failure_markers, prose_numeric_claims, unexecuted_function_bodies, cross_package_private_names | need declared rules or a coverage file; no rule set exists for mlframe yet (cross_package_private_names has a local equivalent) |

## Findings for py-ci-shared

- `unresolved_module_attributes` resolves an absolute `import mlflow` inside `mlframe/integrations/mlflow.py` to the file itself (implicit-relative resolution), and does the same for `io` and `matplotlib`.
- The same gate matches names case-insensitively on Windows, so `MRMR` resolves to the directory `mrmr`.
- `gate_population_canary.gate_modules` returns every `test_*.py`, and `load_gate` fails on test files that cannot be imported standalone; mlframe filters to modules defining `_candidate_files`.
