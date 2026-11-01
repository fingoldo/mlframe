# Full attack-surface audit 2026-11-01: tracker

Ten read-only reports on `src/mlframe` (and the workflow and pre-commit files for supply chain), one per attack class. No code was changed by the audit. Vocabulary: **RESOLVED** (already safe, citation given), **FUTURE** (real, deferred, reason given), **DOC** (intended behaviour, to be documented), **REJECTED** (not a risk, reason given). Each row is a proposal; none is executed. Per the project's audit rule, the owner triages before any code change.

Carried forward from the 2026-10-31 round are cross-referenced, not re-listed. The 2026-10-31 DESER-10 is now RESOLVED in code (`composite/cache_store.py:385`), so its tracker row is stale.

## Counts by class

| Class | Report | High | Medium | Low | Info | Total |
|---|---|---|---|---|---|---|
| Unsafe deserialization | [deserialization.md](deserialization.md) | 0 | 7 | 3 | 1 | 11 |
| Dynamic code execution | [dynamic_exec.md](dynamic_exec.md) | 0 | 0 | 2 | 6 | 8 |
| Subprocess and shell | [subprocess.md](subprocess.md) | 0 | 0 | 2 | 1 | 3 |
| Network and servers | [network.md](network.md) | 0 | 1 | 1 | 4 | 6 |
| Filesystem and temp files | [filesystem.md](filesystem.md) | 0 | 0 | 2 | 2 | 4 |
| Secrets and credentials | [secrets.md](secrets.md) | 0 | 0 | 1 | 2 | 3 |
| Dependencies and supply chain | [supply_chain.md](supply_chain.md) | 0 | 1 | 2 | 5 | 8 |
| Untrusted input and resources | [untrusted_input.md](untrusted_input.md) | 0 | 1 | 1 | 5 | 7 |
| Concurrency and TOCTOU | [concurrency.md](concurrency.md) | 0 | 2 | 2 | 3 | 7 |
| Regex and ReDoS | [redos_regex.md](redos_regex.md) | 0 | 0 | 2 | 4 | 6 |
| **Total** | | **0** | **12** | **18** | **33** | **63** |

No High finding in any class.

## Findings

| ID | Class | Severity | Location | Proposed disposition |
|---|---|---|---|---|
| DESER-01 | deserialization | Medium | `training/io.py:601` | DOC |
| DESER-02 | deserialization | Medium | `training/io.py:524-612` (`load_mlframe_model`, no sha256 gate) | FUTURE |
| DESER-03 | deserialization | Medium | `training/_unpickle_policy.py` (allowlist breadth) | FUTURE |
| DESER-04 | deserialization | Medium | `inference/predict.py:245` | FUTURE |
| DESER-05 | deserialization | Low | `inference/predict.py:123` | FUTURE |
| DESER-06 | deserialization | Medium | `estimators/pipelines.py:111` (`replay_cv_results`, plain `joblib.load`) | FUTURE |
| DESER-07 | deserialization | Medium | `training/core/_metadata_loader.py:20-21` | FUTURE |
| DESER-08 | deserialization | Medium | `training/feature_handling/cache.py:578` | DOC |
| DESER-11 | deserialization | Low | `training/_trainer_train_and_evaluate_parts.py:117-126` | FUTURE |
| DSR2-01 | deserialization | Low | `core/helpers.py:34-50`: `validate_trusted_path` uses `abspath`, not `realpath`, so a symlink inside the root escapes it | FUTURE |
| DSR2-02 | deserialization | Info | `_resolve_denylisted_class` denies by module prefix, not by resolved object | DOC |
| DYN2-1 | dynamic_exec | Low | `training/neural/_flat_torch_module/__init__.py:211-220`: `__setstate__` walks an unrestricted `getattr` chain (behind pickle) | FUTURE |
| DYN2-2 | dynamic_exec | Low | `training/composite/_estimator_dispatch.py:70-72` (duplicate of EXE-10) | FUTURE |
| DYN2-3 | dynamic_exec | Info | `calibration/post.py:79, 143, 154` | RESOLVED |
| DYN2-4 | dynamic_exec | Info | `feature_selection/_benchmarks/kernel_arg_dtype_audit.py:190-193` | RESOLVED |
| DYN2-5 | dynamic_exec | Info | `feature_selection/_benchmarks/profiling/nvprof_autotag.py:40-44, 61-62` | RESOLVED |
| DYN2-6 | dynamic_exec | Info | `_LazyModule` copies (`estimators/pipelines.py:57`, `models/_optimization_shared.py:35`, `feature_engineering/mps.py:58`, `metrics/calibration/_calibration_plot.py:46`) | RESOLVED |
| DYN2-7 | dynamic_exec | Info | `data/synthetic.py:90-104` | RESOLVED |
| DYN2-8 | dynamic_exec | Info | `feature_selection/_benchmarks/bench_mrmr.py:85`, `_benchmarks/fs_hybrid/_manifest.py:110` | RESOLVED |
| SUB2-01 | subprocess | Low | `utils/native_import_probe.py:55-58`, `training/_polars_native_support.py:90-93`, `system/kernel_tuning_cache/_ensure.py:164`: `-c`/`-m` children put cwd on `sys.path`, so a planted `polars.py` runs in the child | FUTURE |
| SUB2-02 | subprocess | Low | `training/_gpu_state_probe.py:86-88`: on Windows `shutil.which("nvidia-smi")` searches cwd first | FUTURE |
| SUB2-03 | subprocess | Info | `__init__.py:83-135`: CUDA env autoconfig inherited by children, by design | REJECTED |
| NET2-01 | network | Medium | `training/feature_handling/hf_provider.py:207`: `bool("False")` is `True`, so `hf://org/model?trust_remote_code=False` runs the hub repo's code | FUTURE |
| NET2-02 | network | Low | `integrations/mlflow.py:36, 41`: userinfo scrubber leaks part of a password containing `@` | FUTURE |
| NET2-03 | network | Info | `feature_selection/filters/_neural_mi.py:280-300`: InfoNet checkpoint has no SHA-256 check (load is `weights_only=True`, so code execution is closed) | FUTURE |
| NET2-04 | network | Info | `_neural_mi.py:310-318, 425-431` | DOC |
| NET2-05 | network | Info | `feature_selection/boruta_shap/__init__.py:823-829` | DOC |
| NET2-06 | network | Info | `integrations/mlflow.py:71-96` | DOC |
| FS2-1 | filesystem | Low | `training/suite_artefact_cache.py:199-201`: `_path` builds `f"{key}.pkl"` with no traversal guard | FUTURE |
| FS2-2 | filesystem | Low | `training/suite_artefact_cache.py:203-205`: cache dir created without `mode=0o700` | FUTURE |
| FS2-3 | filesystem | Info | `training/_training_loop_refit.py:70-80`: config-only path | REJECTED |
| FS2-4 | filesystem | Info | `feature_engineering/transformer/_key_bank.py:168, 251-291`: fingerprint is always 64 hex | REJECTED |
| SEC2-1 | secrets | Low | `training/feature_handling/providers.py:235-240`: userinfo in a non-secret param is hashed into the cache signature (cross-ref SEC-04) | FUTURE |
| SEC2-2 | secrets | Info | `training/feature_handling/hf_provider.py:222-225`: model string logged at INFO | DOC |
| SEC2-3 | secrets | Info | `tests/integrations/test_mlflow_scrubbed_giveup_log.py:15`: fake fixture credential | REJECTED |
| SUP2-1 | supply_chain | Medium | `.github/workflows/release.yml:70`, `ci.yml:754`: unpinned `uvx twine`; release build job holds `id-token: write` | FUTURE |
| SUP2-2 | supply_chain | Low | `.pre-commit-config.yaml` (9 places): hook revs pinned to mutable tags, not SHAs | FUTURE |
| SUP2-3 | supply_chain | Low | `.pre-commit-config.yaml:291`: mypy pins do not match `uv.lock` | FUTURE |
| SUP2-4 | supply_chain | Low | `feature_selection/filters/_vendored/infonet/NOTICE.md`: provenance note | DOC |
| SUP2-5 | supply_chain | Info | `.github/workflows/ci.yml:421` | RESOLVED |
| SUP2-6 | supply_chain | Info | several workflow `uses:` lines (`ci.yml`, `release.yml`, `dep-floors.yml`, nightlies) | DOC |
| SUP2-7 | supply_chain | Info | `.github/workflows/release.yml:31-37`: PyPI gate, re-verified | RESOLVED |
| SUP2-8 | supply_chain | Info | `fingoldo/py-ci-shared/.github/actions/*` call sites | DOC |
| INP2-1 | untrusted_input | Medium | `feature_selection/shap_proxied_fs/_shap_proxy_search.py:354`: `generate_combinations` materialises every C(n,r) row; forced `bruteforce` and a user `prescreen_top` have no byte cap | FUTURE |
| INP2-2 | untrusted_input | Low | `feature_engineering/transformer/multi_aux_ensemble.py:60,69,77,82` (and four sibling constructors): `num_leaves=2**max_depth` uncapped | FUTURE |
| INP2-3 | untrusted_input | Info | `feature_selection/filters/engineered_recipes/_recipe_name_simplify.py:72-91` | RESOLVED |
| INP2-4 | untrusted_input | Info | `inference/predict.py:111-112` | DOC |
| INP2-5 | untrusted_input | Info | `training/cb/_cb_gpu_budget.py:58-59` | REJECTED |
| INP2-6 | untrusted_input | Info | `training/core/_metadata_loader.py:27-29` | RESOLVED |
| INP2-7 | untrusted_input | Info | `while True` loops in `boruta_shap/_shadow_stats.py`, `_synthetic_bases.py`, `_mi_greedy_cmi_fe.py`: all terminate | RESOLVED |
| CON2-1 | concurrency | Medium | `system/kernel_tuning_cache/_ensure.py:132-147`: tuning lock created before PID is written; stale takeover unlinks without re-check, so two sweeps can run at once | FUTURE |
| CON2-2 | concurrency | Medium | `feature_selection/filters/_analytic_mi_null.py:90-96`: process-wide `_min_n_override` context manager is not thread-safe | FUTURE |
| CON2-3 | concurrency | Low | `training/feature_handling/locking.py:130-136`: stale-lock reclaim can unlink a live lock held by a third process | FUTURE |
| CON2-4 | concurrency | Low | `training/_training_loop_refit.py:62-80` | DOC |
| CON2-5 | concurrency | Info | `feature_engineering/transformer/_key_bank.py:240-293` | RESOLVED |
| CON2-6 | concurrency | Info | see finding body in `concurrency.md` | RESOLVED |
| CON2-7 | concurrency | Info | `training/suite_artefact_cache.py:161, 372-385` | DOC |
| RDS2-1 | regex_redos | Low | `integrations/mlflow.py:36`: userinfo redaction is quadratic on a long alphanumeric run with no `://` | FUTURE |
| RDS2-2 | regex_redos | Low | `reporting/output.py:55`: `plot_outputs` clause regex, overlapping `\s` quantifiers | FUTURE |
| RDS2-3 | regex_redos | Info | `preprocessing/_cleaning_helpers.py:188`: `exclude_mask` is an operator-trusted regex | DOC |
| RDS2-4 | regex_redos | Info | composite-target forbidden base patterns from config | DOC |
| RDS2-5 | regex_redos | Info | calibration include/skip patterns from config | DOC |
| RDS2-6 | regex_redos | Info | per-group aggregate column pattern from config | DOC |

## Items to triage with the owner before any change

- **Medium (12 findings):** 10 are FUTURE (DESER-02, DESER-03, DESER-04, DESER-06, DESER-07, NET2-01, SUP2-1, INP2-1, CON2-1, CON2-2); 2 are DOC (DESER-01, DESER-08).
- **Real and cheap to fix:** NET2-01 (`bool()` parsing of `trust_remote_code`), SUP2-1 (pin `twine`), DSR2-01 (`realpath` in `validate_trusted_path`), CON2-1 (lock ordering).
- **RESOLVED rows** need their citation checked before the tracker marks them closed.

## Limits stated by the reports

- Secrets: the key-material regexes (AKIA, ghp_, PRIVATE KEY) were not re-run across tests and YAML.
- Untrusted input: sampled, not exhaustive.
- Concurrency: the kernel-tuning write path lives in external `pyutilz` and was out of scope.
- Supply chain: no live PyPI or branch-protection check from this environment; SUP-02 from the prior round still holds (`pyutilz` 404 on PyPI).
- Subprocess: the scan found no `os.system`, `os.popen`, or `shell=True` in non-benchmark code.

## Implemented in this round

| ID | Commit | What changed | Regression test |
|---|---|---|---|
| NET2-01 | `d3a067b00` | `trust_remote_code` parsed as a flag (`parse_trust_remote_code`); unknown values refused | `tests/training/feature_handling/test_hf_trust_remote_code_parse.py` (imports the new helper, so it fails on the previous code at import) |
| DSR2-01 | `1c3e12f11` | `validate_trusted_path` resolves both sides with `realpath` | `tests/core/test_validate_trusted_path_symlink_escape.py` (fails on the previous code) |
| SUP2-1 | `b2955e718` | `twine` pinned to 6.2.0 in `ci.yml` and `release.yml` | workflow change, checked by YAML parse and the `git diff` |

| INP2-1 | `94f9a13e9` | `brute_force_top_n` refuses a combination table above 1 GiB before materialising it | `tests/feature_selection/shap_proxied/test_brute_force_combination_cap.py` (the refusal test was not run on the previous code, which would allocate the whole table) |
| CON2-1 | `d77a62c80` | tuning lock created with its pid already in it (hard link); a young empty lock counts as live | `tests/system/test_kernel_tuning_lock_race.py` (the young-lock test fails on the previous code) |
| CON2-2 | `80ec158f5` | analytic-null floor override kept per block in a locked list, not a saved and restored global | `tests/feature_selection/filters/test_analytic_null_override_overlap.py` (both tests fail on the previous code) |

| DESER-07 | `8a02c732b` | joblib suite metadata needs a sha256 sidecar like the pickle kinds | `tests/training/test_metadata_loader_sidecar_required.py` (joblib refusal fails on the previous code) |
| DESER-02 | `71d53db0e` | model bundle bytes must match `bundle_sha256` before unpickling; three tests that pinned the old behaviour updated | `tests/training/test_load_model_bundle_digest_required.py` (tampered bundle fails on the previous code) |
| DESER-06 | `981c6a956` | gridsearch dump read through `safe_joblib_load` | `tests/estimators/test_replay_cv_results_refuses_gadget.py` (marker created on the previous code) |
| DESER-03/04 | DOC | denylist trust model stated in the `safe_joblib_load` docstring | docstring (`io.py`) |
| SUB2-01 | `9f0ee5cd2`, `28161bf23` | `-P` on the probe, polars probe and tuning children | `tests/utils/test_native_import_probe_cwd_not_on_path.py` (fails on the previous code) |
| SUB2-02 | `5093beb1b` | nvidia-smi resolved from absolute PATH entries only | `tests/training/test_gpu_state_probe_smi_not_from_cwd.py` (fails on the previous code) |
| INP2-2 | `c52afb8f0`, `5ac588b1e` | auxiliary LightGBM leaves capped at 63 | `tests/feature_engineering/transformer/test_aux_leaf_cap.py` (LightGBM fatal on the previous code) |
| CON2-3 | `ee447a266` | held feature-handling lock never reclaimed | `tests/training/test_feature_handling_high_feature_handling.py::test_h_fh_10_*` (rewritten; does not separate the versions on Windows, POSIX check pending) |
| SEC2-1 | `6f00c4c4a` | URL userinfo scrubbed from provider signature and repr | `tests/training/feature_handling/test_provider_signature_strips_url_credentials.py` (fails on the previous code) |
| DYN2-1 | `26f0976bd` | optimizer class restored only from `torch.optim` | `tests/training/neural/test_flat_torch_setstate_optimizer_ref.py` (fails on the previous code) |
| DYN2-2 / EXE-10 | `66b4221a1` | recommendation constructs only an estimator class | `tests/training/composite/test_instantiate_recommended_estimator_class_only.py` (fails on the previous code) |
| SUP2-2 | `3f22d8738` | external pre-commit revs pinned to SHAs (ruff stays on its tag: the shared check requires it) | config |
| NET2-03 | `6cdf0b0c2` | InfoNet checkpoint pinned by sha256, trust on first use | `tests/feature_selection/filters/test_infonet_checkpoint_digest.py` (replaced-file test fails on the previous code) |
| RDS2-1, NET2-02 | `17f8f9840` | userinfo scrub redacts a whole password and runs in linear time | `tests/integrations/test_mlflow_userinfo_scrub.py` |
| RDS2-2 | `4108b3871` | plot-output DSL linear on whitespace runs | `tests/reporting/test_plot_output_dsl_linear.py` |
| FS2-1, FS2-2 | `1579c7253` | cache keys guarded, directory mode 0o700 | `tests/training/test_suite_artefact_cache_key_guard.py` |
| (tests) | `075d68530` | GPU-stalling tests `slow`, CPU-heavy tests `timeout(3600)` | policy commit |

**SUP2-3 (DOC):** the mypy pins in pre-commit differ from `uv.lock` on purpose: they reproduce the CI environment where mypy errors were traced to a different numpy, scipy, scikit-learn and pydantic combination (see the comment above the hook). Changing them needs a whole-project mypy run, so the drift is documented here and checked on every pre-push.

## DOC rows documented here

- **DESER-01**: `dill.load` in `load_mlframe_model` runs only with `safe=False`, which warns; no in-repo caller passes it. Documented: `safe=False` is an explicit trust-the-source opt-in.
- **DESER-08**: the feature-handling cache falls back to pickle with `allow_pickle=True` only for its own files. Documented: that cache directory is trusted like the process itself.
- **DSR2-02**: the denylist matches module prefixes, not resolved objects. Documented as part of the `safe_joblib_load` trust model (see DESER-03/04).
- **NET2-04, NET2-05, NET2-06**: intended behaviour of the InfoNet download hint, the Boruta-SHAP network path and the MLflow scrub order. Documented as stated in the audit report.
- **SEC2-2**: model string logged at INFO on the HF provider. Documented: the model name is not a secret.
- **SUP2-4**: the vendored InfoNet NOTICE is the provenance record. Documented.
- **SUP2-6, SUP2-8**: `uses:` lines pinned to tags in workflows and py-ci-shared actions are the project's pinning convention for those repos. Documented.
- **INP2-4**: `inference/predict.py` accepts caller-sized inputs by design. Documented.
- **CON2-4**: `_training_loop_refit` shares a process-wide state by design for one fit. Documented.
- **CON2-7**: `suite_artefact_cache` lock and write ordering is per process. Documented.
- **RDS2-3, RDS2-4, RDS2-5, RDS2-6**: regexes from operator config are trusted input, not data; the field descriptions say they must be linear-time. Documented.

The other FUTURE and DOC rows above are not implemented yet. Each needs the owner's triage first.
