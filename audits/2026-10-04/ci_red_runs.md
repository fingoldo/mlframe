# Red CI runs on master, triage and fixes

Source: `gh api repos/fingoldo/mlframe/actions/runs?branch=master`, jobs and raw job logs. Base tree origin/master 3a9678dc5 (master later moved to 75487cc48).
The main `CI` workflow runs for 7da5ecb64 (37221641979), 372868f8c and dcceabd52 were cancelled by the old concurrency policy; 3a9678dc5 (37224458006) was still pending,
so the only complete evidence for the CI workflow is run 37202909154 (d7b1f79ad) plus local reproduction. Linux-only failures were reproduced locally; macOS-only ones were reasoned from logs.

## Fixed in this change

### sklearn-matrix, runs 37219300228 (372868f8c) and 37215223224 (dcceabd52), all 5 jobs (1.6.1/py3.11, 1.7.2/py3.11, 1.8.0/py3.11, py3.13, py3.14)
- Test: `tests/training/composite/screening/test_composite_gate_and_edges.py::TestCorrThresholdEdges::test_corr_exactly_at_threshold_drops`
- Log: `assert all(s.base_column != "x_near_y" for s in disc.specs_)` -> `assert False` (line 364). Reproduced locally (same failure).
- Root cause: the test names `base_candidates=["x_near_y"]` explicitly. Production now deliberately readmits an explicitly named base past the corr filter
  (`composite/discovery/__init__.py`, "explicit base_candidates readmitted past the corr filter", pinned by `test_explicit_base_overrules_corr_filter.py`).
  The test predates that contract. The filter still records the drop (first assertion passes).
- Fix: test uses `base_candidates="auto"` so the corr filter applies to automatic selection, which is what the boundary test is about. Production unchanged.
- Validation: the test passes locally (1 passed); the three other tests of the class pass.

### codecov-full, run 37212057965 (d7b1f79ad), job "Merge ci.yml + numba-coverage.yml + deep-nightly raw coverage"
- Log: `No source for code: '/home/runner/work/mlframe/mlframe/src/mlframe/feature_engineering/_welford_njit.py'` then exit 1 in `coverage xml`/`report`.
- Root cause: the merge combines raw `.coverage` data from three workflows run on different commits. The data from the older commit references
  `feature_engineering/_welford_njit.py`, which e7d2d89ab added and ef3448d85 deleted, so coverage cannot find the source and aborts. Inherent to cross-commit merging.
- Fix: `.github/workflows/codecov-full.yml` uses `coverage xml --ignore-errors` and `coverage report --ignore-errors` (CLI only; the repo coverage config is not weakened).
- Validation: workflow-only change, not runnable locally; one-line flag change on commands that exist in coverage 7.x.

### CI run 37202909154 (d7b1f79ad): job "Black tests/ (filtered, blocking)" 111455977062
- Log: `1/4194 files have non-excluded-class Black findings: tests/training/test_filter_polars_cat_hoisted.py`. Reproduced.
- Fix: `black_filtered_apply --write` on that file. Validation: `--check` exits 0.

### CI run 37202909154: job "lint + security tests/ (blocking)" 111455977070 (vulture exit 3)
- Log: unused variable `quiet_host_for_ks_timing` (tests/metrics/classification/test_classification_extras.py:336), `member_sel`
  (tests/test_meta/test_broad_except_logging_benchmarks_harness.py:370), `X_drop`, `seed_indices`
  (tests/training/test_audit_stable_sort_cluster_followup.py:199, 254), `models_and_predictions` (tests/training/test_default_via_or_trap.py:122). Reproduced with vulture 2.16.
- Root cause: a fixture parameter and stand-in callables whose signatures must match the real callee.
- Fix: the five names added to `tests/vulture_whitelist.py` (the file's documented convention).
- Validation: `vulture tests tests/vulture_whitelist.py --min-confidence 80` exits 0.

### CI run 37202909154: macOS shards 6/10 (111455998784), 9/10 (111455998872), 10/10 (111455998897)
Four of these are platform-independent and failed identically on Windows locally.

| Test | Log | Root cause | Fix |
|---|---|---|---|
| `tests/inference/test_predict_ct_ensemble_save_load.py::test_persist_ct_ensemble_entries_roundtrips`, `tests/inference/test_predict_cte_raw_x.py::test_predict_mlframe_models_suite_matches_oracle[True/False]`, `tests/training/test_conversions_conversions_coverage.py::test_predict_native_probe_loads_each_model_once` | `RuntimeError: ... sha256 sidecar missing or mismatched` (`verify_sidecar: no .sha256 sidecar ... refusing to load (default-strict)`) | Loaders are fail-closed on a missing sidecar by design (pyutilz safe_pickle). The tests hand-write `metadata.pkl.zst` with no sidecar. | Tests call `write_sidecar` after writing the metadata file. Production fail-closed behaviour kept. |
| `tests/training/test_oof_temporal_and_seed.py::test_oof_iid_uses_shuffled_seeded_kfold` | `expected KFold for i.i.d., got StratifiedKFold` | `iid_oof_splitter` deliberately uses shuffled seeded `StratifiedKFold` for a classifier; test asserted the old splitter. Shuffle and seed assertions unchanged. | Assert `StratifiedKFold` (then the existing shuffle/random_state assertions). |
| `tests/training/test_feature_handling_high_feature_handling.py::test_h_fh_10_stale_lock_retry_uses_fresh_filelock` | `DID NOT RAISE Timeout` (the dead-PID reclaim warning did fire) | On POSIX the reclaim unlink succeeds while the holder keeps its flock on the old inode, so the fresh lock on the recreated path is free and the retry acquires. The Timeout is Windows-only (open file cannot be unlinked). | Test expects Timeout only on `os.name == "nt"`; on POSIX the acquire must succeed. The StaleLockReclaimed warning assertion stays on both. |
| `tests/training/test_finally_exception_mask.py::test_pipeline_temp_target_drop_wrapped` | the debug record `temp_target_col drop failed` is missing | Passes locally; on the shard the record never reached caplog, consistent with a `mlframe*` logger level raised by earlier tests in the same xdist worker (`caplog.at_level(DEBUG)` only sets the root logger). Not reproduced, inferred. | `caplog.at_level(logging.DEBUG, logger="mlframe.training.pipeline")` sets the emitting logger explicitly. |
| `tests/training/test_audit_fe_transformer_stable_sort.py::test_apriori_itemsets_uses_lexsort`, `tests/training/test_audit_stable_sort_determinism.py::test_fca_closed_concepts_topk_uses_content_tiebreak` | `No module named 'mlxtend'`, `fca_closed_concepts requires concepts library` | Optional dependencies not installed on the macOS runner. | `pytest.importorskip` with a precise reason. |

Validation: 11 tests (sidecar, oof) and the other 4 edited tests pass locally; ruff clean, filtered Black clean on every edited file.

## Already fixed on master by a newer commit (verified against the tree)
- macOS shard 9/10 `test_f45_cudnn_autotune_skipped_pre_ampere` (`assert None is False`): cudnn.deterministic leaked from earlier tests; test now monkeypatches it (3d21a5ee9 line 172).
- macOS shard 9/10 `test_deserialize_materialises_arrays_before_close` (`owndata` False): test changed since d7b1f79ad (3d21a5ee9).
- deep-nightly 37105028510 (93aa28cf5, 9 shards): `UnboundLocalError: '_r'` at `_fe_stage_cascade_early_b.py:786 return X, _r`, dozens of MRMR/FE failures. Code is gone at HEAD, fixed by 590e17ad1.
- deep-nightly shard 19/20: `_main_train_suite.py LOC=785 exceeds 785 budget`. The facade is 763 LOC at HEAD (92d6a4cf1 moved the render-scope helpers out).
- fs-benchmark-nightly 37109227437: `ModuleNotFoundError: No module named 'pywt'` from `feature_engineering/_timeseries_emit.py:17`; pywt is imported lazily at HEAD (ae6d4f1fd).
- MyPy / mypy-full failure at d7b1f79ad (37202909103, 10 errors in `_recipe_dispatch.py`, `_dummy_bootstrap.py`): cleared by dcceabd52; mypy-full is green on 7da5ecb64 and 3a9678dc5.
- numba-coverage-nightly failure 34326253688 is from an older commit; later runs are skipped (not scheduled), nothing red.

## Left, with reason
- deep-nightly 37105028510 (93aa28cf5), not re-run, nightlies since were cancelled so no fresher evidence exists:
  - `test_fe_float32_replay_parity` f32 lost `esc_poly_legendre_mul(x0,x1)`, `test_biz_val_boruta_heavy_tail_label_noise...` (BorutaShap), `test_biz_value_mrmr_usability_raw_retention`,
    `test_f2_single_compound_across_distributions[with_outliers]`, `test_I5_fe_produces_recoverable_structure_uplift[...]`: MRMR FE / BorutaShap areas owned by other agents in this round; not touched.
  - 11 `tests/feature_selection/shap_proxied/test_biz_val_*` timeouts (>900 s) plus `faith_interaction` (bed premise) and `proxy_mode_auto` (wall ratio 76.95 s vs 1.10x68.96 s): wall-clock sensitive on a loaded 4-core runner,
    no code failure in the log; needs a fresh nightly to see whether they persist after the `_r` fix (many shard-wide failures masked the picture).
  - `test_biz_multimode_beats_single_mode_on_multimode_target[default]`: delta 0.0197 vs 0.02 floor, a 1.5 percent margin miss on a seeded biz-value test; needs multi-seed measurement before any threshold or production change.
- macOS shards 1-5 and 7 of 37202909154 were still running when read; their results (and the run for 3a9678dc5, pending) are not covered. Re-check after the next push.
- macOS LightGBM libomp crash (tracker X5): no new evidence in these logs.
