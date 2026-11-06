# Wave 4 tracker: full attack-class audit of mlframe (2026-11-06)

Eight read-only Opus 5.5 (low effort) audits, one per attack class. Reports sit next to this file. Dispositions: RESOLVED (fixed, regression test in the commit), OPEN (verified, not yet fixed), FUTURE (real, deferred, with reason), DOC (documented, with reason), REJECTED (not a defect, with reason).

| ID | Sev | Status | Location | Note |
|---|---|---|---|---|
| W4-INP-01 | Medium | RESOLVED | `feature_engineering/categorical_powerset_concat.py:78,83-88` | Default `max_order=None` enumerated 2^n-n-1 subsets with no cap; n=20 gave about 1M composite columns. Now `max_subsets=4096` (default) refuses above the cap before any column is built; test `tests/feature_engineering/test_categorical_powerset_subset_cap.py` fails on the old behaviour (the concat helper was reached) and passes now; the 4 existing powerset biz-value tests pass. Production calls use `max_order=2` and are unaffected. |
| W4-INP-02 | Medium | RESOLVED | `feature_selection/shap_proxied_fs/_shap_proxy_interactions.py:253,267-269` | Exhaustive P<=16 pass scored all 65,535 subsets with an n x r x r copy each. Now the exhaustive pass stops at cardinality `exhaustive_max_card=3` (696 subsets at P=16); the greedy pass still covers larger sets. Test `tests/feature_selection/shap_proxied/test_interaction_exhaustive_pass_is_bounded.py` fails on the old code (65535 calls) and passes now; 10 existing interaction tests pass. Behaviour change for P<=16: exhaustive results above cardinality 3 now come from the greedy pass. |
| W4-INP-03 | Low | FUTURE | `feature_selection/filters/_usability_aware_selection.py:239` | Full pair list is built before `max_pairs` caps it; needs a streamed or ranked selection. |
| W4-SUB-01 | Low | RESOLVED | `feature_selection/_benchmarks/profiling/env_sampled_fit.py:16`, `utilization.py:25` | `nvidia-smi` called by bare name; both sites now use `training/_gpu_state_probe._resolve_smi()`. Test `tests/feature_selection/test_profiling_gpu_smi_resolved_from_absolute_path.py` passes; the bare-name call fails its absolute-path assertion (`os.path.isabs('nvidia-smi')` is False). |
| W4-SUB-02 | Info | DOC | `reporting/_benchmarks/_e2e_slice_finder_ab_iter71.py:48` | `git show` ref has no `--end-of-options`; the operator controls the value. |
| NET4-01 | Info | DOC | `integrations/mlflow.py:93-96` vs `:106` | Docstring says "no sanitization"; the code escapes. Docstring fix only. |
| NET4-02 | Low | OPEN | `integrations/mlflow.py:106,111` | `html.escape` does not block `javascript:` or `data:` src schemes in the artifact iframe. |
| DYN4-1 | Low | OPEN | `training/composite/_estimator_dispatch.py:68-77` | Refusal list, not an allow-list, for a recommendation target; recommended to make it an allow-list. |
| DYN4-2 | Info | DOC | `training/neural/_flat_torch_module/__init__.py:215-225` | `getattr` walk under `torch.optim`; the module guard already limits it. |
| W4-SUP-01 | Low | OPEN | `.pre-commit-config.yaml:126,200,459` | ruff pinned by a mutable tag, other remote hooks pinned by SHA. |
| W4-SUP-02 | Low | FUTURE | `.github/workflows/hooks-not-in-ci.yml:56` | `pre-commit>=3.3` floats; same remedy as SUP-05, deferred with the other dependency pins. |

Checked and clean: deserialization (no new findings; earlier DESER-01..22 rulings hold), filesystem and temp files (0), secrets (0).
