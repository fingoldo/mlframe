# 2026-05-24 audit cycle — summary

This page is a brief reader-facing summary of the multi-wave audit landed on master between commits ``da68ca6`` and ``c31b419d`` (125 commits). Full per-finding disposition lives in ``audit/critique_2026_05_24/FINAL_VERIFICATION.md``; this file is the lightweight pointer for docs readers.

## Scope

23 critique deliverables covering 10 functional areas and code/architecture standards:

- **A1** Feature selection (21 findings)
- **A2** Feature engineering (25+ findings)
- **A3** Ensembling (16 findings)
- **A4** Performance hotspots (24 findings)
- **A5** Pipeline caching (16 findings)
- **A6** Polars zero-copy (40 findings)
- **B1** Tests expand — module gaps + weak asserts + fuzz axes + numba coverage + sklearn matrix (51 items)
- **B2** Tests optimize — markers, fixtures, importorskip, pytest infra (50 findings)
- **C1** Monolith split (18 file plans + 7 preventive)
- **D1** ML best practices (13 findings)
- **D2** Code / architecture standards (23 findings)

Plus 14 architectural proposals (AP1-AP14): SuiteArtefactCache, finite-mask threading, K-target joblib, fuzz-axes wiring, numba-coverage CI, safe_pickle, NNLS stacking-aware blend, nbytes streaming dispatch, sklearn-matrix compliance gate, dep upper-caps, pre-commit hooks, calibration policy, honest_diagnostics aggregator, provenance trail.

## Closure (post-Wave-9)

- **DONE**: 172 of 258 atomic-finding rows (66.7%) landed in code or test.
- **DEFERRED**: 86 rows in the Wave-10 backlog; reasons documented inline per row. Predominantly cosmetic / perf-hygiene / test-coverage extensions rather than correctness.
- **REJECTED with bench**: 13 rows (bench numbers in commit messages or inline ``# bench-attempt-rejected`` markers).
- **DROPPED (user)**: 4 items — AP3 K-target joblib, AP10 dep upper-bound caps, AP11-c CI continue-on-error drop, D2#5/S70 dep caps.

Per ``feedback_no_premature_closure``: this is explicitly a Wave-10 backlog, not a fully closed audit. See ``audit/critique_2026_05_24/FINAL_VERIFICATION.md`` section "Wave 10 backlog" for the prioritized remaining items.

## Architectural proposals landed

- **AP1** ``src/mlframe/training/suite_artefact_cache.py`` — cross-process disk cache for heavyweight suite artefacts.
- **AP2** F15 ``_finite_mask`` threading across 9 residual ``_fit`` kernels in ``composite_transforms``.
- **AP4** Fuzz axes F1/F2/F5/F6/F7 wired into ``tests/training/test_fuzz_regression_sensors.py``.
- **AP5** ``.github/workflows/numba-coverage.yml`` — nightly ``NUMBA_DISABLE_JIT=1`` coverage workflow.
- **AP6** ``src/mlframe/utils/safe_pickle.py`` — centralised sha256-sidecar verification across all four pickle entry points.
- **AP7** NNLS stacking-aware weights fed back into ``combine_probs``.
- **AP8** nbytes-aware streaming vs materialised ensemble dispatcher.
- **AP9** ``sklearn-matrix-ci.yml`` runs ``sklearn.utils.estimator_checks`` across sklearn 1.5.2 / 1.6.1 / 1.7.2 / 1.8.0.
- **AP11** ``.pre-commit-config.yaml`` enforces ruff + black + scoped mypy.
- **AP12** ``pick_best_calibrator`` policy with OOF ECE + bootstrap CI. See ``docs/calibration_policy.md``.
- **AP13** ``honest_diagnostics`` aggregator wired into finalize; default ON via ``ReportingConfig.honest_estimator_diagnostics``. See ``docs/honest_diagnostics_guide.md``.
- **AP14** ``provenance.py`` hyperparam trail wired into 19 source surfaces.

Dropped per user: AP3 (K-target joblib parallelism), AP10 (dep upper-bound caps), AP11-c (CI continue-on-error drop on linters).

## Monoliths carved

| File | Before LOC | After LOC | Sensor |
|---|---|---|---|
| ``composite_transforms.py`` | 1194 | 295 | ``test_monolith_split_w6a_composite_transforms.py`` |
| ``metrics/core.py`` | 1064 | 232 | ``test_monolith_split_w6b_core.py`` |
| ``_setup_helpers.py`` | 1058 | 356 | ``test_monolith_split_w6b_setup_helpers.py`` |
| ``_target_distribution_analyzer.py`` | 1017 | 188 | ``test_monolith_split_w6a_target_dist.py`` |

14 carves remain on the Wave-10+ backlog (``_phase_composite_post.py``, ``wrappers/_rfecv_fit.py``, ``_composite_target_estimator.py``, ``training/helpers.py``, ``training/neural/recurrent.py``, ``boruta_shap.py``, ``target_temporal_audit.py``, ``_phase_helpers.py``, ``baseline_diagnostics.py``, ``train_eval.py``, ``training/neural/flat.py``, ``extractors.py``, ``training/neural/ranker.py``, ``training/neural/base.py``); plans persisted in ``audit/critique_2026_05_24/monoliths-split.md``.

## Related guides

- ``docs/calibration_policy.md`` — AP12 ``pick_best_calibrator`` policy reference.
- ``docs/honest_diagnostics_guide.md`` — AP13 honest_diagnostics aggregator reference.
- ``docs/baseline_diagnostics_guide.md`` — pre-existing baseline diagnostics tutorial (orthogonal feature).
- ``audit/critique_2026_05_24/FINAL_VERIFICATION.md`` — full per-finding disposition table (528 lines).
