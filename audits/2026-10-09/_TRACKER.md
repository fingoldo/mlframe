# follow-up round 2026-10-09 -- master tracker

Eight low-hanging improvements proposed after the CI-red cleanup of 2026-10-08, approved by the owner and closed in one pass. The rows below are the findings of [01_followup_proposals.md](01_followup_proposals.md) with their disposition. The script report [digest_script_report.md](digest_script_report.md) documents item F-07.

Statuses: **RESOLVED** (done; the note names the test, file or commit that pins it), **PARTIAL** (part done; the note says what remains), **TODO** (open), **REJECTED** (measured or decided and declined, with the reason), **NOT A DEFECT** (investigated, behaves correctly), **DOC** (documented), **FUTURE** (deferred with a trigger).

## Summary

| File | Findings | RESOLVED | PARTIAL | TODO | REJECTED | NOT A DEFECT | DOC | FUTURE |
|---|---|---|---|---|---|---|---|---|
| `01_followup_proposals.md` | 13 | 10 | 1 | 0 | 0 | 2 | 0 | 0 |
| **Total** | **13** | **10** | **1** | **0** | **0** | **2** | **0** | **0** |

## Per-report status

| Status | Report | Findings | Area |
|---|---|---|---|
| **OPEN** | [01_followup_proposals.md](01_followup_proposals.md) | 13 | CI, test harness and tooling follow-ups |

### `01_followup_proposals.md`

| Status | Sev | ID | Finding | Evidence / what remains |
|---|---|---|---|---|
| **RESOLVED** | P3 | `F-01` | tests/feature_selection/shap_proxied/test_shap_proxy_treeshap_interactions_gpu.py:167 | RESOLVED - the GPU-versus-numba ratio uses assert_paired_speedup (3 interleaved rounds) and skip_if_host_contended; 8 passed, 1 skipped while the host is busy; the 1.15x target was not re-measured on a quiet host |
| **RESOLVED** | P2 | `F-02` | tests/training/fuzz/test_fuzz_3way_suite.py:148, test_fuzz_hypothesis.py:129, test_fuzz_regression_sensors.py:75 | RESOLVED - tests/test_meta/test_fuzz_extractor_passes_target_type.py scans every extractor construction under tests/training/fuzz for target_type; one intentional flag-only construction is allowlisted with its reason |
| **RESOLVED** | P3 | `F-03` | scripts/sync_and_push.sh | RESOLVED - fetch, merge, commit (retrying the line-ending hook), push loop with exit 3 on a conflict; tests/scripts/test_sync_and_push_script.py runs it in throwaway repositories; documented in CLAUDE.md; used for this round's py-ci-shared push |
| **RESOLVED** | P2 | `F-04` | .github/workflows/polars-matrix.yml | RESOLVED - daily matrix over polars 1.36.1, 1.41.2, 2.0.0 and latest running the polars-facing tests (802 selected locally); workflow meta tests pass; the first scheduled run has not happened yet |
| **PARTIAL** | P2 | `F-05` | .github/workflows/ci.yml (lint-advisory call) | PARTIAL - py-ci-shared 9d0cc27 adds the pip-audit-ignore-vulns input, but the released v1.22.3 (e3c24ab) does not contain it, so mlframe pins v1.22.3 and the three accepted ids (PYSEC-2025-194, PYSEC-2026-139, PYSEC-2026-3447) show in the report again; remains: a py-ci-shared release containing 9d0cc27, then re-add the input to ci.yml |
| **NOT A DEFECT** | P3 | `F-06` | .github/workflows/ci.yml:3-15 | NOT A DEFECT - paths-ignore for **.md, docs/**, audits/** and LICENSE is already on the push trigger, and mypy-full, black-filtered, codeql and hooks-not-in-ci have it too |
| **RESOLVED** | P3 | `F-07` | scripts/ci_failure_digest.py | RESOLVED - per-workflow failed-job digest with NEW/PERSISTING/FIXED, retrying gh helper and 15 tests; a live run parsed real sklearn-matrix logs correctly; FIXED compares only the last two runs |
| **NOT A DEFECT** | P2 | `F-08` | tests/feature_selection/mrmr/fe/test_fe_fusion_scoring_subsample.py (cascade victim) | NOT A DEFECT - compute-sanitizer memcheck 0 errors over the two nearest files and the 22-file window clean under CUDA_LAUNCH_BLOCKING=1; the card is shared at about 88% occupancy; re-run memcheck over the whole window if it recurs on a quiet card |
| **RESOLVED** | P2 | `F-09` | tests/test_meta/test_workflow_polars_install_keeps_runtime_matched.py | RESOLVED - the scan has a teeth test (flags the broken form, accepts the paired and plain forms) and passes over the repository; polars-matrix.yml now uninstalls both packages and installs a matching pair, numpy capped below 2.5 for a catboost import failure. |
| **RESOLVED** | P3 | `F-10` | scripts/sync_and_push.sh | RESOLVED - rejection and lock errors retry, any other failure prints up to 40 finding lines and exits 4; test_a_failing_pre_push_hook_stops_with_its_findings_instead_of_retrying pins it. |
| **RESOLVED** | P3 | `F-11` | .github/workflows/ci-failure-digest.yml | RESOLVED - runs daily at 06:47 UTC and on dispatch, writes the digest to the job summary and an artifact; exempt from the push-trigger rule by name with the reason; not yet run on the remote. |
| **RESOLVED** | P2 | `F-12` | tests/test_meta/test_py_ci_shared_pin.py | RESOLVED - test_the_pin_is_a_released_tag_not_an_unreleased_commit with an empty allowlist plus a teeth test for the marker regex. The repository is currently on a plain release pin. |
| **RESOLVED** | P3 | `F-13` | .gitignore | RESOLVED - audits/**/*.log and /sc_*.log are ignored. |
