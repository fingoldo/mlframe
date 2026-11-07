# Wave 5 security audit: SECRETS AND CREDENTIALS

Date: 2026-11-07. Scope: `src/mlframe`, `.github/workflows`. Read-only. Prior rounds: `audits/2026-10-31/secrets.md` (SEC-01..05), `audits/2026-11-06/wave4/secrets.md` (0 findings).

## (1) Findings

| ID | Severity | file:line | Failure scenario | Proposed disposition |
|---|---|---|---|---|
| (none) | - | - | No new concrete secret-exposure path on HEAD. | - |

Counts: High 0, Medium 0, Low 0, Info 0.

Re-verification on HEAD (line numbers shifted since wave 4; rulings hold):

- SEC-01 RESOLVED holds: `src/mlframe/integrations/mlflow.py:171-173` logs `_strip_userinfo(e)` with `exc_info=_scrubbed_exc_info(e)` (`:48-53`, rebuilt scrubbed RuntimeError).
- SEC-02 RESOLVED holds: `.github/workflows/pandas-matrix.yml:165-181` sends the token via `git -c http.extraheader`; `:166` registers the base64 value with `::add-mask::` (masking, not exposure). `polars-matrix.yml` has the same 5 hits.
- SEC-03/04 RESOLVED, SEC-05 DOC: unchanged in `providers.py`.

## (2) Checked and safe

- `training/feature_handling/apply.py:483,487,574,583`, `training/core/_phase_train_one_target_dataset_cache.py:165`: `df_token`/`train_idx_token` are cache-key content tokens, not credentials.
- `nosec B311` comments (`utils/_param_oracle.py:521`, `composite/hpo.py:741`, `wrappers/_helpers.py:226`, `_benchmarks/bench_eng_base_set_cache.py:10-14`): non-crypto RNG, no secret generation.
- `.github/workflows/release.yml:4`: PyPI Trusted Publishing (OIDC), no stored token.
- `# pragma: allowlist secret` lines in `ci.yml`, `docs.yml`, `black-filtered.yml`, `mypy-full.yml`, `fs-benchmark-nightly.yml`, `macos-abort-probe.yml`: pinned commit SHAs, not credentials.
- `ci.yml:421`: `CODECOV_TOKEN` passed via action `with: token`, not echoed.
- `hf_provider.py`, `_neural_mi.py`: as ruled in wave 4 (resolved params local only; revision SHAs).
