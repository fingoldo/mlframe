# Wave 4 security audit: SECRETS AND CREDENTIALS

Date: 2026-11-06. Scope: `src/mlframe`, `.github/workflows`. Read-only. Prior round: `audits/2026-10-31/secrets.md` (SEC-01..05).

## (1) Findings

| ID | Severity | file:line | Failure scenario | Proposed disposition |
|---|---|---|---|---|
| (none) | - | - | No new concrete secret-exposure path found on HEAD. | - |

Counts: High 0, Medium 0, Low 0, Info 0.

Re-verification of prior dispositions on HEAD (all hold):

- SEC-01 RESOLVED holds: `src/mlframe/integrations/mlflow.py:166-169` logs `_strip_userinfo(e)` and passes `exc_info=_scrubbed_exc_info(e)` (`:47-57`), which rebuilds the message scrubbed and drops `__cause__`/`__context__`. Non-retryable path re-raises a scrubbed exception `from None` (`:197`).
- SEC-02 RESOLVED holds: `.github/workflows/pandas-matrix.yml:165-181` and `polars-matrix.yml:169-185` pass the token via `git -c http.extraheader`, not in the remote URL.
- SEC-03/SEC-04 RESOLVED hold: `src/mlframe/training/feature_handling/providers.py:114-118` scrubs URL userinfo in values via `_URL_USERINFO_RE`.
- SEC-05 DOC unchanged.

## (2) Checked and safe

- `src/mlframe/training/feature_handling/hf_provider.py:225`: `resolve_secrets().params` is a local; only non-secret fields (model, revision, device, dtype) are logged (`:245-248`); resolved params are not stored on `self` (only `_cfg`, the unresolved `env:` form, `:183`), so pickled providers carry no resolved secret.
- `hf_provider.py:67-69`, `feature_selection/filters/_neural_mi.py:465-466`: commit SHAs (revision pins), not credentials; allowlisted.
- `integrations/mlflow.py:133-155`: DSL filter built from run name/parent id only; no URI logged.
- `os.environ[...]` writes across `src/` (CUDA, thread counts, TQDM, PYTHONHASHSEED, feature flags): no credential values.
- Workflows: `CODECOV_TOKEN` passed via action `with: token` (`ci.yml:421`, `codecov-full.yml:245`, `deep-nightly.yml:251`, `numba-coverage.yml:224`); `GH_TOKEN`/`github-token` via `env:`/`with:` only (`ci-failure-digest.yml:38`, `codecov-full.yml:80,124,131,177,204,214`, `dependabot-auto-merge.yml:42`); no `echo`/`set -x` of these values found. `hooks-not-in-ci.yml:67-68` runs detect-secrets.
- No hard-coded API keys, bearer tokens, `?token=` URLs, or `requests`/`urlopen` calls with credentials in `src/mlframe`.
