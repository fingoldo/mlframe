# Supply chain and CI security audit, round 2 (mlframe)

Read-only review on 2026-11-01. No file was edited; no git state was changed.

Baseline: `audits/2026-10-31/supply_chain.md` (SUP-01..SUP-15) and `audits/2026-10-31/_TRACKER.md`. Items already recorded there are not re-reported. This pass covers what that round did not: the pre-commit hook revs, the hook dependency pins against `uv.lock`, the release workflow's publish chain, unpinned `uvx` and `setup-uv` use, vendored code provenance, and the CODECOV_TOKEN exposure on `pull_request` jobs.

## Summary

- 8 findings: 1 Medium, 3 Low, 4 Info.
- Nothing here is exploitable by an outside fork PR. The one Medium needs an upstream compromise of a PyPI tool, and it reaches PyPI only on a release event.
- Re-verified SUP-02 from the prior round: `https://pypi.org/pypi/pyutilz/json` still returns 404 today, so the release gate at `.github/workflows/release.yml:31-37` still blocks upload. The prior disposition (FUTURE) stands.

## Findings table

| ID | File:line | Severity | Proposed disposition |
|---|---|---|---|
| SUP2-1 | .github/workflows/release.yml:70; .github/workflows/ci.yml:754 | Medium | FUTURE |
| SUP2-2 | .pre-commit-config.yaml:56,86,126,175,186,200,221,267,459 | Low | FUTURE |
| SUP2-3 | .pre-commit-config.yaml:291; uv.lock:8827-8939, 11423 | Low | FUTURE |
| SUP2-4 | src/mlframe/feature_selection/filters/_vendored/infonet/NOTICE.md | Low | DOC |
| SUP2-5 | .github/workflows/ci.yml:421 | Info | RESOLVED |
| SUP2-6 | .github/workflows/ci.yml:241,494,746,959; release.yml:43; dep-floors.yml:73,116; deep-nightly.yml:81,171; fs-benchmark-nightly.yml:75; macos-abort-probe.yml:183 | Info | DOC |
| SUP2-7 | .github/workflows/release.yml:31-37 (PyPI gate, re-verified) | Info | RESOLVED |
| SUP2-8 | .github/workflows/ci.yml:421 and other `fingoldo/py-ci-shared/.github/actions/*` call sites | Info | DOC |

## Details

### SUP2-1 (Medium, FUTURE): unpinned `uvx twine` runs in the release build job, which holds `id-token: write`

- Where: `.github/workflows/release.yml:70` (`run: uvx twine check dist/*`), in job `build`. That job grants `id-token: write` and `attestations: write` (release.yml:25-27). The same unpinned call is at `.github/workflows/ci.yml:754`.
- Chain: `uvx twine` resolves the latest `twine` from PyPI on each run (`twine` is not in the L1.1 pin list at ci.yml:655-670). Its output is `dist/*`, which the same job then uploads as the artifact (release.yml:84-90 area, `upload-artifact`). The `publish` job downloads that artifact and runs `pypa/gh-action-pypi-publish` with `id-token: write` (release.yml:99-100).
- Failure scenario: a compromised `twine` release runs in the build job. It rewrites or adds files in `dist/` before the upload step. The tampered artifact is then published to PyPI through Trusted Publishing. The attestation step at release.yml:76-82 hashes whatever `dist/` holds at that point, so it does not detect the change.
- Exploitable in practice: needs a compromised PyPI release of `twine`, and only on a `release` event (publish is gated at release.yml:94-96). Not reachable from a fork PR.
- Proposed fix: pin `uvx twine==<exact>` (the same way ci.yml:655-670 pins the other tools), or install it from a hashed requirements file. Better still, run the check in a separate job that has no `id-token` and no artifact-write path.
- Disposition: FUTURE. The fix is one line but it touches release tooling, so the owner should choose the pin and confirm the release path.

### SUP2-2 (Low, FUTURE): pre-commit hook revs are mutable tags

- Where: `.pre-commit-config.yaml` lines 56 (`pre-commit/pre-commit-hooks` `v6.0.0`), 86 (`Yelp/detect-secrets` `v1.5.0`), 126, 200, 459 (`astral-sh/ruff-pre-commit` `v0.16.10`), 175 (`actionlint-py` `v1.7.12.24`), 186 (`shellcheck-py` `v0.11.0.1`), 221 (`psf/black` `26.5.1`), 267 (`pre-commit/mirrors-mypy` `v2.1.0`).
- Contrast: `py-ci-shared` at line 671 is pinned to a full SHA. The prior round's SUP-10 applied that reasoning to `requirements-dev.txt`, but not to the hook revs.
- Failure scenario: a tag is force-moved to a different commit upstream. The next `pre-commit run` for a developer, or a CI job that runs hooks, fetches the new code. Hook code runs locally with the developer's file access.
- Exploitable in practice: needs an upstream tag repoint on one of these repos. Hooks run on developer machines, not on the release path. Low.
- Proposed fix: pin each `rev:` to a full commit SHA with a version comment, as done for `py-ci-shared`. `pre-commit autoupdate --freeze` can produce these.
- Disposition: FUTURE (cheap, but touches every hook; defer to a single dedicated change).

### SUP2-3 (Low, FUTURE): mypy hook dependency pins do not match `uv.lock`

- Where: `.pre-commit-config.yaml:291`: `additional_dependencies: ['numpy==2.3.5', 'scipy==1.17.1', 'scikit-learn==1.8.0', 'pydantic==2.13.4']`. The comment directly above (lines 283-290) says "Versions below match the project's real dev env."
- Evidence: `uv.lock` has numpy 1.26.4, 2.2.6 and 2.4.6 (lines 8827, 8874, 8939). It has no 2.3.5. It has pydantic 2.13.5 (line 11423), not 2.13.4. scipy 1.17.1 and scikit-learn 1.8.0 are present.
- Failure scenario: the beachhead mypy gate checks code against stubs for numpy 2.3.5 and pydantic 2.13.4. The CI and dev environment use the lock versions. A stub or typing difference between those versions can pass locally and fail in CI, or the reverse. The comment's stated purpose (avoid drift) is not met.
- Exploitable in practice: no. This is a drift and reproducibility defect, not an attack path.
- Proposed fix: derive these pins from `uv.lock` (or from `requirements`) so the hook and the lock cannot disagree, and add a test that compares them.
- Disposition: FUTURE. The fix needs a decision on which environment is authoritative, and it may change which mypy errors appear.

### SUP2-4 (Low, DOC): vendored InfoNet code has provenance but no license

- Where: `src/mlframe/feature_selection/filters/_vendored/infonet/` (the `infer.py` and `model/*.py` files). `NOTICE.md` records the upstream URL (`datou30/InfoNet`), the vendored commit `3808a6866a43b6627f74b4f5fa217c8602963169`, and the verification date. It also records that upstream has no LICENSE.
- Provenance is recorded, so it is not an unknown-origin risk. The residual risk is licensing: the code is redistributed inside the mlframe wheel without any terms granted.
- Not checked: that the vendored files match commit `3808a686` byte for byte. That needs a GET to GitHub, which I did not run in this pass.
- Proposed disposition: DOC. `NOTICE.md` already tells redistributors to verify the license or exclude the directory. Keep it, and link it from the README or the release notes.

### SUP2-5 (Info, RESOLVED): CODECOV_TOKEN on a `pull_request` job

- Where: `.github/workflows/ci.yml:421` (`token: ${{ secrets.CODECOV_TOKEN }}`). The job runs on `pull_request` (ci.yml:16).
- Why safe: GitHub does not pass repository secrets to workflows triggered by a fork's `pull_request`, so the token is empty there. Same-repo and dependabot PRs also do not receive it. The step is further gated to one matrix leg (ci.yml:415-419).
- The `pull_request` jobs carry only the read-only `permissions:` set at ci.yml:37, which the prior round already recorded as SUP-12.

### SUP2-6 (Info, DOC): `setup-uv` is used without a `version:` input

- Where: ten `astral-sh/setup-uv@1c37ad07...` call sites, for example `ci.yml:241`, `release.yml:43`, `dep-floors.yml:73`. None sets `version:`, so each run installs the current `uv` binary.
- The action itself is SHA-pinned (SUP-11 in the prior round). The `uv` binary it downloads is not.
- Risk: a compromised `uv` release, or a `uv` change in how it reads `uv.lock`, reaches every job. The lock format is `revision = 3`, so a breaking change would show up as a lock diff or a CI failure, not silently.
- Proposed disposition: DOC. Pinning `version:` to the tested release is the stronger option and is cheap; the owner can choose it at the next uv upgrade.

### SUP2-7 (Info, RESOLVED): PyPI gate for the pyutilz runtime dependency (re-verified)

- Where: `.github/workflows/release.yml:31-37` asserts `https://pypi.org/pypi/pyutilz/json` returns 200 before upload. The gate is set on `release` events only.
- Re-verified on 2026-11-01: that URL returns 404. The gate therefore fails any release today, which blocks publishing an uninstallable wheel. This confirms the prior round's SUP-02 status and does not change it.
- Disposition: RESOLVED for the release-gate question. SUP-02 itself stays FUTURE (see the prior round).

### SUP2-8 (Info, DOC): reusable actions in `fingoldo/py-ci-shared` receive secrets

- Where: `.github/actions/upload-codecov` (`ci.yml:421`, `deep-nightly.yml:250-252`, `numba-coverage.yml:224`) and the reusable workflows at the pinned SHA `670b80cb...`. These receive `CODECOV_TOKEN` or run with `contents: write` (ci.yml:553-559 calls out to shared code).
- The SHA pin (prior SUP-10, SUP-11) makes the code immutable. Its contents were not reviewed here, because `py-ci-shared` is a separate repository and was out of scope, as the prior round also noted.
- Proposed disposition: DOC. Record in the README or CONTRIBUTING that `py-ci-shared` is a trust boundary, and that its SHA bump needs a review of the diff.

## Checks not performed from this environment

- Live state of the `py-ci-shared` composite and reusable workflows at SHA `670b80cb...` (separate repository; out of scope).
- Live state of branch protection and required checks (needed for prior SUP-03; still a maintainer check).
- Byte-level comparison of the vendored InfoNet files to upstream commit `3808a686` (needs a GitHub GET; not run).
- Whether the `pull_request` secret behaviour in SUP2-5 holds for this repo's visibility and settings (assumed public, the worst case).
- Current PyPI state of `twine` and the exact version it resolves to on each run (SUP2-1). The release path was not exercised.
- Whether `uv.lock` matches `pyproject.toml` after the most recent edit (no `uv lock --check` was run; it writes nothing, but the tool was not invoked in this read-only pass).
