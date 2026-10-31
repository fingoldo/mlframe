# Supply chain and CI security audit (mlframe)

Read-only review. No code or git state was changed.

Scope: pyproject.toml, uv.lock (header, source entries, registry hosts), requirements-dev.txt, .github/dependabot.yml, all .github/workflows/*.yml (triggers, permissions, secrets, `run:` interpolation, `uses:` refs, `pip`/`curl` installs).

## Summary

- 15 findings: 3 Medium, 6 Low, 6 Info.
- The workflows are in good shape. Every `uses:` ref is a full 40-char SHA, no workflow uses `pull_request_target`, the one self-hosted GPU runner is reachable only by `workflow_dispatch`, and write-scoped jobs are gated to push on master.
- The real exposure is dependency intake: a floating first-party git source (`pyutilz`) that pip users resolve from a name that is not on PyPI (dependency-confusion surface), and a dependabot auto-merge with no human review.
- Exploitability is stated per finding. Nothing here is exploitable by an outside fork PR today.

## Findings table

| ID | File:line | Severity | Proposed status |
|---|---|---|---|
| SUP-01 | pyproject.toml:176 | Medium | DOC |
| SUP-02 | pyproject.toml:154 | Medium | FUTURE |
| SUP-03 | .github/workflows/dependabot-auto-merge.yml:36-43 | Medium | FUTURE |
| SUP-04 | .github/workflows/macos-abort-probe.yml:173-175 | Low | FUTURE |
| SUP-05 | .github/workflows/docs.yml:45-46; codecov-full.yml:75; dep-floors.yml:89 | Low | FUTURE |
| SUP-06 | pyproject.toml:64-65, 88, 79 (open-ended runtime floors) | Low | DOC |
| SUP-07 | .github/workflows/pandas-matrix.yml:94; polars-matrix.yml:97; sklearn-matrix-ci.yml:140; numba-coverage.yml:141; gpu-matrix.yml:60 | Low | DOC |
| SUP-08 | .github/workflows/fs-benchmark-nightly.yml:89,98; gpu-matrix.yml:38 | Low | REJECTED |
| SUP-09 | .github/workflows/gpu-matrix.yml:67-68 | Low | REJECTED |
| SUP-10 | requirements-dev.txt:65 | Info | RESOLVED |
| SUP-11 | .github/workflows/*.yml (all `uses:` lines) | Info | RESOLVED |
| SUP-12 | .github/workflows/dependabot-auto-merge.yml:26; gpu-matrix.yml:16-20; ci.yml:16; ci.yml:215,487 | Info | RESOLVED |
| SUP-13 | .github/workflows/ci.yml:548-559; numba-coverage.yml:326-330; codecov-full.yml:56-62 | Info | RESOLVED |
| SUP-14 | .github/workflows/codecov-full.yml:36, 56-62 | Info | RESOLVED |
| SUP-15 | uv.lock:12331, uv.lock (1797 registry refs) | Info | RESOLVED |

## Details

### SUP-01 (Medium, DOC): pyutilz resolves from a floating branch

- Where: `pyproject.toml:176` `pyutilz = { git = "https://github.com/fingoldo/pyutilz.git", branch = "master" }`.
- Risk: anyone who runs `uv lock` or `uv lock --upgrade` without the lock pulls whatever master is. A same-day upstream commit can change shared behaviour. The project's own comment (pyproject.toml:161-172) records this exact incident.
- Exploitable in practice: only by a maintainer or an automated re-lock, or by an attacker who controls the upstream repo. The committed `uv.lock:12331` pins `#38532cd5...`, so a normal `uv sync` or CI run is reproducible. Risk is limited to lock regeneration.
- Disposition DOC: the branch is an intentional design choice, documented in pyproject.toml:161-172. Keep the comment and the lock as the authority. The CI copies of the pin (ci.yml, dep-floors.yml, docs.yml) must stay in step with `uv.lock`.

### SUP-02 (Medium, FUTURE): `pyutilz>=1.0.0` is a non-PyPI name in the runtime dependencies

- Where: `pyproject.toml:154` lists `pyutilz>=1.0.0` under `[project].dependencies`. The comment at pyproject.toml:146 tells plain-pip users to install `pyutilz @ git+...@38532cd...` first.
- Risk: pip resolves `pyutilz` against PyPI. The project says the name is not published there. If anyone registers `pyutilz` on PyPI, `pip install mlframe` installs that package (dependency confusion). Nothing in the metadata pins the source.
- Exploitable in practice: yes, for pip users who skip the manual pre-install step. Requires an attacker to register the name. It is not exploitable by a PR. I could not verify the current PyPI state from this read-only environment.
- Fix: publish pyutilz to PyPI under the same name so the name is owned by the project, or keep it out of the install path that pip users hit and fail loudly on resolution. Add a PyPI-side name reservation.
- Disposition FUTURE: real, but fixing it means a release of pyutilz, which is outside this repo.

### SUP-03 (Medium, FUTURE): dependabot PRs auto-merge with no human review

- Where: `.github/workflows/dependabot-auto-merge.yml:36-43`. Every PR from `dependabot[bot]` (any ecosystem, including major runtime bumps and `uv.lock` changes) gets `gh pr merge --auto --squash`.
- Mitigations already present: `.github/dependabot.yml` sets a 7-day cooldown on the uv and github-actions ecosystems. The workflow runs on `pull_request`, so the token is limited.
- Risk: a malicious or compromised upstream release that stays unflagged for 7 days and passes CI merges to master unreviewed. CI runs the test suite, which is not a security review.
- Exploitable in practice: needs an upstream compromise that survives the cooldown and CI. Not exploitable by a fork PR, since dependabot PRs are same-repo branches.
- Fix: restrict auto-merge to semver-patch and minor updates for `uv`, and require a human approval for major bumps or for any change to `uv.lock` that adds a new package. Also add dependency-review on those PRs.
- Disposition FUTURE: a policy change for the maintainer to decide.

### SUP-04 (Low, FUTURE): Homebrew formula fetched by curl with no checksum

- Where: `.github/workflows/macos-abort-probe.yml:173-175` downloads `Formula/lib/libomp.rb` at commit `ddbdc91ebe` from raw.githubusercontent.com and installs it.
- Risk: no SHA-256 check on the downloaded formula. A tampered response would be installed on the macOS runner.
- Exploitable in practice: only if GitHub's raw content host or the Homebrew repo is compromised. The short commit SHA is content-addressed, so the risk is low.
- Fix: add `echo "<sha256>  file" | shasum -a 256 -c -` after the curl, or fetch via `git clone` pinned to the full commit.
- Disposition FUTURE: cheap hardening, deferred.

### SUP-05 (Low, FUTURE): unpinned `pip install` of tooling in CI

- Where: `.github/workflows/docs.yml:45-46` (`mkdocs-material>=9.5`, `mkdocs-jupyter>=0.24`); `codecov-full.yml:75` (`coverage[toml]>=7.0`); `dep-floors.yml:89` (`pip-audit>=2.7`).
- Risk: these resolve to the latest release on every run. A compromised release of any of them runs in a job that can see a `pages: write` / `id-token: write` token (docs.yml:28-31, same-repo PRs and push only).
- Exploitable in practice: needs a compromised upstream package; fork PRs do not receive `id-token` for pull_request events.
- Fix: pin exact versions with hashes (`--require-hashes`), or install from `uv.lock`.
- Disposition FUTURE: real, low impact.

### SUP-06 (Low, DOC): open-ended runtime floors

- Where: `pyproject.toml:64-65` (`polars>=0.20`, `scipy>=1.10`), `pyproject.toml:88` (`pyarrow>=25.0.0`), and most other entries in `[project].dependencies`.
- Risk: pip users get whatever major is current. Breaking upstream majors can land.
- Exploitable in practice: this is a breakage risk, not an attack; the lock covers all CI and uv installs. Upper bounds on the runtime deps are already present for pandas (`<3.0`), numpy and scikit-learn.
- Disposition DOC: standard library convention; the `uv.lock` file is the reproducibility mechanism.

### SUP-07 (Low, DOC): matrix jobs install floating dependencies

- Where: `pandas-matrix.yml:94`, `polars-matrix.yml:97`, `sklearn-matrix-ci.yml:140`, `numba-coverage.yml:141`, `gpu-matrix.yml:60` run `pip install -e ".[...]"` without the lock.
- Risk: results drift as upstream releases.
- Exploitable in practice: no; these jobs test upstream compatibility on purpose (version matrices).
- Disposition DOC: the matrix intentionally tests against floating upstream. `ci.yml` uses the lock.

### SUP-08 (Low, REJECTED): `${{ inputs.* }}` interpolated into run blocks

- Where: `fs-benchmark-nightly.yml:89` and `:98` (`--tier "${{ inputs.tier || 'nightly' }}"`); `gpu-matrix.yml:38` in a job `name:` only.
- Risk: classic template injection.
- Why rejected: `inputs.tier` is `type: choice` with options `[nightly, weekly]` (fs-benchmark-nightly.yml:28-31). GitHub enforces the choice list on `workflow_dispatch`, and only users with write access can dispatch. No untrusted input reaches the interpolation.

### SUP-09 (Low, REJECTED): free-form `torch_spec` input

- Where: `gpu-matrix.yml:65-68`. `torch_spec` is a free string, passed through `env: TORCH_SPEC` and used as `pip install "torch==${TORCH_SPEC}"`.
- Why rejected: the value goes through `env:` and is expanded inside double quotes, so it stays one argv element. Only write-access users can dispatch, and that user can already run arbitrary code by editing a workflow on a branch. Worst case is a broken install on their own run. Optional hardening: validate with a regex.

### SUP-10 (Info, RESOLVED): py-ci-shared pinned to a full SHA

- `requirements-dev.txt:65` pins `py-ci-shared` to `670b80cbfd4b851578346741c21b62b8aa0ce4dc`. The file comments explain the choice (a tag can be repointed). The same SHA is used across all reusable-workflow `uses:` lines.

### SUP-11 (Info, RESOLVED): all `uses:` refs are full commit SHAs

- Checked every `uses:` line under `.github/workflows` (including reusable workflows and `pypa/gh-action-pypi-publish@dc37677b...` at release.yml:109). Each has a 40-char SHA with a version comment. No `@v4`, `@main` or tag-only refs.

### SUP-12 (Info, RESOLVED): no `pull_request_target`, no fork-reachable self-hosted runner

- No `pull_request_target` in any workflow. `dependabot-auto-merge.yml:21-23` documents why it is avoided.
- The self-hosted GPU runner (`gpu-matrix.yml:39`, `runs-on: [self-hosted, gpu]`) is reachable only by `workflow_dispatch` (gpu-matrix.yml:16-20). `gpu-extras-install-matrix.yml` uses an ubuntu-24.04 runner.
- `ci.yml:16` runs `pull_request` on fork code. It runs with no secrets, and test jobs set `persist-credentials: false` (ci.yml:215, 487).

### SUP-13 (Info, RESOLVED): write-scoped jobs are gated to push on master

- `ci.yml:548-559` (`contents: write`, `persist-credentials: true`) is gated by `if: always() && github.event_name == 'push' && github.ref == 'refs/heads/master'`.
- `numba-coverage.yml:326-330` uses the same gate on `workflow_run` with `event == 'push'`.

### SUP-14 (Info, RESOLVED): `workflow_run` with CODECOV_TOKEN is gated

- `codecov-full.yml:36` triggers on `workflow_run`, which normally exposes secrets to forks. The job-level `if` at codecov-full.yml:56-62 accepts only a successful upstream run whose event is `push`. `numba-coverage.yml:83-88` has a matching gate with a comment about this exact risk.

### SUP-15 (Info, RESOLVED): lockfile sources are PyPI plus one commit-pinned git source

- `uv.lock` has 1797 `registry = "https://pypi.org/simple"` references and no other registry host. The only git source is `uv.lock:12331`, pinned to `#38532cd5...`. The one editable source is the project itself (uv.lock:7298).

## Notes

- Repo visibility was not checked, per instructions. The `pull_request` triggers and the fork-PR analysis above assume a public repo, which is the worst case.
- Not verified: the live PyPI state for `pyutilz` (SUP-02), and whether branch protection requires the checks dependabot auto-merge depends on (SUP-03). Both need a maintainer to confirm.
- Not verified job-by-job in the reusable workflow `py-ci-shared` at its pinned SHA; that repo was out of scope.
