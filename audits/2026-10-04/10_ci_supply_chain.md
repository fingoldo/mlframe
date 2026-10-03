# 10 CI / supply chain / build / portability (read-only audit, 2026-10-04)

Scope: .github/workflows/*.yml (18), pyproject.toml, uv.lock, requirements-dev.txt, .gitattributes, MANIFEST.in, release path.
Method: grep/read of every `uses:`, trigger, permissions, concurrency, install step; pin consistency by counting SHAs; one read-only `gh run list`.
Not run: pip-audit/uv (vulnerabilities UNVERIFIED), license scan (UNVERIFIED), wheel build (read the smoke-test assertions instead).
Dedup: audits/ci_review_2026-09-08/_TRACKER.md consulted (it records concurrency history and macOS crash work; neither finding C1 nor C2 below is in it as open).

Checkpoint 1: actions pinning, triggers, permissions done. Checkpoint 2: pins, installs, release done. Checkpoint 3: report written.

## Verified clean (no finding)
- All third-party `uses:` are 40-char SHA pinned with version comments (22 checkout, 17 setup-python, etc.); only the pypi publish action carries `# release/v1` as comment, still SHA.
- No `pull_request_target`; `workflow_run` only in codecov-full.yml, gated by conclusion and event (lines 36, 60-62). dependabot-auto-merge uses `pull_request` and env-passed PR_URL.
- Top-level `permissions: contents: read` on workflows checked; release.yml uses OIDC trusted publishing, publish gated `if: github.event_name == 'release'`, environment `pypi`, tag==version assert, provenance attestation.
- pyutilz pin 635d0c5a... identical in 17 places (ci, deep-nightly, dep-floors, fs-benchmark, smoke tests, pyproject comment); py-ci-shared 9779e9f9... identical in workflows, requirements-dev.txt:61 and pre-commit rev.
- Wheel: package discovery excludes tests/legacy/benchmarks/scripts/docs (pyproject.toml:577), py.typed in package-data (581), version single-sourced via attr (571-572), smoke job asserts shipped package-data.
- pywt is imported lazily (wavelet_dwt.py:66, _timeseries_emit.py:133), so `import mlframe` no longer needs the signal extra.

## Findings
| ID | file:line | sev | evidence | impact | proposed fix | Disposition |
|---|---|---|---|---|---|---|
| C1 | .github/workflows/ci.yml:40 | P1 | `cancel-in-progress: true`; `gh run list --branch master --workflow CI --limit 15`: 14 of 14 completed runs `cancelled`, the newest one empty (in progress). | CI has not completed on master in the last 15 runs, so no commit gets a green signal or a codecov upload; the comment at lines 32-39 shows the non-cancel alternative was tried and reverted for queue backlog. | Keep cancel for PRs, but on master use per-run groups for scheduled completeness: e.g. group `ci-${{github.ref}}-${{ github.run_number / N }}` bucket, or a daily scheduled full CI run on master (`schedule:` cron, own non-cancelling group) so at least one full run per day finishes. | OPEN |
| C2 | .github/workflows/fs-benchmark-nightly.yml:83 | P2 | `project-extras: ".[dev]"` while pywavelets lives only in `signal` extra (pyproject.toml:389-394); `all` includes signal but this workflow does not use `all`. | Any benchmark arm that reaches wavelet feature code raises ModuleNotFoundError pywt (same class as the earlier nightly failure). Whether run_experiment reaches it: UNVERIFIED. | Install `.[dev,signal]` (or `.[all,dev]` like deep-nightly), or add a pre-run import probe step. | OPEN |
| C3 | .github/workflows/dep-floors.yml:124 | P3 | `uv pip install --system --resolution=lowest-direct -e ".[dev]"`, `continue-on-error: true` (line ~103), tests tests/metrics + calibration. | Floors of every optional extra (signal, boosting...) never verified; job cannot fail the build anyway. | Acceptable as advisory; add extras only if floors of those are meant to be a contract. | OPEN |
| C4 | pyproject.toml:175 vs workflows | P3 | `pyutilz = { git = ..., branch = "master" }` in [tool.uv.sources]; `uv sync --frozen` used at ci.yml:972 while every other path pins SHA 635d0c5a. | Lock job resolves pyutilz at the locked master commit, other jobs at 635d0c5; two different pyutilz versions across jobs possible. Locked commit in uv.lock not read (UNVERIFIED). | Assert in a meta-test that the uv.lock pyutilz commit equals the workflow SHA, or bump both together. | OPEN |
| C5 | 58 sites in src, e.g. src/mlframe/feature_selection/wrappers/rfecv/_cb_border_cache.py:113,126; filters/_vendored/infonet/infer.py:17 | P3 | `with open(path, "w") as fh:` / `open(config_path, "r")` with no encoding (grep count 58 incl. benchmarks). | Windows cp1252 default vs Linux UTF-8: non-ASCII content round-trips differently; runtime cache path is the relevant one. | `encoding="utf-8"` on runtime (non-_benchmarks) sites; ruff PLW1514 ratchet. | OPEN |
| C6 | .github/workflows/gpu-matrix.yml:39 | P3 | `runs-on: [self-hosted, gpu]`. | Self-hosted runner executing repo code; safe only if trigger is dispatch-only (not verified beyond `github.event.inputs` use at 38/57/63, which are passed via env, not interpolated into run). | Confirm triggers are `workflow_dispatch` only and repo setting forbids fork PR on self-hosted. | OPEN |
| C7 | .github/workflows/ci.yml:277-288 | P3 | `actions/cache` with `restore-keys: numba-<os>-py<ver>-` and key including hashFiles('src/**/*.py'). | Cache is shared by branch scope per GitHub rules (PR caches cannot poison master); numba cache is bytecode-ish JIT output. Low risk. | None needed beyond keeping save only on master. | OPEN |
| C8 | pyproject.toml (coverage section) | P3 | grep for `fail_under` in pyproject.toml, codecov.yml, workflows returned nothing (only interrogate at 768, ci.yml:648). | Contradicts the memory note that a coverage fail_under gates runs; either it lives in a file I did not search (.coveragerc absent) or no gate exists. UNVERIFIED. | Owner to confirm where the gate is defined. | OPEN |
| C9 | .github/workflows/release.yml:12-13 | P3 | NOTE says upload requires pyutilz on PyPI; pyproject depends on `pyutilz>=1.0.0` (line 153). | A published wheel would be uninstallable via plain pip until pyutilz is on PyPI. Publish job is release-event gated, so risk is on a deliberate release only. | Block release until pyutilz is on PyPI, or check in build job. | OPEN |

Counts: P1 1, P2 1, P3 7.

## Verdict
Supply-chain hygiene is unusually strong (SHA pins, consistent pins, OIDC publish, minimal permissions). The one real problem is C1: master CI is effectively never completing. Not worth fixing: C7 (cache scoping already safe), C3 (advisory by design), Python matrix/classifier mismatch (3.9-3.14 consistent in pyproject; not an issue found). Not verified: vulnerable dependencies, licenses, macOS path-style staleness fix, shard durations/timeouts beyond reading `timeout-minutes: 360` vs `--timeout=900` (consistent).
