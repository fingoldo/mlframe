# Wave 1 (2026-11-05), attack class A3: network, secrets and supply chain

Read-only audit by one Opus 5.5 low-effort agent. The tree is mostly hardened in this class: three verified findings, all Low, none High or Medium. The report is the agent's returned text, condensed; each finding is re-checked on HEAD before it is fixed. The agent finished while its own background work was still running, so this report may be interim; the secret sweep below did not complete.

## Findings

| ID | Severity | Location | Defect |
|---|---|---|---|
| N3-01 | Low | `.pre-commit-config.yaml:126`, `:200`, `:459` | `astral-sh/ruff-pre-commit` is pinned to the mutable tag `rev: v0.16.10`, while every other external hook is pinned to a commit SHA. |
| N3-02 | Low | `src/mlframe/feature_selection/filters/_neural_mi.py:278-291` | The InfoNet checkpoint pin trusts whatever file it sees first: it records the sha256 of that file instead of checking it against a known digest. |
| N3-03 | Low | `src/mlframe/training/feature_handling/hf_provider.py:74-84` | `resolve_revision` loads hub `main` for any model not in the pinned table when the caller gives no revision, and only logs a warning. |

### N3-01

The comment marks pre-commit hooks as pinned. These three entries resolve a git tag, and a tag can be force-moved upstream. If the ruff-pre-commit repo or its tag were compromised, that code would run on every developer commit and in `hooks-not-in-ci.yml` (`pre-commit>=3.3`, also open-ended). Fix: pin the SHA, as is already done for black, mypy, actionlint and detect-secrets.

### N3-02

The docstring says the checkpoint "is downloaded by hand", following the gdown `download_folder` hint printed at `:305`. On first use, whatever file sits at `~/.cache/mlframe/infonet/` has its digest recorded and is trusted from then on. If an attacker swaps the Google Drive folder, or plants the file before first use, the poisoned weights are pinned permanently. The damage stops short of code execution because the load is `torch.load(..., weights_only=True)` (`_vendored/infonet/infer.py:51`); what remains is silently corrupted MI scores feeding feature selection. Fix: ship the expected sha256 as a constant and compare against it on first use.

### N3-03

Suppose a user passes an unlisted model such as `hf://org/model` without `revision`. Each fit can then pull a different upstream commit, and a hijacked or retagged repo changes the embeddings with no error. `trust_remote_code` is hard-coded to False unless the caller opts in (`:225-232`), so this cannot run remote code by default; it is a reproducibility and integrity gap.

## Checked and clean

- Workflows: every `uses:` is pinned to a SHA, including the py-ci-shared reusable workflows; top-level `permissions:` blocks are present; there is no `pull_request_target`; Dependabot auto-merge uses `pull_request` gated on the dependabot login; `github.event.pull_request.html_url` is passed through `env:`, not interpolated into the script; `codecov-full`'s `workflow_run` trigger, which carries `CODECOV_TOKEN`, is gated on success and on the upstream event being `push` or `workflow_run`; `${{ matrix.extra }}` in `gpu-extras-install-matrix.yml:84` takes its values from a static matrix.
- pyutilz clones in ci, gpu-matrix, gpu-extras, mypy-full and dep-floors all check out SHA `38532cd5...`; `requirements-dev.txt:65` is py-ci-shared pinned to a SHA.
- mlflow integration: credentials embedded in URLs are scrubbed from exception text (`integrations/mlflow.py:40-57`) and exception chains are dropped; the iframe URL is escaped with `html.escape`; the search filter strings are escaped for the DSL.
- `torch.load` is called with `weights_only=True` everywhere found; the MIST model is loaded with a pinned revision (`_neural_mi.py:491`).
- No source code reads token, key or secret environment variables; there are no `requests`/`urllib`/`httpx` calls in `src` outside `_benchmarks`; nothing in `src` binds a server or a socket.

## Not covered

- Full-tree secret scan: the grep for hard-coded credentials across src, tests, scripts and docs timed out and its results were never obtained. detect-secrets is present in pre-commit; its baseline was not checked.
- Called workflow internals: the jobs inside the py-ci-shared reusable workflows, the upload-codecov and install-pyutilz actions, and `scripts/sync_and_push.sh`.
- `uv.lock`: not checked for direct-URL entries or hash coverage.
- `release.yml`: the trusted-publishing and OIDC scope was not reviewed in depth.
- Dash or other report servers: none appeared in the greps; dynamic launches were not traced.
