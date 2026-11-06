# Supply chain audit, wave 4 (mlframe, 2026-11-06)

Read-only. Prior dispositions in audits/2026-10-31/supply_chain.md (SUP-01..15) are not re-reported. Live PyPI/GitHub checks were unavailable; tag-to-SHA mappings could not be verified.

## Findings

| ID | Severity | File:line | Failure scenario | Disposition |
|---|---|---|---|---|
| W4-SUP-01 | Low | .pre-commit-config.yaml:126, :200, :459 (`astral-sh/ruff-pre-commit rev: v0.16.10`) | Every other remote hook repo is pinned to a 40-char SHA; ruff is pinned by a mutable git tag. A force-moved or re-pushed tag (repo compromise) makes `pre-commit install`/autoupdate-free fresh caches (CI hooks-not-in-ci.yml cache key only hashes the config, so a cache miss refetches) execute different code on dev machines and CI. | RESOLVED-needed: pin to the commit SHA with `# v0.16.10` comment, matching the other hooks. |
| W4-SUP-02 | Low | .github/workflows/hooks-not-in-ci.yml:56 (`pip install "pre-commit>=3.3"`) | Floating install of the tool that then executes all hooks; a compromised latest pre-commit release runs in CI on push/PR. Not covered by SUP-05 (which listed docs/codecov-full/dep-floors only). Impact bounded: `permissions: contents: read` (:33-34), `persist-credentials: false`. | FUTURE: same remedy as SUP-05 (exact pin, ideally with hash). |

## Checked and safe

- All `uses:` in .github/workflows/*.yml pinned to full 40-char SHAs (grep found no non-SHA ref).
- Other remote pre-commit repos (pre-commit-hooks :56, detect-secrets :86, actionlint-py :175, shellcheck-py :186, black :221, mirrors-mypy :267, py-ci-shared :671) pinned by full SHA.
- requirements-dev.txt:65 py-ci-shared git URL pinned to full SHA.
- CI pyutilz installs use a local checkout (`./pyutilz[...]`), e.g. ci.yml:654, gpu-matrix.yml:53; no network git+ URL installs in workflows.
- scripts/check_dependency_python_floors.py:74 and scripts/ci_version_matrix.py:102 only read JSON metadata from PyPI; nothing downloaded is executed.
- macOS curl (macos-abort-probe.yml:173) already dispositioned as SUP-04.
