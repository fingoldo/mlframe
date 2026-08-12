# Cluster: x_cicd_dependencies

## Scope

`.github/workflows/**`, `.pre-commit-config.yaml`, `requirements-dev.txt`, `pyproject.toml` (dependency specs,
version floors/pins, dependency groups/extras), `.semgrep.yml`, `.yamllint`, `pre_test_gate.ps1`.

Files reviewed in full (19 files, 3016 LOC total, via `wc -l`):

- `.github/workflows/ci.yml` (275)
- `.github/workflows/black-filtered.yml` (26)
- `.github/workflows/codeql.yml` (44)
- `.github/workflows/dependabot-auto-merge.yml` (62)
- `.github/workflows/dependency-review.yml` (27)
- `.github/workflows/docs.yml` (43)
- `.github/workflows/gpu-extras-install-matrix.yml` (93)
- `.github/workflows/gpu-matrix.yml` (88)
- `.github/workflows/mypy-full.yml` (34)
- `.github/workflows/numba-coverage.yml` (126)
- `.github/workflows/release.yml` (86)
- `.github/workflows/sklearn-matrix-ci.yml` (200)
- `.github/workflows/update-test-durations.yml` (87)
- `.pre-commit-config.yaml` (655)
- `requirements-dev.txt` (16)
- `pyproject.toml` (1026)
- `.semgrep.yml` (63)
- `.yamllint` (16)
- `pre_test_gate.ps1` (62)

**Files reviewed: 19. LOC reviewed: 3016.**

`src/mlframe/feature_selection/filters/**` and `src/mlframe/feature_selection/shap_proxied_fs/**` were out of
scope and not touched. `.github/dependabot.yml`, `.github/actionlint.yaml`, and `CONTRIBUTING.md` are not in
this cluster's file list either, but were read for cross-referencing context (they are the only way to confirm
or refute several findings below that live in in-scope files); no finding here is anchored in those files alone.

This cluster was previously audited on 2026-07-21 (`audits/full_audit_2026-07-21/x_cicd_dependencies.md`,
findings F1-F7, tracker status CLOSED). All 7 of those findings were verified fixed in the current state of the
repo (duplicate stale-`ruff`-pinned tests/ pre-commit block removed, `ci.yml` `build` and `release.yml`
`publish` jobs both now carry `timeout-minutes`, `dependabot.yml`'s `pip` ecosystem re-enabled at
`open-pull-requests-limit: 2`, the `numba-coverage.yml` inert commented-out job block trimmed to a short
"Future idea" note, and the `gpu-extras-install-matrix.yml` `gpu`/py3.14 row now marked `experimental`). No
finding below duplicates any of F1-F7; this pass found new, distinct issues.

## Findings

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|----|----------|-----------|---------|----------------|-----------------|
| X_CICD_DEPENDENCIES-1 | P1 | `.github/workflows/ci.yml:181,262`, `mypy-full.yml:36`, `gpu-matrix.yml:51`, `gpu-extras-install-matrix.yml:75`, `sklearn-matrix-ci.yml:124`, `numba-coverage.yml:66` | Every workflow that installs pyutilz via a direct `git clone` does `git clone --depth 1 https://github.com/fingoldo/pyutilz.git` with no commit SHA, ref, or tag — always resolving to whatever is on the default branch HEAD at run time. | Pin every direct clone to an explicit commit SHA (`git clone --depth 1 <url> && git -C pyutilz checkout <sha>`, or a PEP 508 `pyutilz @ git+...@<sha>` install), bump deliberately with a CHANGELOG entry, mirroring the SHA-pin discipline `pyproject.toml` already documents for its own `pyutilz>=1.0.0` install instructions and for the `py-ci-shared` dev dependency. | A repo-scanning check that greps every `.github/workflows/*.yml` for `git clone` / `pip install .*@ git+` targeting a first-party sibling repo and asserts each one is followed by (or embeds) a 40-char commit SHA, not a bare branch clone. |
| X_CICD_DEPENDENCIES-2 | P1 | `pyproject.toml:405-407` vs `.pre-commit-config.yaml:96,175,430` | `pyproject.toml`'s `dev` extra pins `ruff==0.16.0` with a comment claiming this "match[es] `.pre-commit-config.yaml`'s ruff-pre-commit rev" — but all three `ruff-pre-commit` hook entries in `.pre-commit-config.yaml` are pinned to `rev: v0.15.22`, an older release. | Bump `.pre-commit-config.yaml`'s three `rev: v0.15.22` entries to `v0.16.0` (or drop `pyproject.toml`'s pin back to `0.15.22`) so the comment's claim is true again, and re-verify 0 new findings under whichever version is chosen. | A meta-test that parses the exact-pinned `ruff==X.Y.Z` line in `pyproject.toml`'s `dev` extra and every `rev:` under `repo: https://github.com/astral-sh/ruff-pre-commit` in `.pre-commit-config.yaml`, asserting they are all equal — generic form: any comment claiming two config files "match" a tool version should be backed by a scanner diffing the actual pinned strings, not trusted as prose. |
| X_CICD_DEPENDENCIES-3 | P1 | `.github/workflows/dependabot-auto-merge.yml:32-54` | The auto-merge eligibility regex requires the PR title to contain a single `from X.Y.Z ... to A.B.C` pair, but `.github/dependabot.yml` configures BOTH ecosystems (`github-actions` and `pip`) to group routine bumps into one PR via `groups:` (`github-actions: patterns: ["*"]`, and `python-dependencies: update-types: [minor, patch]`) — Dependabot's own grouped-PR title format (e.g. "Bump the python-dependencies group with N updates") contains no such single-pair version string at all. | Either parse Dependabot's grouped-PR title/body format (it lists each dependency's `from`/`to` pair in the PR body, not a single title pair) and gate on the max major-version delta across all of them, or scope this workflow explicitly to Dependabot's ungrouped/security-only PRs and document that grouped routine bumps still require a human click. | A workflow-level fixture test (already-established pattern in this repo, e.g. `test_numba_coverage_workflow_exists.py`) that feeds a real Dependabot grouped-PR title sample through the same bash regex extracted into a small script, asserting `eligible=true` for at least one realistic grouped-and-ungrouped title pair — would immediately show grouped titles never match. |
| X_CICD_DEPENDENCIES-4 | P1 | `.pre-commit-config.yaml:618-654` | The `mypy-full-manual` hook (from `fingoldo/py-ci-shared`, overridden here to run on every commit via `stages: [pre-commit, pre-merge-commit]`) still points `--cache-dir` at `../.mlframe_mypy_cache_shared`, the cross-worktree shared cache directory that TWO other mypy hooks in this SAME file (`beachhead` at line 257 and whole-project `mypy-full` at line 288) were deliberately moved OFF on 2026-08-03, per this file's own comment, after proving the shared cache can make a rerun against a known-buggy revision silently report "Success" because a different worktree's cache entry for the same module name masks the real error. | Give `mypy-full-manual` its own local `--cache-dir` (e.g. `.mypy_cache_precommit_manual`) exactly like its two siblings in this file, or delete the hook if the whole-project `mypy-full` hook already makes it redundant (see X_CICD_DEPENDENCIES-7). | A meta-test parsing `.pre-commit-config.yaml` for every `mypy`/`mypy-full*` hook's `--cache-dir` argument and asserting none of them resolve outside the current checkout (no `../` prefix) — generic form: any blocking correctness gate's cache directory must be scoped to the invocation, never shared across concurrent worktrees/sessions. |
| X_CICD_DEPENDENCIES-5 | P1 | `pyproject.toml:816` (`[tool.ruff] extend = "$PY_CI_SHARED_DIR/configs/ruff-base.toml"`) | Ruff's base config is pulled in via the `$PY_CI_SHARED_DIR` environment variable, but nothing tracked in this repo ever sets it: not `.pre-commit-config.yaml` (the `ruff (real bugs block)` hook runs `language: system` in the raw contributor shell), not any `.github/workflows/*.yml`, not `CONTRIBUTING.md`'s "Development setup" section (which walks through cloning pyutilz, installing extras, and running `pre-commit install`, but never mentions cloning `py-ci-shared` or exporting this variable). A fresh clone following `CONTRIBUTING.md` verbatim gets an unresolved `$PY_CI_SHARED_DIR` on their very first commit's ruff hook (the first blocking gate in the file's own documented ordering). | Either document the required `git clone .../py-ci-shared && export PY_CI_SHARED_DIR=...` step in `CONTRIBUTING.md`'s setup block, or have the local `ruff`/`black-filtered`/`interrogate` hooks resolve/clone `py-ci-shared` themselves (mirroring how `pyutilz` is bootstrapped), so a clean environment isn't silently missing a required var with no signal beyond a cryptic ruff config-resolution error. | A CI job (or a meta-test run in a genuinely clean/minimal env, e.g. a fresh Docker layer with only the documented `CONTRIBUTING.md` steps applied) that runs `pre-commit run --all-files` with `PY_CI_SHARED_DIR` deliberately unset and asserts it does NOT fail with a config-resolution error — generic form: any config `extend`/env-var indirection should have an explicit "onboarding smoke test" proving the documented setup path actually produces a working toolchain. |
| X_CICD_DEPENDENCIES-6 | P2 | `pre_test_gate.ps1:53` | The full-suite worker count is computed as `physical cores / 2` (`[Math]::Floor(... / 2)`), but the user's own standing convention (repeatedly documented) is quarter-cores for pytest -n (e.g. 16 physical → 4 workers) specifically to avoid oversubscription/Windows paging-file exhaustion under joblib fan-out from concurrent per-test model fits. This script computes double that worker count. | Change the divisor from `2` to `4` to match the documented convention, or add a comment explaining why this one entry point deliberately deviates (e.g. it is meant to run alone, not alongside other concurrent sessions) if that is intentional. | A meta-test (or a simple grep-based check) scanning `*.ps1`/CI YAML for `NumberOfCores` or `-n <expr>` pytest-xdist worker-count formulas and asserting the divisor constant matches the repo's documented policy value, so a future edit to one script doesn't silently drift from the others. |
| X_CICD_DEPENDENCIES-7 | P2 | `.pre-commit-config.yaml:268-292` and `:618-654` | Two separate mypy hooks now run on every commit: the local `mypy-full` hook (`python -m mypy src/mlframe --cache-dir=.mypy_cache_precommit_full`, whole project, added because the beachhead-only hook missed most of the codebase) and the shared `mypy-full-manual` hook (scoped to changed files, from `fingoldo/py-ci-shared`). Both are blocking (`stages: [pre-commit, pre-merge-commit]`) and both effectively check the same rule (project mypy cleanliness) on the same commit, roughly doubling mypy wall time paid on every commit for no additional coverage (the whole-project hook is already a superset of the changed-files one). | Drop `mypy-full-manual` (once its cache-dir hazard from X_CICD_DEPENDENCIES-4 is moot) and keep only the whole-project `mypy-full` hook, since it already dominates the changed-files-scoped one in coverage; or, if the changed-files-only hook is kept deliberately for its faster warm-cache time on large repos, add a comment explaining why both are needed simultaneously (the current comments for each hook explain its own history but never why the other one also still runs). | A meta-test asserting the pre-commit config has at most one blocking (`stages` including `pre-commit`) hook per (tool, target-scope) pair, flagging accidental duplicate blocking gates for the same tool — the exact bug class the earlier F1 finding (duplicated `tests/` lint bundle) already caught once for a different set of hooks. |
| X_CICD_DEPENDENCIES-8 | P3 | `.github/workflows/sklearn-matrix-ci.yml:68-72` | The job sets `defaults: run: shell: bash` with a comment explaining it's needed "since the windows-latest default shell (PowerShell) cannot parse" this job's bash line-continuations — but the job's own `strategy.matrix.os` and every `include` row are now `ubuntu-latest` only (the file's own comment at line 101-106 documents the Windows leg was dropped 2026-07-09). The `shell: bash` block and its rationale comment are now stale: harmless (bash is also the default on ubuntu-latest), but misleading to a reader who'd reasonably infer a Windows row still exists in the matrix from this comment alone. | Either trim the comment to note it is now vestigial (kept only in case a Windows leg is re-added, as the nearby comment already anticipates), or delete the now-unnecessary explicit `defaults.run.shell` block since `ubuntu-latest` already defaults to bash. | None needed beyond the existing "stale comment" hygiene sweep this cluster's own review already applies manually; not a generically automatable pattern (would require correlating a `defaults` block's stated rationale against the current matrix contents). |

Severity counts: **P0: 0, P1: 5, P2: 2, P3: 1** (8 findings total).

## Narrative

**X_CICD_DEPENDENCIES-1 (unpinned pyutilz clones, P1).** `pyproject.toml`'s own dependency comment for
`pyutilz>=1.0.0` explicitly documents the "X_SECURITY_API_PACKAGING-5" incident from `mrmr_audit_2026-07-22`:
an unpinned, same-day pyutilz commit once silently flipped a shared helper's default argument and broke
`MRMR.fit()` for every caller, with "no version boundary anywhere in this dependency graph that could have
caught or rolled back the change" — and the fix applied there was to pin `pyutilz`/`py-ci-shared` installs to
an exact commit SHA in documentation and in the `dev` extra's `py-ci-shared` git dependency. I grepped every
`.github/workflows/*.yml` for `pyutilz.git` and `git clone` and found 7 occurrences across 6 files
(`ci.yml` twice — the `lint-blocking` job's `deptry-install-command` and the `build` job's wheel-smoke-install —
plus `mypy-full.yml`, `gpu-matrix.yml`, `gpu-extras-install-matrix.yml`, `sklearn-matrix-ci.yml`,
`numba-coverage.yml`) that all still do a bare `git clone --depth 1 https://github.com/fingoldo/pyutilz.git`
with no SHA. `ci.yml`'s main `test` job and `update-test-durations.yml` route through the
`fingoldo/py-ci-shared/.github/actions/install-pyutilz` composite action instead, which lives in an external
repo I could not inspect from this sandbox — it may already pin internally, but the 6 direct-clone sites
definitely do not, so any of them can pick up a same-day breaking pyutilz commit mid-run with zero warning,
reproducing the exact incident class this repo has already been burned by once.

**X_CICD_DEPENDENCIES-2 (ruff version drift, P1).** Grepped both files directly: `pyproject.toml:405` pins
`"ruff==0.16.0"` in the `dev` extra with an explicit comment "pinned exact, matching
`.pre-commit-config.yaml`'s ruff-pre-commit rev AND CI's `uvx ruff==0.16.0`... an open range here is exactly how
the prior v0.8.6 pre-commit-vs-CI drift went unnoticed." But every `rev:` line under
`repo: https://github.com/astral-sh/ruff-pre-commit` in `.pre-commit-config.yaml` (lines 96, 175, 430 — the
blocking hook, the manual-fix hook, and the `tests/` blocking hook) is pinned to `v0.15.22`, not `v0.16.0`. This
is the identical bug class the previous audit's F1 finding fixed (a stale ruff pin silently diverging from the
"source of truth" pin elsewhere) recurring on a different axis: this time the comment's own claim of parity is
simply false against the current file contents. Concretely, a commit that's clean under the pre-commit hook's
older 0.15.22 ruleset can still fail once it reaches whatever actually resolves `ruff==0.16.0` (the `dev` extra,
and per the comment, CI itself), reintroducing exactly the "passes locally, fails once pushed" gap this repo's
own commit history says it already fixed once.

**X_CICD_DEPENDENCIES-3 (dependabot-auto-merge regex vs grouped PRs, P1).** Read `dependabot-auto-merge.yml`'s
`Decide whether this is a minor/patch bump` step: it bash-regex-matches the PR title against
`from[[:space:]]v?([0-9]+)\.[0-9]+\.[0-9]+.*[[:space:]]to[[:space:]]v?([0-9]+)\.[0-9]+\.[0-9]+`, and any title
that doesn't match falls through to `eligible=false` ("could not parse... needs manual review"). I then
cross-checked against `.github/dependabot.yml` (not itself in this cluster's file list, but load-bearing context
for a bug that lives entirely in the in-scope workflow file): BOTH configured ecosystems set a `groups:` block
— `github-actions` groups every single action bump under one PR (`patterns: ["*"]`), and `pip` groups every
minor/patch bump under one `python-dependencies` PR. Dependabot's own documented behavior for a grouped update
is a single PR whose title summarizes the group (e.g. "Bump the python-dependencies group with N updates"),
which contains no single `from X.Y.Z to A.B.C` pair at all — the regex can only ever match an UNGROUPED,
single-dependency PR (which, given this repo's own grouping config, is now the atypical case, not the norm).
The practical effect: the auto-merge workflow's documented purpose ("Enables Dependabot PRs... to merge
themselves once this repo's own CI is green, without a human clicking merge for every routine bump") silently
fails to fire for the majority of the PRs it exists to handle, and nothing in the workflow signals this — the
"could not parse" branch just logs a line and skips, indistinguishable in outcome from "correctly deferred to a
human for a major bump."

**X_CICD_DEPENDENCIES-4 (mypy-full-manual's unsafe shared cache dir, P1).** While reading `.pre-commit-config.yaml`
end to end I found the `mypy-full-manual` hook (lines 618-654, sourced from `fingoldo/py-ci-shared@v1.0.0` but
overridden here to run on every commit) still passing `args: ['--cache-dir=../.mlframe_mypy_cache_shared']` — a
directory OUTSIDE this checkout, shared across sibling worktrees. Forty lines earlier in the SAME file, the
`mypy-full` hook's own comment (lines 280-287) documents why this exact pattern was abandoned there on
2026-08-03: "with the shared cache warm, swapping a file back to a KNOWN-BUGGY prior revision and rerunning
mypy against that same shared cache still reported 'Success' — a different worktree's cache write for the same
module name can mask a real error in this one." The identical hazard is proven, documented, and fixed for two
of the three mypy hooks in this file (`beachhead` and `mypy-full`) but was never applied to the third
(`mypy-full-manual`), which is still active on every commit. Since this repo's own CLAUDE.md explicitly flags
"multiple parallel agent sessions share this working tree" as a standing condition, this is a live, reachable
gap in a blocking correctness gate, not a hypothetical one.

**X_CICD_DEPENDENCIES-5 (undocumented `$PY_CI_SHARED_DIR` requirement, P1).** `pyproject.toml:816` sets
`[tool.ruff] extend = "$PY_CI_SHARED_DIR/configs/ruff-base.toml"` (and `tests/ruff.toml` does the analogous
thing for the `tests/` config). I grepped the entire repo for `PY_CI_SHARED_DIR` and found it referenced only
in these two config files — never assigned, exported, or even mentioned anywhere else: not in any
`.github/workflows/*.yml`, not in `.pre-commit-config.yaml`'s `env:`/hook definitions, not in
`CONTRIBUTING.md`'s "Development setup" walkthrough (read in full — it covers cloning pyutilz, installing
extras, and `pre-commit install`, then stops), not in `README.md`. Since the `ruff (real bugs block)` pre-commit
hook runs `language: system` directly in the contributor's own shell (not an isolated/bootstrapped env), a
fresh clone following `CONTRIBUTING.md` exactly has no way to have this variable set unless they happen to
already have a sibling `py-ci-shared` checkout and have manually exported it from outside any documented step —
and this hook is explicitly the FIRST one in the file's own stated ordering ("moved ahead of... so the modal
'oops, typo' commit fails fast"), so this would be the very first thing to break for a new contributor's first
commit. I could not execute `pre-commit run` in this sandbox to confirm the exact failure mode (unresolved
shell-style `$VAR` left literal vs. ruff erroring on a nonexistent path), but either outcome is a broken/empty
lint config, not a working one.

**X_CICD_DEPENDENCIES-6 (pre_test_gate.ps1 worker-count divisor, P2).** The full-suite invocation computes
`$n` as `physical cores / 2` (line 53), rounded down. This directly conflicts with the standing documented
policy of quarter-cores for pytest-xdist worker counts on this machine (16 physical → 4 workers), a policy
motivated by this repo's own well-documented Windows paging-file exhaustion failure mode under joblib fan-out
from concurrent per-test model fits (`WinError 1455`). Doubling the worker count relative to the documented-safe
value increases the chance of hitting that exact failure mode on the one script whose entire purpose is running
"the full suite" (as opposed to a scoped/targeted subset where fewer concurrent heavy fits are in flight).

**X_CICD_DEPENDENCIES-7 (duplicate blocking mypy hooks, P2).** Directly follows from investigating
X_CICD_DEPENDENCIES-4: once both `mypy-full` (whole-project) and `mypy-full-manual` (changed-files) are
confirmed to both run on every commit with `stages: [pre-commit, pre-merge-commit]`, the changed-files hook
adds no coverage the whole-project hook doesn't already provide (a whole-project scan is a strict superset), so
every commit currently pays for mypy twice for zero net new correctness signal. This is the same "two hooks
checking the same thing, one stale" shape as the previous audit's F1 finding, just recurring on mypy hooks
instead of the `tests/` ruff/black/bandit/interrogate/codespell bundle.

**X_CICD_DEPENDENCIES-8 (stale Windows-shell comment in sklearn-matrix-ci.yml, P3).** The job's
`defaults: run: shell: bash` block (lines 68-72) carries a comment explaining it exists because
"the windows-latest default shell (PowerShell) cannot parse" the job's bash continuations — but the same file,
36 lines later, documents that the Windows leg was dropped entirely on 2026-07-09 for cost reasons, and the
current `matrix.os` plus every `include` entry is `ubuntu-latest` only. The config itself is harmless
(`ubuntu-latest` already defaults to bash), but the comment now describes a scenario (a Windows row in this
matrix) that no longer exists, which is exactly the "stale/misleading docstrings and comments" class this
review's dimension 6 calls out.

## Coverage notes on review dimensions

- **ML correctness (leakage/reproducibility/calibration/sample_weight/OOS/imbalance):** not applicable to this
  cluster — no ML logic lives in CI/CD/dependency config. The closest analog, reproducibility of the CI
  environment itself, is covered by findings 1 and 2 above.
- **Computational efficiency:** covered by findings 6 (oversubscribed worker count) and 7 (duplicate mypy runs);
  no other efficiency issues found (caching strategy for numba JIT and uv/pip is already well-reasoned and
  documented inline in `ci.yml`/`sklearn-matrix-ci.yml`).
- **Edge cases/robustness:** covered by findings 3 (grouped-PR title format) and 5 (missing env var on a clean
  clone); no other robustness gaps found in the matrix/trigger/permission structure of the workflow files.
- **Test coverage gaps:** every finding above proposes a meta-test; no additional CI/CD-specific test-coverage
  gap was found beyond what's already listed.
- **Code quality/architecture:** covered by finding 8 (stale comment) and finding 7 (duplication); the rest of
  `.pre-commit-config.yaml` and the workflow files are unusually well-commented and internally consistent for a
  file of this size — no dead code, no misleading naming, no broad `except` clauses (this cluster has no Python
  execution logic of its own to except-clause).
- **OSS/hygiene:** no mojibake, no stray audit/phase markers beyond the legitimate historical fix-log comments
  this repo deliberately keeps (which are dated engineering rationale, not process metadata, and match this
  project's own comment-style convention); `.semgrep.yml` and `.yamllint` are both small, correct, and current.
