# CI / linters / presentation review 2026-09-08 - disposition tracker

One row per finding from every agent report in this folder. Dispositions: **RESOLVED** (fixed, with
commit), **FUTURE** (deferred, with reason and a concrete next action), **DOC** (no code change; the
finding is recorded because it is true and worth knowing), **REJECTED** (with the evidence that
disproves it). No finding is dropped.

Reports: `workflows.md` (D/G/W/T), `readme_badges.md` (F/S/P), `linters.md` (L).

## Facts established before dispositions

Several findings were marked INFERRED by their agent. Checked directly:

| Question | Answer | Method |
|---|---|---|
| Does branch `main` exist? | **No** | `git ls-remote --heads origin main` -> empty |
| Is the docs site live? | **Yes** | fetched `https://fingoldo.github.io/mlframe/`, renders "mlframe documentation index" |
| Is `pyutilz` on PyPI? | **No** | `pypi.org/pypi/pyutilz/json` -> 404 |
| Is `mlframe` on PyPI? | **Yes - and it is someone else's project** | `pypi.org/pypi/mlframe/json`: author Sam Stoltenberg, `github.com/skelouse/mlframe`, last release 0.1.15 on 2020-12-18 |
| Is the Dependabot `GITHUB_TOKEN` downgraded to read-only? | **No** | run 34230983189 log: `Contents: write`, `PullRequests: write`; PRs #23 and #26 merged via auto-merge |
| Do mypy-full / black-filtered / codeql run on push to master? | **Yes**, all three | their `on:` blocks |
| Do the six README gallery PNGs exist at the quoted paths? | **Yes**, all six | filesystem check |

### Escalation, found while checking the above and outside both agents' scope

**The distribution name `mlframe` is not available on PyPI.** It belongs to an unrelated 2020 package
by another author. `release.yml`'s Trusted Publishing upload therefore cannot ever succeed under this
name - PyPI will reject it as a project the account does not own - and a `pypi/v/mlframe` badge would
display a stranger's `0.1.15`. This is a naming decision for the owner, not something to fix in CI;
recorded here as row **X1** and raised directly.

---

## Agent 1 — `workflows.md`

### Defects

| # | Finding | Disposition | Notes |
|---|---|---|---|
| D1 | Shard durations artifact name omits `matrix.os`, so three OS legs collide | **RESOLVED** `1121abf65` | Name now carries the OS. Latent only because macOS shards died earlier and Windows was cancelled. |
| D1b | Durations merge is last-write-wins across platforms | **RESOLVED** `191466f53` | The download pattern is narrowed to `COVERAGE_OS`; balancing on a blend of three platforms is not reproducible. |
| D2 | `ci-required` passes vacuously when `matrix-setup` fails or a gate is skipped | **RESOLVED** `1121abf65` | Added `matrix-setup` to `needs` and made `skipped` a failure. Severity was higher than the agent could know: master's branch protection **does** require "CI required checks" (verified via the API), so this was a green required check over zero tests. |
| D3 | `numba-coverage.yml`'s serial job lacks the `workflow_run` gate | **RESOLVED** `191466f53` | Copied the sharded job's condition verbatim. |
| D4 | `dependabot-auto-merge` cannot merge (inferred read-only token) | **REJECTED** | Disproven by evidence. Run 34230983189's log shows `Contents: write`, `PullRequests: write` — no downgrade — and `gh pr merge --auto` exits 0. PRs #23 and #26 merged through this workflow. The unmerged github-actions-group PRs are superseded ones dependabot closed itself. No change. |
| D5 | `release.yml` publishes to PyPI from a dispatch on any ref | **RESOLVED** `191466f53` | `publish` is now `if: github.event_name == 'release'`. |
| D6 | Reusable workflows on the movable tag `@v1` while every action is SHA-pinned | **RESOLVED** `191466f53` | All nine pinned to `915217a4a8cd…` (the commit `v1` dereferences to). The composite actions were already SHA-pinned. |
| D7 | Three pushers to master with no rebase and no shared lock | **RESOLVED** `191466f53` | Two remain after W4; both rebase-and-retry and share `concurrency: test-durations-commit`. |
| D8 | Durations committed from failed/cancelled shards | **RESOLVED** `191466f53` | Commit step requires a green test job; uploads stay unconditional. |
| D9 | `main` is a trigger branch several jobs ignore | **RESOLVED** `191466f53` | `git ls-remote --heads origin main` is empty; dropped from all seven workflows. |

### Gaps

| # | Finding | Disposition | Notes |
|---|---|---|---|
| G1 | No lockfile / no pinned resolution | **RESOLVED** | `uv.lock` committed, 594 packages over Python 3.9.2-3.14. Every blocker it hit was a real defect in the declared metadata, not an artefact of locking — four of them, itemised in Round 2 below. Gated by a new `lockfile` CI job (`uv lock --check` plus an actual `uv sync --frozen`), and dependabot moved from the `pip` ecosystem to `uv` so a bumped dependency and its lock arrive in one PR. |
| G2 | No lowest-supported-version resolution job | **RESOLVED** `191466f53` | New `dep-floors.yml`: weekly + on `pyproject.toml` PRs, `--resolution=lowest-direct` on Python 3.9, install + import smoke + a scoped test slice. Non-blocking until the failure count is known. |
| G3 | No aggregated test-failure report | **RESOLVED** `191466f53` | Every shard writes JUnit XML; a non-gating `report-tests` job renders one table into the run summary. The renderer was run against a synthetic report first. |
| G4 | Build/install smoke is Linux-only | **RESOLVED** `191466f53` | New `install-smoke` job installs the built wheel on all three platforms and gates. |
| G5 | No release provenance / SBOM | **RESOLVED (provenance)** `191466f53`; **REJECTED (SBOM)** | `actions/attest-build-provenance` added to `release.yml`. SBOM rejected for now: nothing consumes one here, and it is ceremony until the distribution can actually be published (X1). |
| G6 | Nothing verifies the workflows themselves | **REJECTED** | Premise false. `actionlint` runs as a blocking pre-commit hook (`.pre-commit-config.yaml:149-152`), alongside yamllint and zizmor. The agent checked only `.github/workflows/`. Ran it against every change in this round: clean. |
| G7 | `deep-nightly` durations are never consumed | **FUTURE** | The workflow's own header already records this as deliberate and says to wire it once the run is green. It is not green yet, and balancing 20 shards on timings from a never-completed suite is guesswork. **Next action:** add the merge-and-commit job after the first fully green deep run. |

### Waste

| # | Finding | Disposition | Notes |
|---|---|---|---|
| W1 | `deep-nightly` has no uv cache: 20 cold multi-GB resolves nightly | **RESOLVED** `191466f53` | `enable-cache` + `cache-dependency-glob`, matching every other uv call site. |
| W2 | `cache: pip` on setup-python while installing with uv | **RESOLVED** `191466f53` | Dropped from `deep-nightly`; kept in `numba-coverage`, where the installs really are pip. |
| W3 | No `paths-ignore`: a README edit runs 30 shards | **RESOLVED (push only)** `191466f53` | Applied to `push`, deliberately **not** to `pull_request`. The agent justified it with "this repo has no required checks" — it has three, so a required check that never runs leaves a docs-only PR permanently pending and unmergeable. Push is where the spend is anyway. |
| W4 | `update-test-durations.yml` is superseded and cannot finish | **RESOLVED** `191466f53` | Deleted. Its own header records the supersession; `.test_durations` has 30 131 entries, all from the ci.yml path. |
| W5 | Four near-identical pyutilz install stanzas | **SUPERSEDED by G1** | Originally dispositioned as "extend `install-pyutilz` to cover the CUDA cases, then convert all of them" — which was wrong, and the user caught it. The seven hand-synced SHA copies exist because there was no lock; adding machinery to keep copies in step treats the symptom. With `uv.lock` the resolved commit is recorded once, machine-side. **Next action:** move the CI installs to `uv sync --frozen` and retire the `pyutilz-ref` inputs entirely. See M1. |
| W6 | `codecov-full.yml` has no `concurrency` group | **RESOLVED** `191466f53` | Added, `cancel-in-progress: true`. |

### Tools table

| Tool | Disposition |
|---|---|
| `uv lock` + `uv sync --frozen` | **FUTURE** — see G1. |
| `actionlint` | **REJECTED** — already running, see G6. |
| `--junitxml` + summary | **RESOLVED** — see G3. |
| `attest-build-provenance` | **RESOLVED** — see G5. |
| `--resolution=lowest-direct` | **RESOLVED** — see G2. |
| `step-security/harden-runner` | **REJECTED** — audit mode produces an egress log nobody is assigned to read, on a repo whose dependency intake is already gated by dependabot, dependency-review and pip-audit. Revisit if a supply-chain incident gives it a reader. |
| `pytest-timeout method=signal` | **RESOLVED** `191466f53` — `signal` on POSIX, `thread` on Windows (no SIGALRM). |
| `--max-worker-restart` | **REJECTED** — it converts a native crash into a silently restarted worker. This repo's own convention treats a crash as a bug detector; masking it is the opposite of the fix. |

---

## Agent 2 — `readme_badges.md`

| # | Finding | Disposition | Notes |
|---|---|---|---|
| F1 | The three codecov flag badges are well-formed | **DOC** | Verified clean by the agent; no action. Residual risk it names (a flag with no data renders `unknown`) is real and unaddressed — recorded, not fixed, because the fix is "keep the nightlies green". |
| F2 | Four uninterpretable coverage badges in the header | **RESOLVED** `e441b93fc` | One headline badge in the header; all four moved under Testing with a line each. Descriptions checked against the workflows, not inferred from flag names. |
| F3 | The `docs` badge may link to a 404 | **REJECTED** | Fetched `https://fingoldo.github.io/mlframe/`: live, renders "mlframe documentation index". No change needed. |
| F4 | Three workflow badges are branch- and event-agnostic | **RESOLVED** `e441b93fc` | Modern path form, `branch=master`, links filtered to master. `event=push` added only after checking each workflow's `on:` block; omitted for sklearn-matrix, whose push trigger is paths-filtered. |
| F5 | All three flag badges link to the same undifferentiated page | **RESOLVED** `e441b93fc` | Superseded by the F2 restructure, as the agent expected. |
| a–e | Missing badges: Python versions, Ruff, CodeQL, pre-commit, py.typed | **RESOLVED** `e441b93fc` | All five added; each verified against what the repo actually declares. |
| — | Badges NOT to add (PyPI, conda, Scorecard, Codacy, Contributor Covenant, decoration) | **DOC** | Agreed, and X1 makes the PyPI ones actively harmful: `pypi/v/mlframe` would render a stranger's 0.1.15. |
| S1 | The install section recommends a PyPI install of a package not on PyPI | **RESOLVED** `1121abf65` | Now the editable local-path form every sibling line uses. |
| S2 | No "why", no runnable example above the fold | **RESOLVED** `e441b93fc` | Tagline, Why (five substantiated claims), Quickstart, Documentation section. The Quickstart is the existing worked example trimmed — a first draft written from memory had the wrong import path, arguments and return shape. |
| S3 | No table of contents in a 715-line file | **RESOLVED** `e441b93fc` | Added after the badges. |
| S4 | `docs/README.md` mixes user guides with research notes | **RESOLVED** `e441b93fc` | Research block behind a `<details>` fold. Also fixed NESTED_PARALLEL.md, which was sitting above that section's intro as a bare bullet among tables. |
| S5 | CHANGELOG `[Unreleased]` has two `### Fixed` headings | **RESOLVED** `e441b93fc` | Two `### Fixed` **and** two `### Added`. Merged into Keep a Changelog's order; all 103 entries preserved, counted before and after. `## [0.9.0]` does exist, so the agent's second worry is clear. |
| P4-a | README's relative links and images break on the PyPI page | **FUTURE** | Correct, and all six PNG paths verified to exist. Not done now for two reasons: the distribution cannot be published under this name at all (X1), and absolutising every cross-file link degrades the mkdocs site, which consumes this same file. **Next action:** do it in the same change that settles the distribution name, and check the rendered mkdocs output. |
| P4-b | `pyutilz>=1.0.0` is a hard runtime dep not on PyPI | **DOC** | Confirmed: `pypi.org/pypi/pyutilz/json` returns 404. Nothing to fix in the metadata; it gates X1. |
| P4-c | `description` is 178 chars, PyPI truncates it | **RESOLVED** `e441b93fc` | Now 130, distinguishing part first. |
| P4-d | No MIT license classifier, and there should not be | **REJECTED (correctly)** | The agent's own recommendation is "do not fix"; recorded so it is not re-flagged. |
| — | Incidental: `codecov.yml` comments name a python-3.11 leg that no longer exists | **RESOLVED** `1121abf65` | Comments now name `REPRESENTATIVE_PYTHON` / `COVERAGE_OS`. |

---

## Agent 3 — `linters.md`

### Misconfiguration

| # | Finding | Disposition | Notes |
|---|---|---|---|
| L1.1 | Every CI lint tool runs unpinned via `uvx` | **RESOLVED (2026-09-12)** | Added one `*-version` input per uvx-run tool to `fingoldo/py-ci-shared`'s `lint-blocking.yml` (`64e2b6b`), each defaulting to `""` (unpinned) so every OTHER caller of this shared, multi-repo workflow sees zero behaviour change. Wired mlframe's own exact pins in via `with:` on both call sites in `ci.yml`: `actionlint-py 1.7.12.24` (from `.pre-commit-config.yaml`'s `rev:`), `codespell/yamllint/vulture/interrogate` (from `[project.optional-dependencies].dev`, the same exact pins L1.4/L1.2 already assert against the installed copy). `bandit` and `zizmor` stay unpinned -- no exact version exists anywhere in this repo for either, and pinning one blind would be exactly the "guessed number" L1.6/L3.3 already reject. Bumped both `lint-blocking.yml@<sha>` references to the new commit. |
| L1.2 | The pinned ruff is not the ruff that runs | **RESOLVED** | Confirmed live: every config said 0.16.1, the installed ruff was 0.15.22. Added `test_installed_tool_versions_match_their_exact_pins`, verified failing before the fix, then installed the pin. **The real pin then found 28 findings the stale one could not see** — 26 ISC004, 1 RUF036, and a genuine duplicate entry in `mlframe.training.__all__` (RUF068). All 28 fixed; the tree is clean under the pinned ruff. |
| L1.3 | mypy is a blocking zero-error gate on `mypy>=1.0` | **RESOLVED** | Pinned `mypy==2.1.0`, matching `mypy-full.yml`, and covered by the new version test. |
| L1.4 | Five blocking hooks invoke tools declared nowhere | **RESOLVED** | `codespell`, `deptry`, `interrogate`, `vulture`, `yamllint` pinned exactly in the dev extra and added to deptry's DEP002 list. |
| L1.5 | `detect-secrets` / `shellcheck` run only in pre-commit | **RESOLVED (advisory)** | New `hooks-not-in-ci.yml` runs exactly the nine hooks with no CI counterpart, one per step so a failure names itself. Advisory for now because three of them are auto-fixers whose failure mode on a fresh checkout is "I rewrote your files" and nobody has measured how many that touches. **Next action:** flip `continue-on-error` off once a run reads clean. |
| L1.6 | No coverage gate anywhere | **RESOLVED** | The first green `codecov-full` run (2026-09-10, run `34487120883`) reported a combined TOTAL of 77.79%. `codecov.yml`'s `coverage.status.patch.default` now targets that number with `informational: true` -- reports on every PR/commit without blocking, since the number has never gated anything before and needs to be watched under real traffic first. **Next action:** flip `informational` off once a handful of real patches have reported cleanly against it. |
| L1.7 | `vulture --min-confidence 80` reports nothing | **DOC** | Reproduced (0 at 80, 1071 at 60). Not lowered: 1071 findings need a baseline first, and the uncalled-function half is already gated separately. The residue the agent identifies — unused attributes — is a genuine ML-config bug class and is the right shape for a future scoped scan. Recorded rather than fixed. |
| L1.8 | The two vulture runs contradict the comment justifying them | **DOC** | Accurate. Left as-is deliberately: correcting the comment without changing the invocation is honest, but the invocation change (one pass over `src` + `tests`) belongs with L1.7's baseline work, and splitting them would mean editing the same comment twice. |
| L1.9 | `doctest_optionflags` set, no doctest ever runs | **DOC** | True. Kept rather than dropped because 3.4 (a scoped `--doctest-modules` leg) would need exactly this setting; deleting it now to re-add it later is churn. |
| L1.10 | The mccabe comment says 70, ruff reports 92 | **RESOLVED** | Comment corrected to the measured 92, and `test_c901_debt_ratchet.py` added: the count may fall freely, a rise fails, a ceiling drifting more than 10 above the real count fails, and the pyproject comment must agree with the ratchet. |
| L1.11 | `LOC_BUDGET_EXEMPT` has no staleness check and its docstring is wrong | **RESOLVED** | Docstring corrected, `pytest.skip` on a missing src tree replaced with a hard failure, and two assertions added. **They immediately found two drained exemptions** — `_gpu_resident_basis.py` at 868 LOC and `transforms/nonlinear.py` at 624 — both removed from the set, so those files are now gated like every other. |
| L1.12 | Baselines drain-warn but never require draining | **PARTIALLY RESOLVED** | The agent's own caveat blocks the general form: the large baselines are `file:line`-keyed, so failing on drained entries turns every unrelated line drift red. The nine empty ones have no such tension, and the risk there is the opposite one the finding names -- silencing a NEW violation by appending to a zero-tolerance file. `test_zero_tolerance_baselines_stay_empty.py` now asserts all nine stay at zero and that any baseline draining to zero joins the list. **Next action:** measure a week of churn on one large baseline before deciding the strict form for the rest. |
| L1.13 | `regen_baselines.py` covers 7 of 27 baselines; three unchecked lists | **RESOLVED** | Added `test_baseline_registry_consistency.py`. Resolving flags by asking the live pytest config, not by grepping conftest — several are registered indirectly through `py_ci_shared`, which a text scan reports as missing. **It found four refresh flags that every one of their own test modules documents and that pytest rejects outright**: `pytest --refresh-tick-isinstance-baseline` errored with "unrecognized arguments". All four registered and verified working. Twenty baselines were undocumented in BASELINES_README.md; all now documented. Six non-conventional routes recorded explicitly, two of which (`_code_audit_tests`, `_uncalled_functions`) genuinely have no automated refresh — see below. |
| L1.13b | `_code_audit_tests_baseline.json` and `_uncalled_functions_baseline.json` have no refresh route at all | **RESOLVED** | Verified against the tree rather than trusted from the original note (`test_baseline_registry_consistency.py`'s own `FLAG_EXEMPT` predates `py_ci_shared` gaining a refresh flag for each scanner): `_code_audit_tests_baseline.json` already refreshes via the SAME `--refresh-code-audit-baseline` flag the src-level baseline uses (`assert_no_new_code_audit_findings` gates both baselines through one option) -- it was never actually missing a route, only missing one matching the file-name-derived guess `test_baseline_registry_consistency.py` checks for; `FLAG_EXEMPT`'s entry corrected to name the real flag instead of "no refresh route yet". `_uncalled_functions_baseline.json` genuinely lacked wiring: `py_ci_shared.uncalled_functions` already exposes `--refresh-uncalled-functions-baseline` + `register_refresh_option`, but `tests/test_meta/conftest.py` never called it (confirmed empirically: `pytest ... --refresh-uncalled-functions-baseline` failed with "unrecognized arguments" before the fix). Registered it alongside the other two shared-flag imports there; re-verified the same invocation now succeeds and the file's `FLAG_EXEMPT` entry was removed (it satisfies the filename-derived convention directly once registered). `test_baseline_registry_consistency.py` (3 tests) passes. |
| L1.14 | `test_precommit_ci_scope_parity.py` covers 4 paired hooks, not the ruff pair | **RESOLVED** | The obstacle the agent identified was real and slightly worse than described: three hooks share the id `ruff` (blocking src, blocking tests, manual auto-fixer), so the id-keyed lookup silently kept only the last -- the highest-traffic gate was the one pair unchecked. Lookup re-keyed on `name` (falling back to id where unique) and the pair added. Teeth verified: the two names resolve to distinct hooks and partition the src/tests samples exactly XOR. |
| L1.15 | `bandit -ll` discards all LOW-severity findings | **REJECTED** | The agent rates it low priority itself and the residual is subprocess handling in benchmark scripts. `-l -i` would add findings nobody is assigned to triage, on a class the bare-except meta-gates already partly cover. |

### Coverage gaps in the checking

| # | Finding | Disposition | Notes |
|---|---|---|---|
| L2.1 | `RuntimeWarning` never fails anything | **RESOLVED (advisory leg), FUTURE (blocking)** | Agreed this is the highest-value gap for a numerics library, and the agent could not size it. Measured instead of guessed: `tests/metrics tests/calibration tests/evaluation` under `-W error::RuntimeWarning:mlframe`, 1592 tests, **zero** warning-caused failures (the one failure was unrelated — see X2). Added a non-blocking `runtime-warnings` census job to `deep-nightly.yml` running the per-push selection with the filter, to produce the same number for the feature-engineering and feature-selection trees the measurement did not reach — which is where the kernels that emit these warnings live. **Next action:** once the census reads zero, move the filter into pyproject's `filterwarnings` and delete the job. |
| L2.2 | Tool-version-to-gate-result coupling unchecked | **RESOLVED** | This is L1.2's generalisation and the new test is written that way: it checks every exact pin in the dev extra against `importlib.metadata`, not just ruff. |
| L2.3 | Wheel package-data only half-verified | **RESOLVED** `191466f53` | All three declared `package-data` paths are asserted in the built wheel, not just `py.typed`. |
| L2.4 | Public typing surface never checked from a consumer's position | **RESOLVED** `191466f53` | An advisory step in `install-smoke` runs mypy against `import mlframe` from a clean env holding only the wheel. Advisory because stub availability in a bare env is not this repo's to guarantee. |
| L2.5 | Docstring examples unverified; interrogate's denominator is narrow | **DOC** | The scope is defensible as the agent says, and the executional half is L1.9/3.4. No change. |

### New tools

| # | Finding | Disposition | Notes |
|---|---|---|---|
| L3.1 | Enable `PLW0127`, `PGH003`, `PGH004`, `PLE2515` | **RESOLVED** | All four enabled and all 46 findings closed. The 33 type-ignore codes were **measured**, by running mypy over an identical checkout with every blanket ignore stripped, not guessed; the five remaining sit under `_benchmarks/`, which mypy does not check, so there is no code to name and those paths are exempted explicitly. `PLE0605` left off as the agent advises (all ten hits are `__all__ = sorted([...])`). |
| L3.1b | `predict.py:489` is a probable live bug; should be `uncertainty[:, 0]` | **REJECTED (the inference) / RESOLVED (the dead code)** | It must not be `uncertainty[:, 0]`. `uncertainty` is `std_preds.mean(axis=1)` over the `(N, K)` Welford std, so it is already `(N,)` and subscripting it would raise `IndexError`. The line is dead, not wrong; the branch is removed and the reason recorded in the comment. |
| L3.2 | `mypy: strict_equality = true` | **RESOLVED** | Measured before enabling, on the whole project: **one** finding. `_cv_aggregation.py`'s `arr.size < 2 and mode != "mean"` -- the second half is always true, since the early return above is the only path handling `"mean"`. Dead condition, not a bug; removed, rule enabled, and whole-project mypy re-run clean over 1559 files. `warn_unreachable` stays off, as the agent advised: the optional-dependency `try/except ImportError` gates are what it over-reports on. |
| L3.3 | `pip-audit` blocking rather than advisory | **PARTIALLY RESOLVED** | The audit now exists over the right subject and runs on a schedule; only the blocking half is deferred, and for the reason the agent predicted. Measured: 18 advisories, of which **two are direct dependencies** (`aiohttp` needs 3.14.3, `tornado` needs 6.5.8) and **two are structurally capped by other packages** (mlflow holds `cryptography<49`, torch holds `setuptools<82`) -- the latter pair is the `--ignore-vuln` list the finding asks for, and it is a fact about upstream rather than about this repo. The aiohttp bump is deliberately NOT made here: dependabot PR #27 has carried exactly it since 2026-08-07 and is mergeable, so a duplicate commit would conflict with an open PR for the same result. **Method, because getting it wrong is easy and silent:** audit the *lock export* with `--extra all --extra dev` (a bare export emits 42 packages and zero advisories, which looks like good news), pass `--no-deps` (an already-pinned file must not be re-resolved), and run it on ubuntu with the representative Python (uv 0.11 has no `--python-platform` on `export`, so pip evaluates the universal file's markers for whatever interpreter it runs on -- the runner IS the targeting). Auditing the installed environment instead measures a developer's machine, not what the project declares. **Next action:** merge the fix backlog, then flip `continue-on-error` off with the two structural pins in `--ignore-vuln`. **Status check, corrected (2026-09-12):** the earlier same-day note above ("real pytest/sklearn-matrix failures, not staleness") was wrong -- checked without reading the run's own date. `gh pr checks 27` names run `31141643773`, created `2026-08-07T02:34Z`; the PR's `updatedAt` is `2026-09-08`, so its CI has not re-run in over a month, against a `master` five weeks and dozens of unrelated fixes stale (the vulture failure inspected is exactly the kind of finding L1.4's later pin round closed). `mergeable` reads `UNKNOWN`. This is a stale-CI PR, not a red one -- it needs a `@dependabot rebase` (or equivalent re-trigger) to get a real signal, not a diagnosis of failures that no longer exist. Not triggered here: commenting on someone else's PR/bot command is the owner's call, not something to do unprompted mid-audit. |
| L3.4 | `pytest --doctest-modules` on a scoped subpackage | **RESOLVED** | Landed exactly as planned: a new step inside L2.1's existing `runtime-warnings` job in `deep-nightly.yml`, not a second leg. `pytest --doctest-modules src/mlframe/calibration src/mlframe/metrics` (advisory). Neither subpackage has a doctest example yet, so the step's real job today is exit-code hygiene: `--doctest-modules` returns 5 ("no tests collected") on an empty tree, which is not a failure worth a red advisory pill -- folded into success explicitly (`|| test "$?" = 5`) rather than left to fail. |
| L3.5 | Spend the existing `hypothesis` dep on numeric invariants | **DOC** | Not a tool change and not a gate: it is a standing suggestion for how to write the next tests over the MI/discretization/rank-correlation surface. Recorded so it is not re-proposed as new. |
| L3.6 | Explicitly NOT recommended (pylint, flake8, pyright, xenon/radon, safety, refurb) | **DOC** | Agreed, with the measurements the agent gives. |

### Ceremony to remove

| # | Finding | Disposition | Notes |
|---|---|---|---|
| L4.1 | `.semgrep.yml`: 2 advisory rules, one duplicating a blocking gate | **RESOLVED** | `broad-except-silent-swallow` confirmed a genuine duplicate: `pyutilz.dev.code_audit`'s `broad_except_swallow` check is already blocking-gated via `test_code_audit_baseline.py` (runs every registered check, no `checks=` narrowing). `module-global-write-via-reexport-alias` ported to `test_module_global_write_via_reexport_alias.py` -- an AST scanner resolving `from . import X as alias` module aliases and flagging `alias.CONST = value` writes where the imported module only re-exports `CONST` rather than defining it, same scope as the semgrep pattern (private ALL_CAPS const, private lowercase alias), same `# nosemgrep: module-global-write-via-reexport-alias` suppression marker honoured for continuity. Verified with a live synthetic fixture (facade re-export vs. owning-module write) before trusting it, not just "runs green on current code" -- confirmed it flags the wrong-target write and stays silent on the correct one. `.semgrep.yml` deleted along with its pre-commit hook, `lint-advisory.yml`'s `semgrep-config-path` input, the `semgrep` dev dependency, and its `deptry` ignore-list entry. |
| L4.2 | `[tool.importlinter]`: 2 contracts, advisory, and its own config says so | **RESOLVED** | Verified against the tree (2026-09-12), disposition corrected from PARTIALLY RESOLVED: the section, its pre-commit hook, and the `import-linter` dev dependency are already gone -- `pyproject.toml`'s comment at the old `[tool.importlinter]` location records the removal and names the two meta-tests (`test_no_sibling_backimport_to_facade.py`, `test_no_inbound_edge_to_benchmarking.py`) that replaced it. The mechanical follow-up this row's "Next action" asked for had already landed by the time this correction was made. |
| L4.3 | The nine empty baselines' machinery | **REJECTED** | The agent states the counter-argument itself and it wins here: the uniform shape is what lets the next regression be baselined in one command, and L1.13's new consistency gate makes the uniformity load-bearing rather than decorative. |
| L4.4 | `[tool.pydoclint]`: advisory with ~200 known-residual findings | **RESOLVED** | The residual count had grown to 1066 by 2026-09-12 (the ~200 figure was measured 2026-08-23, before this session's other work), too large to fix outright and too large to trust a bare "flip to blocking" against. Baselined instead, the same pattern every other `_*_baseline.json` in this directory uses: `tests/test_meta/test_pydoclint_baseline.py` runs `pydoclint src/mlframe`, parses its `path:line:code` findings, and fails only on one absent from `_pydoclint_baseline.json` (1066 entries, frozen 2026-09-12) via `py_ci_shared.baseline_ratchet.Baseline` -- reused rather than hand-rolled, since it is exactly this diff-against-frozen-set logic already shared with glossum/flutter_app_core. `--refresh-pydoclint-baseline` registered in `conftest.py`; teeth-verified by deleting one accepted entry and confirming the test fails, then restoring it. This is a real BLOCKING gate today (any new docstring/signature mismatch fails the build immediately) even though the existing 1066 are not yet fixed -- the ~200-vs-blocking tradeoff the finding described is resolved by baselining, not by picking one side of it. |
| L4.5 | `rev:` pins on `language: system` hooks are noise | **RESOLVED (the substance)** | The `rev:` stays, because the upstream repo entry is what dependabot updates. What was missing is the runtime assertion, and that now exists (L1.2). |

### Could not check

| # | Item | Outcome |
|---|---|---|
| — | Branch protection / required checks — "worth verifying first, the highest-impact item in this document" | **CHECKED.** master requires exactly three: "CI required checks", "mypy-full / mypy-full", "black / black". So `black-filtered` and `mypy-full` **are** required in their own right, and the agent's worst case does not hold. It also raises D2's severity and reverses W3's justification — both handled above. `enforce_admins` is off and `required_linear_history` is off; force pushes and deletions are blocked. |
| — | Empty-scan fail-closed: ~50 of 56 file-scanning gates have no guard | **RESOLVED (as a ratchet)** | Confirmed: 57 gates glob a tree, 2 already carried a floor, 55 did not, and `test_no_file_over_1k_loc.py` used `pytest.skip` — green in CI. Added `tests/test_meta/_scan_guard.py` with the shared assertion, and wired it into that gate plus the new ones. The remaining 53 are NOT edited blind: their scan shapes differ and there is no way to prove a mechanical edit preserved each gate. Instead `test_scanning_gates_fail_closed.py` holds the line — a new scanning gate must carry a guard, the list may only shrink, and a drained or deleted entry must leave it. **Next action:** drain the 53 as each file is touched for other reasons. |
| — | `py-ci-shared`'s reusable workflows read from a local checkout that may differ from the pinned ref | **DOC** | Now less of a risk in one direction: the reusable workflows are SHA-pinned (D6), so what CI resolves is fixed. L1.1's re-confirmation against that exact ref is still outstanding and is part of L1.1's next action. |
| — | GitHub org secret push protection | **UNRESOLVED** | Not visible through the API available here. Feeds L1.5. |

---

## Cross-agent duplicates

- Agent 1 W3's justification and Agent 3's branch-protection question are the same fact seen from two sides. Resolved once, by checking: the protection exists, which made W3 unsafe as proposed.
- Agent 2's incidental `codecov.yml` finding is the same stale-comment class as Agent 1's D1 secondary note about the merge comment reasoning only about python versions. Both fixed.
- Agent 1 G6 (actionlint) and Agent 3's tool inventory disagree about whether actionlint runs. Agent 3 is right; it is a pre-commit hook.
- Agent 1 G1/G2 and Agent 3 L3.3 are three views of "the declared dependency set is unverified". G2 shipped; G1 and L3.3 are FUTURE with distinct next actions.

## Escalation

| # | Item | Status |
|---|---|---|
| X2 | `test_scalar_njit_nan_precheck_actually_parallelizes` spawns `python -c` without handing it the tree under test, so the subprocess imports whatever mlframe is installed. In a worktree that fails outright (`ModuleNotFoundError`); where an installed copy exists it would silently assert against code the suite is not testing. Found while measuring L2.1. | **RESOLVED.** The subprocess now gets `PYTHONPATH` derived from the `mlframe.__file__` this process imported. Verified failing before, passing after. |
| X4 | `calibration.html`, a rendered plotly document, was tracked at the repository root. It is a test artifact: some reporting test renders to a bare filename rather than `tmp_path`, so it lands in the working directory, looks like an ordinary untracked change, and gets swept in by `git add -A`. **It reached master in commit `bd8e212f4` of this very round -- my own commit, about linter configuration.** Noticed only because a later run under a different plotly version rewrote it and the diff surfaced again. | **RESOLVED (2026-09-12).** Found the producer by working backward from the literal string: `tests/metrics/test_show_calibration_plot_plotly_no_consumer.py::test_plotly_backend_default_show_plots_still_returns_none_without_crashing` calls `show_calibration_plot(..., backend="plotly")` with every OTHER argument at its default. `_calibration_plot.py`'s DSL-render branch fires on `backend == "plotly" and (plot_file or show_plots)` -- `show_plots` defaults `True`, so a bare call with no `plot_file` reaches it, and derives the save path as `_root or "calibration"` when `plot_file` is empty, writing `calibration.html` into whatever the CWD happens to be (the repo root, under pytest). The matplotlib branch a few lines below already has the correct guard for exactly this ("`show_plots` alone, no real `plot_file`, no interactive session -> no-op"); the plotly branch never had it. Added the same interactive-session check to the plotly branch's condition (an explicit `plot_file` still always forces a save, matching every other branch). Verified directly: reproduced the stray write pre-fix, confirmed it stops post-fix, added a regression test that asserts NOTHING is written to a `tmp_path`-backed cwd on the exact default-args call, verified failing before the fix (temporarily reverted, reran, restored) and passing after. Full `tests/metrics -k calibration` suite (89 tests) passes unchanged. |
| X1 | The distribution name `mlframe` on PyPI belongs to an unrelated 2020 project (Sam Stoltenberg, `github.com/skelouse/mlframe`, last release 0.1.15 on 2020-12-18). `release.yml`'s Trusted Publishing upload can never succeed under it, and `pypi/v/mlframe` would render a stranger's version. `pyutilz` is separately unpublished (404). | **DEFERRED by the owner (2026-09-08).** Nothing in CI to fix. The README now states the situation rather than "not published yet". P4-a is deferred behind this. |

---

## Round 2 — what the agents missed, found by acting on G1

Recorded separately because these are gaps in the **review**, not in the repo. Both come from the same
root: all three agents read only this repository. The sibling repos `pyutilz` and `py-ci-shared` carry
already-paid-for experience with exactly the problems G1 touches, and none of it reached the reports.

### M1 — W5 and G1 are the same finding, and the tracker repeated the mistake

Agent 1 filed **W5** (the pinned pyutilz SHA is hand-copied into seven call sites, "keeping seven copies in
sync by hand is exactly the drift the action was extracted to prevent") and **G1** (no lockfile) as
unrelated rows. They are one problem seen twice: the hand-copied SHA exists *because* there is no lock. A
lockfile records the resolved commit machine-side, so the SHA leaves `pyproject.toml` and the workflows
entirely, and `uv lock --upgrade-package pyutilz` becomes the whole update procedure.

My tracker propagated the error: W5's disposition was "extend `install-pyutilz` to cover the CUDA cases,
then convert all of them" — treating the symptom, adding machinery to keep copies in sync rather than
removing the copies. **Corrected:** W5's real fix is G1, and W5's next action is now "retire the
`pyutilz-ref` inputs once CI installs from the lock", not "extend the composite action".

The user caught this, and was right to: the SHA-in-`pyproject.toml` was never an agent recommendation. It
is this repo's own convention, recorded in the dependency comment, adopted after an unpinned same-day
pyutilz commit silently changed a shared helper's default and broke `MRMR.fit`. That incident is an
argument FOR a lock, not for hand-pinning: with no lock, every install took whatever `master` was at that
moment. The lock is the fix; the hand-written SHA was the stopgap for not having one.

### M2 — the constraint that decides HOW this can be done was already written down next door

`pyutilz/requirements-dev.txt` documents, from experience:

> A PEP 440 direct reference (`name @ git+https://...`) in project metadata makes the built distribution
> unuploadable -- PyPI answers `400 ... Can't have direct dependency`, and `twine check` passes it
> silently -- so it must stay out of `[project.optional-dependencies]` entirely.

No agent looked, so no report mentioned it, and it is the binding constraint on any "just add the git URL"
approach. Two consequences, both acted on:

- **It does not block the chosen design.** `[tool.uv.sources]` is uv configuration, not project metadata.
  Verified rather than assumed: built the wheel and read its `METADATA` -- `Requires-Dist: pyutilz>=1.0.0`,
  a plain specifier, no direct reference. The source is stripped from the distribution.
- **It exposed a live defect in this repo** -- see X3.

### X3 — `py-ci-shared` is a direct reference inside `[project.optional-dependencies].dev`

**MEASURED**, by reading the built wheel's `METADATA`:

```
Requires-Dist: py-ci-shared @ git+https://github.com/fingoldo/py-ci-shared.git@915217a4... ; extra == "dev"
```

This is precisely the shape `pyutilz` documents as making a distribution unuploadable, and `twine check`
(which `release.yml` runs, `ci.yml` too) passes it silently -- so nothing in CI would ever report it. It is
a second, independent blocker on publishing, unrelated to the name being taken (X1).

**Disposition: RESOLVED** -- moved to `requirements-dev.txt`, mirroring `pyutilz`'s own fix for the same
problem, with the install sites updated to pass `-r requirements-dev.txt`.

### M3 — a universal lock over a version RANGE silently resolves to the oldest-supported release

Not a review miss, a finding from doing the work, recorded because it is the trap anyone repeating this
will hit. With `environments` given as one expression spanning 3.9-3.14, uv minimises forking and prefers a
version valid across the whole span: `torch` resolved to **2.0.1**, the last release with cp39 wheels, for
3.10 through 3.14 as well. `uv lock` reported success; `uv sync` then failed on 3.13 with "no wheel for the
current platform". One entry per minor version makes each fork take the newest release that version
supports. A lock that resolves is not a lock that installs -- sync it before trusting it.

### G1 — no lockfile / no pinned resolution

**Disposition changed from FUTURE to RESOLVED.**

`uv.lock` is committed: 595 packages, resolved for Python 3.9.2-3.14 across the three CI platforms.
Everything below was found by the lock refusing to resolve, and each is a real defect in the declared
metadata rather than an artefact of locking:

| What refused to resolve | Why it is a real defect | Fix |
|---|---|---|
| `pyutilz>=1.0.0` | Not on PyPI at all, so no registry resolver can satisfy it -- the reason this project had no lock | `[tool.uv.sources]`, branch `master`, exact commit recorded in `uv.lock` |
| `deptry==0.25.1`, `mypy==2.1.0`, `yamllint==1.38.0` | All three require >=3.10 while the project supports 3.9; pins added earlier in this round, uninstallable on the 3.9 leg | `; python_version >= '3.10'`, gated like `black` already was |
| `cryptography>=48.0.1` (via `[mlflow]`) | Declares `>3.9.0,<3.9.1 \| >3.9.1` -- it refuses exactly 3.9.0 and 3.9.1, which `requires-python = ">=3.9"` includes | Lock universe starts at 3.9.2; `requires-python` untouched, so pip on a real 3.9.0 is unaffected |
| `cupy-cuda11x` in `[gpu-cuda11]` / `[transformer_gpu_cuda11]` | The extras' own comments say "there is NO cuda11x wheel for Python 3.13+", but the dependency lines carried no marker: the metadata claimed support the prose denied | `; python_version < '3.13'` on both |

### The pyutilz commit the lock records

Locking by branch moved pyutilz from the hand-pinned `ec016b15` to branch HEAD `3a04d38d`, **33 commits
ahead**. That is a behaviour change, not a packaging one, and this repo has been broken by exactly that
before (the njit-contract incident). It was validated rather than assumed: 2017 tests from
`tests/metrics`, `tests/calibration` and `tests/feature_selection/filters` -- the surface that exercises
the njit contract -- run inside the locked environment against the new commit. **2009 passed, 13 skipped,
0 failed.**

### What the repo's own gates caught in this work

Worth recording because it is the gates doing their job on my changes, not on someone else's:

* `test_no_undeclared_continue_on_error` failed on all three `continue-on-error` steps added earlier this
  round (consumer-position mypy, the RuntimeWarning census, dep-floors). It requires each to be named in
  an allowlist with an argument for why it must never block. All three are now declared with that
  argument, and two carry the condition under which they stop being advisory.
* `test_f4_dependabot_pip_ecosystem_reenabled` failed on the pip -> uv switch. Re-framed to the invariant
  it actually protects -- this repo keeps an automated security-patch signal at all -- rather than to the
  literal string `pip`, since `dependency-review` only inspects new deps in an incoming diff and
  `pip-audit` is advisory and opens nothing.

**Next action:** wire the CI test installs to `uv sync --frozen` rather than resolving fresh, then retire
the `pyutilz-ref` inputs that W5 counted.

---

## Correction to commit 883f0c2db, and X5

### The `signal` justification recorded in 883f0c2db is wrong

That commit says restricting `--timeout-method=signal` to Linux was needed because generalising it to
macOS "brought back `Fatal Python error: Aborted` on macOS shards -- ... and this repo had already driven
those crashes from 375 to zero."

**Measured after the fact, and the claim does not hold.** The CI run immediately before this round's first
commit -- run 34236387160 on `72b4db410` -- shows **25 aborts** on `pytest 3.12 on macos-latest (shard
1/10)`. The crashes were not at zero when the round began, so the `signal` change did not bring them back.
The run after the fix (34295381760 on `d29a0b627`) shows **13** on the same shard: fewer than the
pre-round baseline, not more.

The change itself stands and should not be reverted: the finding it implements named Linux specifically,
and SIGALRM into a thread running native numba work is a real hazard on macOS regardless of whether it
caused these particular crashes. What was wrong is the reasoning written into the permanent record. The
commit is pushed and shared history is not rewritten here, so the correction lives in this file instead.

Method note, since this is the second time in this round the same mistake shape appeared: "the repo had
driven X to zero" came from a session summary rather than from a measurement. A claim about the state of
CI is checkable in one API call, and was not checked before being committed.

### X5 -- macOS shards abort in test_biz_val_training_core, and have since before this round

| | |
|---|---|
| Signature | `Fatal Python error: Aborted`, xdist workers `gw0`/`gw1`/`gw2` crash |
| Tests | `test_biz_val_training_suite_classification_completes`, `..._regression_completes`, `..._mlframe_models_subset[model_list0]` |
| Count | 25 aborts at `72b4db410` (pre-round), 13 at `d29a0b627` |
| Platform | macOS only; the same shard passes on ubuntu and windows |

**Disposition: OPEN, not caused by this round.** The 25 -> 13 movement is not attributed to anything
specific: several changes landed between those runs and no experiment isolates them, so treat both numbers
as observations rather than as a before/after.

This is the documented macOS numba hazard -- entering a `parallel=True` kernel concurrently from more than
one thread aborts the process there while Linux tolerates it. This round added
`_numba_parallel_guard.py`, the `_nested_parallel_scan` walker and a gate for the class, and those found
zero unguarded reachable paths; the surviving crashes are therefore either reached through a path the
walker cannot see (a dispatch table, a joblib worker boundary) or are a different mechanism wearing the
same signature.

**Next action:** run shard 1's three biz_val tests on macOS alone, with `-p no:xdist`, to establish whether
the abort needs concurrency at all. That single fact splits the remaining possibilities in half and is one
CI dispatch, not an investigation.

### X5, round 1 result: the crash does not need concurrency, and there are probably two of them

Probe run 34297712463, two legs over the same three tests, nothing varied but xdist.

| Leg | Result |
|---|---|
| serial (no xdist) | `Fatal Python error: Segmentation fault`, exit 139. **Zero** aborts, **zero** OpenMP errors. Dies immediately after `collected 5 items`, before the first test reports. |
| xdist (control) | 3 segfaults, plus `OMP: Error #179: Function pthread_mutex_init failed` twice -- a line the serial leg never emits. |

**This closes the question the round asked.** A single pytest process, no xdist, crashes the same way. So
the hypothesis this round invested in -- concurrent entry into a `parallel=True` numba kernel from separate
xdist workers -- does not explain the crash that matters. `_numba_parallel_guard.py`, the
`_nested_parallel_scan` walker and their gate remain worth having: they close a real class, and the class
is real regardless. They simply are not what is killing these shards, and the earlier framing that treated
"macOS aborts" and "nested parallel entry" as the same subject was wrong.

**It also suggests two distinct failures rather than one.** The segfault reproduces without concurrency;
the OpenMP error appears only when several processes each bring up a runtime, which reads as resource
exhaustion rather than as the same fault. Treating them as one cause is the mistake this round already
made once.

### X5, and the framing was too narrow again

Checked which tests the OTHER failing macOS shards die on, in run 34297709357:

| Shard | Signature | Tests |
|---|---|---|
| 1 | aborts | `test_biz_val_training_core.py` -- three training biz_val tests |
| 2 | 2 segfaults, 0 aborts | `test_biz_val_weak_family_adversarial.py` -- feature selection |
| 8 | 3 aborts + 3 segfaults | `test_composite_lazy_prebin_memory.py`, `test_composite_streaming_update.py` -- composite cache |

**Unrelated modules, mixed signatures.** This is not three biz_val tests, and scoping the probe to one file
was the same too-narrow framing as treating the crash as the nested-parallel class. The runner is
`Image: macos-26-arm64` -- `macos-latest` is macOS 26 on Apple Silicon, and the whole native stack
(numba/llvmlite, catboost, lightgbm against libomp, sklearn's OpenMP) is running there.

So the subject is the platform, not any test. That is worth stating plainly because it changes what a fix
would even look like: pinning a runner image, pinning a native dependency, or accepting macOS as
non-blocking until the stack settles are all platform decisions, and none of them is findable by reading
mlframe's own code.

**Round 2, rewritten before dispatch:** the `one-by-one` leg keeps its value -- knowing whether the crash
predates the first test still splits the possibilities -- but the next probe after it should not be about
these tests at all. It should import the heavy native stack on macOS and run one trivial fit per library,
with nothing of mlframe's involved, to find which library is unstable on ARM64. If none of them crashes
alone, the subject is their combination, which is the classic duplicate-OpenMP-runtime problem and has a
known shape.

**Not fixed here, and not fixable in a commit:** whether mlframe blocks its own CI on a platform whose
native stack is currently this unstable is the owner's call, not a defect to close. Recorded so the choice
is made deliberately rather than by leaving two shards permanently red.

### X5, round 4: test the architecture directly

The comparison that looked available was not. On `72b4db410`, the run before this round, **nine of ten**
macOS shards were cancelled and exactly one finished -- and it failed. So "1 red then, 6 red now" compares
one completed shard against six; the honest statement is that **every macOS shard that has finished, in
either run, is red**, while ubuntu and windows are clean on the same commit.

That is still the strongest evidence available for a platform cause, and it does not depend on reading a
single log: one codebase, three platforms, one of them failing across unrelated modules.

`macos-latest` is now `macos-26-arm64`. The probe gains two legs on `macos-15-intel` -- the same serial
run and the same native-stack sweep, on x86_64. The outcome is decisive either way:

* Intel passes -> the subject is the Apple Silicon migration, and the remedy is a runner label, not a
  change to mlframe.
* Intel crashes too -> the architecture is exonerated and something else in the macOS environment is at
  fault, which redirects the search rather than ending it.

Each leg now prints `uname -m` and the CPU brand, so a result can never be attributed to the wrong
architecture -- the mistake this entry was itself about.

### X5, corrected again: "the platform is broken" was wrong, and the user said so

The owner pushed back -- macOS had been nearly green the day before -- and the pushback was right. Job
status is not test status. Looking inside the shards of run 34297709357:

| Shard | Tests |
|---|---|
| macos 2 | **157 passed**, 1 failed |
| macos 4 | **1352 passed**, 4 failed |
| macos 9 | **396 passed**, 2 failed |
| windows 7 | **4166 passed**, 3 failed |

A job goes red on one crashed xdist worker regardless of how much passed around it. Calling this "полный
отказ платформы" was wrong and alarmist: the failure is a handful of specific tests, not a platform.

That is the fourth time in this round the same mistake shape appeared -- taking a number without checking
what it counts. The others: "the repo drove aborts to zero" (from a summary, not a measurement), "1 red
then vs 6 now" (comparing against a run whose other nine shards were cancelled), and "zero FAILED lines"
(a grep artifact). The correction each time cost one command.

### The nested-parallel gate found a live path, and disagrees with itself across machines

`test_no_unguarded_nested_parallel` failed on Windows CI with a real finding:

    _pairs_core.py::check_prospective_fe_pairs -> _build_shuffle_matrix
      (_batch_mi_with_noise_gate_gpu -> batch_mi_with_noise_gate_cuda_resident -> _resident_y_all_device)

Verified from the source: `check_prospective_fe_pairs` runs a thread pool, `_build_shuffle_matrix` is
`@njit(parallel=True)`, and the call chain exists. **Fixed** -- `parallel_kernel_entry()` is now held
around both `_build_shuffle_matrix` call sites in `batch_mi_noise_gate_gpu.py`, at the kernel rather than
at the dispatcher so the serialisation covers the kernel call and not the surrounding GPU work.

This also revives the hypothesis the previous entry declared dead. `test_concurrent_real_fits_no_exception_and_bounded_cache`
-- a test explicitly about concurrent fits -- is among the crashing tests on macOS. The serial segfault
found in probe round 1 is real and needs no concurrency, but it is not the only mechanism, and writing off
the guard/scanner line was premature.

**Open, and a defect in this round's own work.** Sharpened by one more data point: Windows CI reports the
path, **Ubuntu CI does not**, and neither does this machine -- all three on the identical tree. So it is
not "my machine versus CI"; it is platform-dependent inside the walk itself.

A path-sorting explanation was proposed for that and then **measured and rejected**: `self.where[name]`
comes from `sorted(root.rglob("*.py"))`, and sorting `WindowsPath` orders on backslashes while `PosixPath`
orders on forward slashes -- but across all 153 names with more than two definitions, the first pair is
identical under both orderings. So sort order is not the mechanism, and the cause of the platform
difference is still unknown.

The `[:2]` truncation remains a real defect on its own terms -- the walk follows two definitions of a name
out of however many exist, so it is incomplete by construction and its blind spot moves as files are
added -- but it has not been shown to be what makes the verdict differ per platform.

The scanner reports 0 unguarded paths on this machine and 1 on CI, from an identical tree. A blocking gate that answers differently on two machines cannot be
trusted in either direction, and the "0 unguarded paths" this round reported as a result was never
trustworthy. One concrete cause is visible without explaining the whole discrepancy:
`reachable_kernels` follows only `self.where[callee][:2]` -- two definitions of a name out of however many
exist -- so the walk is incomplete by construction and its blind spot moves when files are added.
**Next action:** drop the `[:2]` truncation, measure the scan cost without it, and only then trust a zero.

### X5, round 4 results (partial): the native stack is clean on ARM64, Intel could not even install

Probe run 34317667601.

**`native-stack` on `macos-latest` (arm64): every check passes.** numpy, scipy, sklearn-with-OpenMP,
numba serial, numba `parallel=True`, lightgbm, catboost, xgboost, torch, joblib-threading, and the
combined import of all of them together -- all OK, no crash. **3 copies of libomp/libiomp are loaded into
the one process at once**, and nothing crashes from that alone. This narrows the earlier hypothesis: a
duplicate-OpenMP-runtime import is not, by itself, fatal on this runner. Whatever crashes the real test
suite needs more than importing the stack -- concurrent load, a specific fit shape, or pytest/xdist's own
process model.

**`native-stack` on `macos-15-intel` (x86_64): failed before running anything.** `uv` could not build
`llvmlite==0.49.0` from source -- no prebuilt wheel for this platform/version, and the runner has no LLVM
for CMake's `find_package` to find. This is an installation gap in the probe, not a finding about the
crash: the Intel leg never got far enough to test the hypothesis it was dispatched to test.

**`serial` on `macos-15-intel`: did not run either.** Same installation failure -- `llvmlite==0.49.0`
has no macOS x86_64 wheel at all (checked directly against PyPI's file listing: 0.49.0 and 0.48.0 both
publish `macosx_..._arm64` wheels only, no `x86_64`, no `universal2`). This is not a probe misconfiguration
to fix and retry: **the current numba/llvmlite pin cannot be installed on an Intel Mac at all**, so the
architecture question this round asked has no answer through this route. Getting one would mean pinning
numba/llvmlite down to the last release with an x86_64 wheel, which is a separate, real decision (an older
numba across the whole test matrix) and not part of this probe.

**`one-by-one` on `macos-latest`: decisive.** `--collect-only` succeeds (5 tests collected, no crash --
rules out import/collection as the trigger). Then each of the three tests, in its own interpreter, with no
sibling process at all: **all three crash**, every one with `Fatal Python error: Segmentation fault`,
every one after `collected 1 item` (so inside the test body, not at collection). All three call
`train_mlframe_models_suite(..., mlframe_models=["lgb"], ...)` with a small (400-row) real LightGBM fit.
This is now a concrete, single-test repro rather than "the shard crashes somewhere."

**`serial` and `xdist` on `macos-latest`: reproduce round 1 exactly** -- segfault with no xdist, 3
segfaults + `OMP: Error #179: pthread_mutex_init failed` twice with it. Not a one-off: same signatures on
a second, independent run.

### X5, round 4 conclusion

The architecture question is unanswered because the current numba pin cannot even install on Intel macOS
-- not because Apple Silicon was cleared. What round 4 DID settle: the native stack survives import and
basic use on ARM64 completely cleanly (11/11 libraries, 3 simultaneous libomp copies, no crash), so the
crash is not "this environment cannot run this stack at all" -- it needs the specific path
`train_mlframe_models_suite` takes with LightGBM. **Next action:** reproduce
`test_biz_val_training_suite_classification_completes` locally or in a minimal script (the suite call plus
its inputs, no pytest) with `faulthandler`/`lldb` attached, since the pytest traceback stops at
`threading.py` frames -- a Python-level trace cannot see further into what is a native crash.

### X5, round 5 results

**`lldb` (first attempt, dispatch 34318603163): a real defect in the probe, not a finding.** The captured
trace showed only `dyld_start` -- lldb stops on every `exec` event, and `python -m pytest` re-execs
through its console-script entry point before reaching the test, so `thread backtrace all` fired at that
startup stop rather than at the crash. Fixed (commit 04e750f14): invoke python directly
(`python -c "import pytest; sys.exit(pytest.main([...]))"`, one process, one exec) and `continue` past
non-fatal stops. Not yet re-dispatched.

**`env-mitigation` (first attempt): all five variants crashed identically.** `KMP_DUPLICATE_LIB_OK=TRUE`,
`OMP_NUM_THREADS=1`, both together, and `NUMBA_NUM_THREADS=1` -- every one, same
`Fatal Python error: Segmentation fault`, same exit 139. The leg reported as a job "success" only because
each variant's failure was individually caught with `|| echo`; the job's own exit code does not reflect
that every variant crashed. **Real finding underneath, though**: one of the traces, for the first time,
showed a Python frame inside the crashing thread itself rather than only `threading.py` in background
threads --

    lightgbm/basic.py:2301 in __init_from_np2d
    lightgbm/basic.py:2170 in _lazy_init
    lightgbm/basic.py:3758 in __init__

`__init_from_np2d` calls `_LIB.LGBM_DatasetCreateFromMat` via ctypes -- LightGBM's C++ entry point for
building a `Dataset`, which OpenMP-parallelises its own histogram binning using LightGBM's OWN
`num_threads` setting, not the process's `OMP_NUM_THREADS` env var. That is why the env-var mitigations
could not have worked: none of them reach the parameter that actually controls this. And
`src/mlframe/training/lgb_shim.py:561-563` explicitly sets `self.n_jobs = os.cpu_count()` before every
fit (to skip LightGBM's slow core-count probe), so every LightGBM fit in this codebase is multi-threaded
by construction, on every platform -- this is not incidental to the test, it is how the shim always
behaves.

**Round 5b, dispatched:** three checks against bare LightGBM, no mlframe involved at all, to settle
whether `n_jobs > 1` on this exact library on this exact runner is sufficient by itself: a single fit at
`n_jobs=1`, a single fit at `n_jobs=os.cpu_count()`, and 50 repeated fits at `n_jobs=os.cpu_count()`
(single-fit runs would not settle instability -- round 1's serial leg died on the very first collected
test, other legs ran further first, so the crash is not obviously 100%-reproducing on every call).

### X5, ROOT CAUSE FOUND (round 5b, dispatch 34319381472)

**`lldb`, with the fixed script, captured the actual crash this time**, not `dyld_start`:

    thread #19/#20, stop reason = EXC_BAD_ACCESS (code=1, address=0x580)
      frame #0: libomp.dylib`__kmp_suspend_initialize_thread + 32

Full call chain from Python down (`bt all` on the main thread at the same stop):

    Python (ctypes) -> LGBM_DatasetCreateFromMat -> LGBM_DatasetCreateFromMats
      -> LightGBM::DatasetLoader::ConstructFromSampleData
        -> __kmpc_fork_call / __kmp_fork_call  (LightGBM opens an OpenMP parallel region)
          -> __kmp_invoke_task_func -> __kmp_invoke_microtask
            -> BinMapper::FindBin -> FindBinWithZeroAsOneBin -> GreedyFindBin
              [CRASH in libomp.dylib itself, initialising a suspended worker thread]

This is **libomp's own thread-pool-initialisation code faulting**, not LightGBM's and not mlframe's. The
`env-mitigation` results support the same conclusion from the other direction: a bare LightGBM fit at
`n_jobs=os.cpu_count()`, 50 repeats, no mlframe, no numba -- 50/50 clean, no crash. So a single LightGBM
fit under load is not sufficient by itself either; something about the process state (thread count
already elevated, a specific `libomp` version, or genuine flakiness in that version's suspend path) is
also load-bearing, and this repo's own `ci.yml` and this probe both install `libomp` via a bare
`brew install libomp` with **no version pin**. The version pulled in this run:

    Pouring libomp--23.1.0.arm64_tahoe.bottle.tar.gz

23.1.0 is a very recent LLVM line (23.x was not yet a stable LLVM release series as of this repo's last
audit), on `arm64_tahoe` (macOS 26). A version-pinned, older, longer-soaked `libomp` is now the leading,
concrete, testable fix candidate -- not a code change to mlframe or to LightGBM's call site, a build
dependency pin.

**Next action, refined after checking:** Homebrew's formula API only serves the CURRENT stable version
(23.1.0, same as what crashed) -- there is no `libomp@<version>` versioned formula the way there is for
e.g. `python@3.12`. Pinning an older `libomp` means either (a) checking out an older commit of
`homebrew-core`'s `libomp.rb` and building/installing from that specific revision, which is slow (a
from-source build) but exact, or (b) installing `libomp` from conda-forge instead of Homebrew, since
conda-forge keeps every past version installable by exact number and mlframe's CI already uses `uv`
alongside real package managers elsewhere. (b) is very likely the faster path to a testable pin. This is
an infrastructure decision (which package channel a CI dependency comes from) and is left for the owner
rather than switched unilaterally; both `ci.yml`'s real leg and this probe currently share the same
unpinned `brew install libomp` call site, so the fix (once chosen) is one line, applied twice. If the pinned version does not crash across enough
repeats, this is closed as a `libomp` bug worked around by pinning, filed upstream if a matching LLVM
issue is not already open. `KMP_DUPLICATE_LIB_OK` / `OMP_NUM_THREADS` env vars, tried earlier this round,
were never going to help -- they gate LightGBM's own OpenMP entry point, not a fault inside libomp's
internal thread-suspend bookkeeping.

### X5, round 6-8: `pinned-libomp` leg -- the Homebrew-pin hypothesis is REJECTED

Owner's explicit instruction was to test the Homebrew pin first, before conda-forge. Getting a version-
pinned `libomp` installed via Homebrew (which serves only the current-stable formula, no
`libomp@<version>`) took three attempts, each rejected by a different, newer Homebrew restriction than
the docs describe:

- Round 6 (dispatch `34320517526`): `brew install <raw.githubusercontent.com URL to an old libomp.rb>`
  -- rejected outright, `No available formula or cask with the name "<URL, lowercased>"`. Modern Homebrew
  does not accept a formula URL as the install target.
- Round 7 (dispatch `34322735114`): download the `.rb` file to a local path first, `brew install
  <local path>` -- also rejected, `Homebrew requires formulae to be in a tap`. An untapped formula file
  is not installable at all anymore, tapped or not, local or remote.
- Round 8 (dispatch `34323769162`): `brew tap-new local/libomp-pin`, copy the formula file into that
  tap's `Formula/` directory, `brew install local/libomp-pin/libomp` -- this is the form that actually
  works. Log confirms: `Bottle Manifest libomp (22.1.8)` ... `Pouring libomp--22.1.8.arm64_tahoe.bottle.tar.gz`
  ... `/opt/homebrew/Cellar/libomp/22.1.8: 11 files, 1.8MB`.

**With libomp 22.1.8 (2026-06-16, three months more soaked than the 23.1.0 that crashed in round 5b)
correctly installed, the SAME test still crashed with the SAME signature**:

    Fatal Python error: Segmentation fault
    Thread 0x0000000171977000 (most recent call first):
      File ".../threading.py", line 359 in wait
      File ".../threading.py", line 655

collected 5 items, crashed at ~80s in (07:28:14 collect, 07:29:35 fault) -- same shape as round 5b's
`EXC_BAD_ACCESS` in `__kmp_suspend_initialize_thread`, same test file
(`test_biz_val_training_core.py`), same `serial`-equivalent single-worker mode.

**Conclusion: the crash is NOT specific to libomp 23.1.0.** Pinning to an older, longer-soaked build does
not help -- this rules out the leading fix candidate from round 5b's writeup above. Either the bug is
present across a wider range of `libomp` versions than assumed (both 22.1.8 and 23.1.0 crash), or the
true trigger is something else entirely that a version pin cannot touch (a genuine race in
LightGBM's/OpenMP's thread-pool init that any recent `libomp` build hits under this exact process state).
conda-forge is very unlikely to fare differently for the same reason -- it would still be shipping a
comparably recent LLVM-derived `libomp` build, not a fundamentally different implementation. **conda-forge
is downgraded from "leading candidate" to "not expected to help, low priority to still try."** Next real
lead: a code-level mitigation (force `n_jobs=1` for LightGBM specifically on this platform, or find
whatever process state -- thread count, prior OpenMP activity -- actually triggers the fault and avoid
it), not a dependency-channel swap.

The three round-6/7/8 Homebrew syntax fixes are committed (`dbe088f40`, `50b3708ac`) and are worth keeping
in the probe even though the pin itself didn't help -- they're the only working documented recipe for
installing a specific historical Homebrew bottle version at all, reusable for any future version-pin
experiment on this or another dependency.

### X5, round 9 (dispatch `34361218930`): `n_jobs=1` mitigation shipped, REJECTED

Owner's explicit instruction after the libomp-pin rejection was to go after the code-level mitigation next.
Shipped (`120609841`): `lgb_default_n_jobs()` in `lgb_shim.py`, forcing LightGBM to `n_jobs=1` on darwin by
default (env-var escape hatch `MLFRAME_LGB_MACOS_ALLOW_MULTITHREAD=1`), wired into every LightGBM
construction site in the package (the dataset-reuse shim's own `fit()`, the central `LGB_GENERAL_PARAMS`
config used by `train_mlframe_models_suite`, and the three explicit `n_jobs=-1` sites in
`_trainer_configure.py`'s gated_outlier/bagging/composite estimators). Verified locally first: 5 new unit
tests pinning the resolution logic, the full existing 18-test `lgb_shim` suite, and the full 3486-test
`tests/training/composite/` suite all green; mypy clean.

**Re-dispatched the probe to confirm on real CI -- the crash still reproduces with `n_jobs=1` confirmed in
effect.** The `one-by-one` leg (each of the three known-crashing tests run alone, in its own pytest
process) segfaulted on all three, individually, with `n_jobs=1` active:

    === test_biz_val_training_suite_classification_completes ===
    Fatal Python error: Segmentation fault
    EXIT=139 for test_biz_val_training_suite_classification_completes
    === test_biz_val_training_suite_regression_completes ===
    OMP: Error #179: Function pthread_mutex_init failed:
    EXIT=139 for test_biz_val_training_suite_regression_completes
    === test_biz_val_training_suite_mlframe_models_subset[model_list0] ===
    Fatal Python error: Segmentation fault
    OMP: Error #179: Function pthread_mutex_init failed:
    EXIT=139 for test_biz_val_training_suite_mlframe_models_subset[model_list0]

The `serial` leg (macos-latest, arm64) crashed identically to every prior round, same
`Fatal Python error: Segmentation fault` in the same test file. The `macos-15-intel` `serial` leg failed too,
but for an unrelated, pre-existing reason (`llvmlite` 0.49.0 has no prebuilt wheel for Intel macOS and its
sdist build fails -- nothing to do with this investigation).

**New signal: `OMP: Error #179: Function pthread_mutex_init failed`.** This did not appear in any prior
round's logs. It surfaced twice, both AFTER at least one prior segfault in the same `one-by-one` job (the
regression test crashed second, after the classification test's segfault; the third test's log shows both
a segfault AND this OMP error). This is consistent with process-level corruption carrying over from an
earlier crash within the same job's shell loop -- each `pytest` invocation in `one-by-one` is a fresh
process, so it is NOT simply "prior Python state in the same interpreter," but the three invocations run
back-to-back on the same macOS runner and the mutex-init failure appearing only after a prior segfault (not
on the very first invocation) suggests OS-level pthread/libomp resource exhaustion or corruption left behind
by the crash, not a fresh independent fault.

**Conclusion: `n_jobs=1` (LightGBM's own declared thread count) does not control the code path that
crashes.** Round 5b's `bt all` showed the fault in `__kmp_suspend_initialize_thread` during
`DatasetLoader::ConstructFromSampleData`'s `__kmpc_fork_call` -- LightGBM's Dataset/bin-construction sampling
phase is a known case (documented in LightGBM's own issue tracker for other platforms) where thread count is
NOT always fully gated by the `num_threads` config at every internal call site; some early sampling paths can
still consult `omp_get_max_threads()` / the process-wide OpenMP default rather than the just-set config value.
Sklearn-level `n_jobs=1` on the estimator is therefore an insufficient lever -- it constrains the trained
booster's own parallelism, not necessarily the Dataset construction phase where round 5b's crash lives.

**Next lead, more direct: set `OMP_NUM_THREADS=1` (and/or `KMP_DUPLICATE_LIB_OK=TRUE`) as an actual OS
environment variable for the process, not just LightGBM's own `num_threads` param** -- this affects the
libomp runtime's own defaults at every call site, including whichever sampling path ignores LightGBM's
config. Round 5's `env-mitigation` leg tried `OMP_NUM_THREADS` before, but only against a bare, isolated
LightGBM fit (which never crashed even without this env var) -- it was never tried against the actual
failing full-suite test, so this combination is still untested. `lgb_default_n_jobs()`'s macOS branch should
additionally set `os.environ.setdefault("OMP_NUM_THREADS", "1")` (setdefault so an explicit user override
still wins) the first time it resolves on darwin, and the probe should re-dispatch once that's in place.

The `n_jobs=1` code change itself is not reverted -- it is still a correct, harmless constraint on LightGBM's
own declared parallelism (and may still matter in combination with the `OMP_NUM_THREADS` env var below) --
but it is REJECTED as a complete fix on its own.

### X5, round 10 (dispatch `34388816137`): `OMP_NUM_THREADS=1` env var shipped, ALSO REJECTED

Shipped (`2ae6693a0`): `mlframe/__init__.py` now runs `_autoconfigure_macos_omp_threads()`
unconditionally at `import mlframe` time (darwin-only), `os.environ.setdefault("OMP_NUM_THREADS", "1")` +
`KMP_DUPLICATE_LIB_OK=TRUE`, deliberately placed at the very top of the package's `__init__.py` so it runs
before any submodule -- including `lgb_shim.py` -- can be reached (Python always executes a parent
package's `__init__` before a submodule import). Verified locally: 4 new unit tests, mypy clean, no
regression in `test_cuda_autoconfig.py`/`test_colorama_reinit_patch.py`.

**Re-dispatched, and the `lldb` leg's live backtrace shows the IDENTICAL fault to round 5b, byte-for-byte**:

    * thread #19, stop reason = EXC_BAD_ACCESS (code=1, address=0x580)
      frame #0: 0x0000000103f9366c libomp.dylib`__kmp_suspend_initialize_thread + 32
    * thread #20, stop reason = EXC_BAD_ACCESS (code=1, address=0x580)
      frame #0: 0x0000000103f9366c libomp.dylib`__kmp_suspend_initialize_thread + 32

**Two threads hit the exact same instruction concurrently** -- if the OpenMP team had genuinely been
constrained to 1 thread by either lever, only one thread could ever reach this call. Both `n_jobs=1`
(estimator-level) and `OMP_NUM_THREADS=1` (process-level, confirmed set before any submodule import) were
active for this dispatch, and the crash is unchanged. The `serial` leg on `macos-latest` crashed identically
(same test file, same `Fatal Python error: Segmentation fault`), and `pinned-libomp` (which now also
carries both mitigations) failed too.

**Conclusion: neither of the two most direct thread-count levers actually reaches the code path that
crashes.** This means the Dataset-construction sampling phase's thread-team size (round 5b's
`ConstructFromSampleData` -> `__kmpc_fork_call` chain) is being determined by something neither
LightGBM's own config nor the process `OMP_NUM_THREADS` env var controls at the point this call fires --
plausibly a `#pragma omp parallel num_threads(N)` explicit clause inside LightGBM's own C++ that computes
`N` from `omp_get_max_threads()` BEFORE either lever's effect is visible to it (e.g. if some earlier
library in the same process -- sklearn's `_openmp_helpers`, numpy's BLAS backend, or numba's own thread
pool -- has already called `omp_set_num_threads()` with a value greater than 1, which per the OpenMP spec
overrides the env-var default for the rest of the process, and no amount of setting `OMP_NUM_THREADS`
afterward can undo that already-programmatic call).

**This closes out the two most direct code-level mitigations mlframe can apply without deeper native
instrumentation.** Two options remain, both a level below what a Python-side config or env var can reach:

1. Instrument WHICH library call, precisely, first sets a process-wide OpenMP thread count above 1 (e.g. an
   `lldb` breakpoint on `omp_set_num_threads`/`__kmp_get_hier_str`, or `KMP_SETTINGS=1` env var to have
   libomp itself log its resolved settings to stderr at first parallel-region entry) -- this would name the
   actual culprit library/call site precisely enough to either avoid it or override it downstream, rather
   than guessing at levers.
2. Route LightGBM's Dataset construction through `n_estimators=0`-then-native-`train()`-style avoidance of
   the specific sampling code path if one exists, or as a harder fallback, skip/xfail the three affected
   tests on macOS CI specifically with an explicit upstream-bug citation (this would be a genuine
   `xfail`-for-third-party-platform-limitation case per this repo's own fuzz/skip convention -- not a
   band-aid over an mlframe bug -- but has not been proposed to the owner yet and should not be applied
   without that discussion, since it reduces macOS test coverage rather than fixing the crash).

Owner has not yet been asked which of these two to pursue -- surfaced directly rather than guessing a third
blind mitigation attempt after two straight rejections. **Owner chose option 1: instrument, don't guess.**

### X5, round 11 (dispatch `34421591792`): CONCLUSIVE -- the crash is NOT gated by thread count at all

Shipped (`38e41b634`): `kmp-settings` probe leg + `scripts/macos_lgb_kmp_settings_probe.sh`, setting
`KMP_SETTINGS=1` + `OMP_DISPLAY_ENV=TRUE` (libomp's own diagnostic, prints every resolved ICV to stderr
the first time any parallel region opens) alongside `OMP_NUM_THREADS=1` + `KMP_DUPLICATE_LIB_OK=TRUE`,
against the actual failing test.

**libomp's own dump confirms the setting was genuinely active, repeatedly, throughout the run**:

    OPENMP DISPLAY ENVIRONMENT BEGIN
       _OPENMP='201611'
      [host] OMP_NUM_THREADS='1'
      ...
    OPENMP DISPLAY ENVIRONMENT END

    scripts/macos_lgb_kmp_settings_probe.sh: line 35:  2662 Segmentation fault: 11  pytest ...
    Fatal Python error: Segmentation fault

(Note: the job's own GitHub Actions status showed "success" here -- a bug in the probe script itself, not
a real pass: the script's last command is `echo "pytest exit=$?"`, which always succeeds regardless of
what pytest returned, so the step's exit code never reflects the segfault. The log is unambiguous: `pytest
exit=139` is printed immediately after `Fatal Python error: Segmentation fault`. Fixed for any future round
so a re-dispatch's job status is trustworthy without needing to read the log.)

**This is the decisive result.** `OMP_NUM_THREADS='1'` was confirmed, by libomp's own runtime, to be the
value in effect for the ENTIRE process -- not assumed, not inferred from a Python-side config, but printed
by libomp itself at the moment a parallel region opened -- and the exact same crash still happened. Round
10's `lldb` catch of two threads concurrently inside `__kmp_suspend_initialize_thread` combined with this
round's confirmation that the requested team size really was 1 rules out every thread-count-based
explanation entirely: this is not "some other library set a higher thread count first" (round 10's
leading theory) and it is not "LightGBM's sampling phase ignores num_threads" (round 9's theory). The fault
is inside libomp's own thread-pool bookkeeping/initialization machinery itself, and it fires even when the
requested team size is exactly 1 -- consistent with libomp still spinning up its own internal
monitor/bookkeeping thread(s) regardless of the configured worker count, and THAT initialization path being
broken on this platform (`arm64_tahoe` / macOS 26).

**Conclusion: this is a genuine upstream libomp defect, not reachable by any mlframe-side configuration.**
No `n_jobs`, no `OMP_NUM_THREADS`, no `KMP_*` environment variable can prevent libomp's own internal
thread-pool init from running -- that code path is not conditional on the requested thread count at all.
Every code-level mitigation mlframe can apply from the outside has now been tried and rejected (rounds 6-11):
an older libomp version (round 6-8), the estimator's own thread count (round 9), the real OS env var (round
10), and confirming via libomp's own diagnostics that the env var genuinely took effect (round 11). There is
no further lever available from application code.

**Remaining options, none of them a code fix**:
1. File upstream against LLVM's `openmp` project (or Homebrew's `libomp` formula) with this exact repro:
   `EXC_BAD_ACCESS` in `__kmp_suspend_initialize_thread`, `arm64_tahoe`/macOS 26, reproduces at
   `OMP_NUM_THREADS=1` and both libomp 22.1.8 and 23.1.0 -- a precise, small, actionable report given
   everything captured in rounds 5b-11.
2. xfail/skip the three affected tests on macOS CI specifically, citing the upstream bug (a genuine
   third-party-platform-limitation skip per this repo's own fuzz/skip convention, not a band-aid over an
   mlframe bug) -- reduces macOS LightGBM-path coverage until libomp fixes this upstream or a future
   Homebrew/LLVM release resolves it. Needs the owner's explicit sign-off before applying, same as before.
3. Avoid LightGBM's OpenMP-parallelised Dataset-construction path on macOS specifically, if one exists that
   doesn't need it (e.g. a from-source libomp-free build, or forcing LightGBM's CPU backend into a mode
   that never opens a parallel region for binning) -- not investigated; would need its own research pass
   and may not exist as a supported LightGBM configuration.

Surfaced to the owner rather than picking one unilaterally -- this crosses from "mlframe code fix" into
"accept an upstream limitation and choose how to route around it," which is the owner's call.

### X5, round 12: owner chose option 2 (skip) -- per-test scoping massively undercounted the real exposure

Shipped (`4d82ba2b9`): `skipif` on the three originally-found tests, citing the upstream bug. Re-dispatched
CI: `one-by-one` on `macos-latest` still failed, on a FOURTH test in the same file
(`test_biz_val_training_suite_metadata_dict_schema`) that fits `mlframe_models=["lgb"]` and was simply
missed when scoping the original three. Fixed (`5c1464b56`) the same way.

Re-dispatched CI again: shard 1 (the real per-push matrix, not the diagnostic probe) failed AGAIN, this
time on `test_suite_api_ergonomics.py::test_default_extractor_regression_matches_explicit` -- a
COMPLETELY DIFFERENT FILE, also fitting `mlframe_models=["lgb"]`. A repo-wide grep at this point
(`grep -rln 'mlframe_models=\[.*"lgb"'  tests/`) found the pattern in **39 test files**, not three or four
-- enumerating them one CI round at a time does not converge, and each round costs a multi-hour macOS
runner queue.

**Owner's direction: stop scoping at the test level, intercept at the crash's own choke point instead.**
Every LightGBM fit path -- sklearn's `.fit()`, raw `lgb.train()`, and mlframe's own `lgb_shim.py` -- passes
through exactly one call before training starts: `Booster.__init__` calls `train_set.construct()`
(verified by reading `lightgbm.basic.Booster.__init__`'s own source, not assumed). `tests/conftest.py`'s
new `_install_macos_lgb_crash_skip()` (wired into the existing `pytest_configure` hook) monkeypatches
`lightgbm.basic.Dataset.construct` on darwin to raise `pytest.skip(...)` instead of letting the real call
through -- `pytest.skip()` raises `_pytest.outcomes.Skipped`, a `BaseException` (not `Exception`)
specifically so a broad `except Exception:` anywhere in the intervening mlframe call frames cannot swallow
it before it reaches pytest's runner. This covers every test that reaches real LightGBM training on macOS,
present or future, without needing another round of "which test crashes next."

Verified with a real fit, not just a synthetic call to `.construct()`: `test_macos_lgb_conftest_skip.py`
simulates `sys.platform == "darwin"`, installs the patch, and asserts a genuine
`LGBMClassifier(...).fit(X, y)` raises `Skipped` -- plus that the patch is a true no-op on linux/win32 (a
real fit completes normally) and honours the `MLFRAME_TESTS_ALLOW_MACOS_LGB_FIT=1` opt-out. The four
original per-test `skipif` markers are left in place (harmless, now redundant, but they document which
concrete instances first motivated the fix). Re-dispatching CI to confirm the conftest-level fix actually
reaches the remaining 35+ files without a fifth round of test-by-test whack-a-mole.

### X5, round 13: the conftest-level fix held -- macOS matrix temporarily isolated for faster iteration

The conftest interception (`51f6662f7`) reached every remaining file: no further LightGBM/libomp crash
was seen on macOS after this round. Per owner direction, `ci.yml`'s push matrix was temporarily narrowed
to `macos-latest` only (`f7f7b1968`) so each iteration round queues far fewer jobs, and five more
push-triggered workflows with no bearing on this investigation (Black, CodeQL, docs, `hooks-not-in-ci`,
`mypy-full`) had their `push` trigger dropped for the same reason (`c0536f500`) -- `pull_request` and
`workflow_dispatch` are untouched, so PR coverage is unaffected. Both are marked TEMPORARY in their own
comments, to be reverted once macOS is confirmed stable across a few consecutive pushes.

### X6: a second, unrelated macOS abort surfaced once X5 stopped masking it -- numba's workqueue threading
layer is not thread-safe for concurrent launch from multiple Python threads

CI (run `34612906516`, shards 4/6/9, 2026-09-11) crashed three shards with "Numba workqueue threading
layer is terminating: Concurrent access has been detected" then "Fatal Python error: Aborted" -- same
crash SIGNATURE as X5 (`Fatal Python error: Aborted`) but a DIFFERENT root cause, confirmed by reading the
actual crashing frame: `tests/feature_selection/mrmr/caching/test_fit_cache_thread_safety.py`'s
concurrent-fit storm reaches `MRMR.fit -> ... -> categorize_dataset -> per_feature_edges ->
edges_fayyad_irani -> mdlp_bin_edges -> _mdlp_recurse_validated_bfs`, a numba `parallel=True` kernel, from
two threads at once. numba's own error message names the cause: the default `'workqueue'` threading layer
is documented as not thread-safe for concurrent entry, and `fit()`'s own docstring already anticipates
concurrent multi-threaded callers (multi-target discovery, joblib-threading callers, web-service workers)
-- a genuine macOS production stability gap, not a test-only artifact, and not masked by X5's fix (X5 only
intercepts the LightGBM Dataset-construction path; this crash never reaches LightGBM at all).

**Disposition: RESOLVED (`7eb4f85dd`).** Switching numba's global threading layer to a thread-safe one
(`tbb`/`omp`) was considered and rejected: it would change scheduling for every already-benchmarked
`njit(parallel=True)` kernel in the codebase, and `tbb` is not an installed dependency. Instead, a
darwin-only process-wide `threading.Lock` (`_MACOS_NUMBA_PARALLEL_FIT_LOCK` in `_mrmr_class.py`) now
serializes `fit()`'s call into `_fit_body`: each fit still uses numba's own internal parallelism, only two
fits can no longer LAUNCH a parallel region at the same instant. Linux/Windows are unaffected -- the lock
variable is `None` off darwin, so `fit()` takes the exact pre-fix path there. Verified the mechanism
directly (not just "CI went green"): `test_macos_concurrent_fit_serialization.py` monkeypatches a real
`threading.Lock()` simulating the darwin branch and proves two concurrent `fit()` calls never run
`_fit_body` simultaneously, plus a sanity check confirming the same spy DOES catch overlap when the lock
is `None` (matching real Linux/Windows behaviour). Full existing thread-safety suite passes unchanged.

### X6, round 2: the `fit()`-level lock did not cover it -- the real concurrency is INSIDE one fit

The next CI round (`34628887279`) went from 3 failing macOS shards to 9/10: the crash reproduces from a
SINGLE `MRMR.fit()` call, not two concurrent ones. `_run_fe_step_impl`'s `parallel_run` (a joblib wrapper)
silently falls back from the loky (process-based, safe) backend to a threading one whenever loky cannot
spawn a nested process pool -- which happens whenever the caller is already on a non-main thread, exactly
what a pytest-xdist worker is -- and spawns several worker threads that all call the same numba
`parallel=True` dispatcher (`_dispatch_batch_mi_with_noise_gate`) at once. Round 1's lock only guards the
OUTER `fit()` entry point; it does nothing for concurrency joblib introduces inside a single fit.

**Attempted next: switch numba's threading layer to `tbb` on macOS (`7ebde971c`).** REJECTED, fast and
hard: the `tbb` PyPI package ships NO macOS wheel at all (only `manylinux_2_28_x86_64` and `win_amd64`),
so the install itself failed on every macOS shard (round 3, `34658837510`, 10/10 failures -- worse than
before, since now nothing even reached the tests). Reverted immediately.

**Real fix: this repo already has the right mechanism, built for exactly this crash class
(`mlframe._numba_parallel_guard.parallel_kernel_entry`, added after an earlier three-OS CI crash) --
it just had a gap.** `_nested_parallel_scan.py`'s static gate (`tests/test_meta/
test_no_unguarded_nested_parallel.py`) only recognised a literal `ThreadPoolExecutor(` call as a "thread
fan-out start"; it had no idea `Parallel`/`parallel_run` (joblib's own wrapper, and this repo's shim over
it) can silently become one too. Guarded `_dispatch_batch_mi_with_noise_gate` itself (the shared dispatcher
>10 FE-family modules call into) with `parallel_kernel_entry()` via a thin wrapper delegating to a renamed
`_dispatch_batch_mi_with_noise_gate_impl`, so all three of its internal `_cpu_kernel(...)` call sites are
covered without touching each one. Verified against the existing fe-pairs test suite plus the scanner's own
gate (10/10 green).

**Disposition on the scanner gap itself: FUTURE, not fixed here.** Extending `_POOL_STARTER_NAMES` to
include `parallel_run`/`Parallel` (the correct, durable fix -- otherwise the next joblib-threading crash is
invisible to this gate too, the same way this one was) surfaces **36 currently-unguarded reachable paths
across ~15 unrelated modules** (shap-proxy, ensembling, baselines, composite discovery, polynom-pair FE,
...) -- a real, pre-existing gap this session's narrow fix does not touch, not something safe to blanket
allowlist (the gate's own convention requires "the reason it cannot race", not "not yet audited") or fix
blind under CI-outage time pressure. Reverted the scanner widening for this round. **Next action:** a
dedicated pass auditing each of the 36 paths `python -m mlframe._nested_parallel_scan` reports once
`_POOL_STARTER_NAMES` includes `parallel_run`/`Parallel` -- guard each real one, allowlist genuinely
single-threaded ones (e.g. a blocking-future watchdog, per the existing allowlist's own precedent) with a
real reason.
