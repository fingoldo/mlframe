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
| L1.1 | Every CI lint tool runs unpinned via `uvx` | **PARTIAL / FUTURE** | The pins now exist on this side: all five previously-undeclared tools are pinned exactly in the dev extra (L1.4), and a new meta-test asserts the **installed** version matches the pin. The `uvx` invocations themselves live in `fingoldo/py-ci-shared`, a different repository. **Next action:** add version inputs to `lint-blocking.yml` there, defaulted to these pins. |
| L1.2 | The pinned ruff is not the ruff that runs | **RESOLVED** | Confirmed live: every config said 0.16.1, the installed ruff was 0.15.22. Added `test_installed_tool_versions_match_their_exact_pins`, verified failing before the fix, then installed the pin. **The real pin then found 28 findings the stale one could not see** — 26 ISC004, 1 RUF036, and a genuine duplicate entry in `mlframe.training.__all__` (RUF068). All 28 fixed; the tree is clean under the pinned ruff. |
| L1.3 | mypy is a blocking zero-error gate on `mypy>=1.0` | **RESOLVED** | Pinned `mypy==2.1.0`, matching `mypy-full.yml`, and covered by the new version test. |
| L1.4 | Five blocking hooks invoke tools declared nowhere | **RESOLVED** | `codespell`, `deptry`, `interrogate`, `vulture`, `yamllint` pinned exactly in the dev extra and added to deptry's DEP002 list. |
| L1.5 | `detect-secrets` / `shellcheck` run only in pre-commit | **RESOLVED (advisory)** | New `hooks-not-in-ci.yml` runs exactly the nine hooks with no CI counterpart, one per step so a failure names itself. Advisory for now because three of them are auto-fixers whose failure mode on a fresh checkout is "I rewrote your files" and nobody has measured how many that touches. **Next action:** flip `continue-on-error` off once a run reads clean. |
| L1.6 | No coverage gate anywhere | **FUTURE** | Agreed in principle, and the agent is right that `patch` status is the high-signal half. Deliberately not set from a guessed number: the threshold has to come from a measured full-suite total, and the combined figure is currently assembled by nightlies that are not all green. **Next action:** take the number from the first green `codecov-full` run, set `patch` with `informational: true`, then flip. |
| L1.7 | `vulture --min-confidence 80` reports nothing | **DOC** | Reproduced (0 at 80, 1071 at 60). Not lowered: 1071 findings need a baseline first, and the uncalled-function half is already gated separately. The residue the agent identifies — unused attributes — is a genuine ML-config bug class and is the right shape for a future scoped scan. Recorded rather than fixed. |
| L1.8 | The two vulture runs contradict the comment justifying them | **DOC** | Accurate. Left as-is deliberately: correcting the comment without changing the invocation is honest, but the invocation change (one pass over `src` + `tests`) belongs with L1.7's baseline work, and splitting them would mean editing the same comment twice. |
| L1.9 | `doctest_optionflags` set, no doctest ever runs | **DOC** | True. Kept rather than dropped because 3.4 (a scoped `--doctest-modules` leg) would need exactly this setting; deleting it now to re-add it later is churn. |
| L1.10 | The mccabe comment says 70, ruff reports 92 | **RESOLVED** | Comment corrected to the measured 92, and `test_c901_debt_ratchet.py` added: the count may fall freely, a rise fails, a ceiling drifting more than 10 above the real count fails, and the pyproject comment must agree with the ratchet. |
| L1.11 | `LOC_BUDGET_EXEMPT` has no staleness check and its docstring is wrong | **RESOLVED** | Docstring corrected, `pytest.skip` on a missing src tree replaced with a hard failure, and two assertions added. **They immediately found two drained exemptions** — `_gpu_resident_basis.py` at 868 LOC and `transforms/nonlinear.py` at 624 — both removed from the set, so those files are now gated like every other. |
| L1.12 | Baselines drain-warn but never require draining | **PARTIALLY RESOLVED** | The agent's own caveat blocks the general form: the large baselines are `file:line`-keyed, so failing on drained entries turns every unrelated line drift red. The nine empty ones have no such tension, and the risk there is the opposite one the finding names -- silencing a NEW violation by appending to a zero-tolerance file. `test_zero_tolerance_baselines_stay_empty.py` now asserts all nine stay at zero and that any baseline draining to zero joins the list. **Next action:** measure a week of churn on one large baseline before deciding the strict form for the rest. |
| L1.13 | `regen_baselines.py` covers 7 of 27 baselines; three unchecked lists | **RESOLVED** | Added `test_baseline_registry_consistency.py`. Resolving flags by asking the live pytest config, not by grepping conftest — several are registered indirectly through `py_ci_shared`, which a text scan reports as missing. **It found four refresh flags that every one of their own test modules documents and that pytest rejects outright**: `pytest --refresh-tick-isinstance-baseline` errored with "unrecognized arguments". All four registered and verified working. Twenty baselines were undocumented in BASELINES_README.md; all now documented. Six non-conventional routes recorded explicitly, two of which (`_code_audit_tests`, `_uncalled_functions`) genuinely have no automated refresh — see below. |
| L1.13b | `_code_audit_tests_baseline.json` and `_uncalled_functions_baseline.json` have no refresh route at all | **FUTURE** (found while closing L1.13) | Both docstrings say "refresh"; neither module implements one, so a drained entry must be deleted by hand. Recorded in `FLAG_EXEMPT` with that exact wording rather than hidden. **Next action:** add a `regenerate_baseline()` to each and register the flag. |
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
| L3.3 | `pip-audit` blocking rather than advisory | **PARTIALLY RESOLVED** | The audit now exists over the right subject and runs on a schedule; only the blocking half is deferred, and for the reason the agent predicted. Measured: 18 advisories, of which **two are direct dependencies** (`aiohttp` needs 3.14.3, `tornado` needs 6.5.8) and **two are structurally capped by other packages** (mlflow holds `cryptography<49`, torch holds `setuptools<82`) -- the latter pair is the `--ignore-vuln` list the finding asks for, and it is a fact about upstream rather than about this repo. The aiohttp bump is deliberately NOT made here: dependabot PR #27 has carried exactly it since 2026-08-07 and is mergeable, so a duplicate commit would conflict with an open PR for the same result. **Method, because getting it wrong is easy and silent:** audit the *lock export* with `--extra all --extra dev` (a bare export emits 42 packages and zero advisories, which looks like good news), pass `--no-deps` (an already-pinned file must not be re-resolved), and run it on ubuntu with the representative Python (uv 0.11 has no `--python-platform` on `export`, so pip evaluates the universal file's markers for whatever interpreter it runs on -- the runner IS the targeting). Auditing the installed environment instead measures a developer's machine, not what the project declares. **Next action:** merge the fix backlog, then flip `continue-on-error` off with the two structural pins in `--ignore-vuln`. |
| L3.4 | `pytest --doctest-modules` on a scoped subpackage | **FUTURE** | Right scope (calibration, metrics — the mypy beachhead). Deferred behind L2.1: both add a new CI leg over the same subpackages, and they should land as one leg, not two. |
| L3.5 | Spend the existing `hypothesis` dep on numeric invariants | **DOC** | Not a tool change and not a gate: it is a standing suggestion for how to write the next tests over the MI/discretization/rank-correlation surface. Recorded so it is not re-proposed as new. |
| L3.6 | Explicitly NOT recommended (pylint, flake8, pyright, xenon/radon, safety, refurb) | **DOC** | Agreed, with the measurements the agent gives. |

### Ceremony to remove

| # | Finding | Disposition | Notes |
|---|---|---|---|
| L4.1 | `.semgrep.yml`: 2 advisory rules, one duplicating a blocking gate | **FUTURE** | Agreed on the net-negative judgement. Removing it means porting `module-global-write-via-reexport-alias` to a meta-test first — deleting the rule and the dep in the same change without the port would lose the one rule that is not duplicated. **Next action:** write the meta-test, then drop semgrep. |
| L4.2 | `[tool.importlinter]`: 2 contracts, advisory, and its own config says so | **FUTURE** | Same shape and same order: convert contract #1 to a meta-test, then delete the section and the dep. |
| L4.3 | The nine empty baselines' machinery | **REJECTED** | The agent states the counter-argument itself and it wins here: the uniform shape is what lets the next regression be baselined in one command, and L1.13's new consistency gate makes the uniformity load-bearing rather than decorative. |
| L4.4 | `[tool.pydoclint]`: advisory with ~200 known-residual findings | **FUTURE** | The agent's two options are both defensible; baselining the ~200 and making it blocking is the one that keeps the value. **Next action:** generate the baseline, flip to blocking. |
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
| X4 | `calibration.html`, a rendered plotly document, was tracked at the repository root. It is a test artifact: some reporting test renders to a bare filename rather than `tmp_path`, so it lands in the working directory, looks like an ordinary untracked change, and gets swept in by `git add -A`. **It reached master in commit `bd8e212f4` of this very round -- my own commit, about linter configuration.** Noticed only because a later run under a different plotly version rewrote it and the diff surfaced again. | **PARTIALLY RESOLVED.** Removed from the index, `.gitignore` now excludes root-level chart artifacts, and `test_no_chart_artifacts_in_repo_root.py` gates all three halves: none tracked, the ignore patterns still present, and a standing skip naming the file if a stray render is sitting there. **Next action:** find the test that renders without `tmp_path` -- roughly sixty reporting tests do not use the fixture, and grep for the filename finds no producer, so it needs a run with a watched working directory rather than a search. |
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
