# GitHub Actions review - 2026-09-08

Scope: all 15 files in `.github/workflows/`, `.github/dependabot.yml`, `.github/actionlint.yaml`.
Method: static read of the files in this worktree only. No run logs, no API, no test execution.

Legend: **READ** = literally in the file. **INFERRED** = follows from documented Actions
behaviour but not observed in a run of this repo. Findings whose rationale is already
documented in the files are listed in the "deliberate, not defects" section and NOT counted
as defects.

---

## 1. REAL DEFECTS

### D1. `ci.yml` shard artifact names collide across the three OS legs (high)

**READ** `ci.yml:335-343`:

```yaml
      - name: Upload shard test durations
        if: always()
        uses: actions/upload-artifact@...
        with:
          name: test-durations-${{ matrix.python-version }}-${{ matrix.group }}
```

The name contains **python-version and group but not `matrix.os`**. Since 2026-09-08 a push
runs `("ubuntu-latest", "windows-latest", "macos-latest") x ONE python x groups 1..10`
(`ci.yml:120-125`), so exactly three concurrent jobs compute the same artifact name, e.g.
`test-durations-3.12-7`. The weekly sweep has the same collision (three OSes per
(version, group) pair, `ci.yml:105-114`).

**INFERRED** (documented `actions/upload-artifact` v4+ behaviour, not observed here): the
second and third uploader get `Conflict: an artifact with this name already exists`, a
non-retryable error. The step has no `continue-on-error`, so two of every three shard jobs
fail on a step that has nothing to do with the tests - and these shard jobs are exactly what
`ci-required` gates on (`ci.yml:657`). Note this only became reachable when the push matrix
gained the Windows/macOS legs; `raw-coverage-ci-${{ matrix.group }}` (`ci.yml:367`) does NOT
collide because it is gated to `COVERAGE_OS` + `REPRESENTATIVE_PYTHON`.

Fix: `name: test-durations-${{ runner.os }}-${{ matrix.python-version }}-${{ matrix.group }}`.
`merge-test-durations`'s glob (`pattern: test-durations-*`, `ci.yml:480`) still matches. Do
NOT use `overwrite: true` - that keeps only one OS's durations and hides the problem.

Secondary consequence once fixed: the merge script (`ci.yml:494-499`) is last-write-wins per
node id, so a macOS shard's timings can overwrite an Ubuntu shard's in the file that balances
all three platforms. Its comment only reasons about *python versions* overwriting each other
(`ci.yml:491-493`), written before the OS axis existed. Cheapest correct fix: only upload
durations from `runner.os == 'Linux'`, or key the merge to prefer the Linux value.

### D2. `ci-required` passes vacuously when `matrix-setup` fails or a gate is skipped (high)

**READ** `ci.yml:652-667`:

```yaml
    needs: [test, test-heavy-serial, ruff-blocking, lint-blocking, tests-ruff-blocking,
            tests-black-filtered, tests-lint-blocking, build]
          ANY_FAILED: ${{ contains(needs.*.result, 'failure') || contains(needs.*.result, 'cancelled') }}
```

Two holes:

1. `matrix-setup` is **not** in `needs`. If it fails (or its inline Python emits malformed
   JSON), `test` and `test-heavy-serial` are *skipped*, `build` is skipped, and every
   `needs.*.result` is `skipped` - neither `failure` nor `cancelled`. The one aggregation
   check that branch protection is meant to use reports green with zero tests run.
2. `'skipped'` is not checked at all, so any other path to a skipped gate (a reusable
   workflow whose own `if` short-circuits, a job removed by a matrix edit) reads as pass.

Fix: add `matrix-setup` to `needs`, and treat `skipped` as failure:
`contains(needs.*.result, 'failure') || contains(needs.*.result, 'cancelled') || contains(needs.*.result, 'skipped')`.

### D3. `numba-coverage.yml`'s `test-heavy-serial-numba-disabled` is missing the workflow_run gate (medium-high)

**READ**, `numba-coverage.yml:86-88` (the sharded job) gates on:

```yaml
      (github.event_name != 'workflow_run' || (github.event.workflow_run.conclusion == 'success' && github.event.workflow_run.event == 'push'))
```

**READ**, `numba-coverage.yml:246` (the serial job) gates only on:

```yaml
    if: github.event_name != 'workflow_dispatch' || inputs.run
```

Two consequences, neither addressed in that job's long comment block (`:141-154`), which
discusses only OOM/sharding - so this reads as an oversight, not a decision:

* **Waste**: 4 serial jobs (180-minute cap each) fire on *every* completion of CI on master,
  including failed and cancelled ones - the exact spend pattern the file's own header says
  once exhausted the account's Actions minutes (`:3-12`).
* **Security**: it checks out `${{ github.event.workflow_run.head_sha }}` (`:262`) and then
  `pip install -e .` + runs the suite. The sharded job's own comment (`:80-85`) explains
  precisely why an ungated `workflow_run` leg is the "pwn request" shape for a public repo
  whose CI also runs on `pull_request`. Mitigating factor (**INFERRED**): the `branches:
  [master]` filter on the trigger (`:51`) matches the *head branch of the triggering run*, so
  a fork PR only reaches this if its head branch is named `master` - common enough with
  fork-from-master workflows to not rely on.

Fix: copy the identical `if:` from `:86-88` onto this job.

### D4. `dependabot-auto-merge.yml` almost certainly cannot merge anything (medium-high)

**READ** `dependabot-auto-merge.yml:89-92`: the file explicitly chose `pull_request` over
`pull_request_target`, reasoning that a same-repo PR "already gets a GITHUB_TOKEN with the
write permissions requested below".

**INFERRED** (GitHub's documented Dependabot behaviour, unchanged since 2021): for events
*triggered by Dependabot*, `GITHUB_TOKEN` is downgraded to **read-only** and repo secrets are
withheld, regardless of the `permissions:` block - the same-repo-vs-fork distinction the
comment reasons about is not what governs it. `gh pr merge --auto` (`:112`) would then fail
with a permissions error. This is consistent with the observed symptom the comment itself
records ("3 unmerged/closed-without-merge github-actions group PRs since July"), which it
attributes to the old major-version gate instead.

Fix (GitHub's own documented recipe): switch to `pull_request_target` - safe here because the
job never checks out PR code - or drive it with a PAT / GitHub App token. Worth confirming
against one real run's log before changing, since the whole finding rests on the token
downgrade.

### D5. `release.yml` will publish to PyPI from a manual dispatch on any ref (medium)

**READ** `release.yml:232-235, 266-272, 286-304`: `workflow_dispatch` is a trigger; the
tag/version equality assert is `if: github.event_name == 'release'`; the `publish` job has no
event condition at all. A dispatch from any branch therefore builds whatever `version.py`
says and uploads it via Trusted Publishing. PyPI uploads are irreversible.

Mitigations that already exist: the `pypi` environment (`:291`) can carry a reviewer/branch
rule, and dispatch requires write access. Still, the guard belongs in the file.

Fix: `publish` gets `if: github.event_name == 'release'`, or the assert step drops its `if:`
and compares against the ref for dispatch runs too.

### D6. Reusable workflows are referenced by the mutable tag `@v1` while every action is SHA-pinned (medium)

**READ**: every `uses:` for an *action* in this repo is a 40-char SHA with a version comment
(e.g. `ci.yml:197, 202, 225, 238, 254`). Every *reusable workflow* is not:

* `ci.yml:529` `ruff-blocking.yml@v1`
* `ci.yml:535` `lint-blocking.yml@v1`
* `ci.yml:569, 576, 582` (tests-scoped trio)
* `black-filtered.yml:23`, `docs.yml:171`, `mypy-full.yml:211`

`@v1` is a movable tag. The blast radius is not theoretical: `docs.yml` calls its `@v1` under
`pages: write` + `id-token: write` (`docs.yml:164-167`). Pin these to SHAs like everything
else - Dependabot's `github-actions` ecosystem (`dependabot.yml:9-24`) updates reusable-workflow
SHAs too, so this costs nothing ongoing.

### D7. `merge-test-durations` pushes to master with no rebase and no cross-workflow lock (low-medium)

**READ** `ci.yml:503-515`, `numba-coverage.yml:342-354`, `update-test-durations.yml:390-405`:
three jobs `git commit` + `git push origin HEAD:"$REF_NAME"` from a checkout of a possibly
older master, none of them with `git pull --rebase`, a retry, or a shared `concurrency:` group
across the three. Any concurrent lands (this repo pushes often - see `black-filtered.yml:9-11`)
produce a non-fast-forward rejection and a red job. `ci.yml`'s copy is at least outside
`ci-required`'s `needs`, so it does not gate.

Fix: `git pull --rebase --autostash origin "$REF_NAME"` before push plus one retry, and give
all three jobs the same `concurrency: group: test-durations-commit, cancel-in-progress: false`.

### D8. `merge-test-durations` commits durations harvested from failed/cancelled shards (low)

**READ** `ci.yml:469` `if: always() && ...` with `needs: test`. Combined with the per-shard
`if: always()` upload (`:336`), a run where most shards were cancelled at the 360-minute cap
still commits a merged file built from partial data. Since the merge is `dict.update`, stale
good values survive, so this degrades balance rather than destroying it - but it does mean a
"chore: refresh .test_durations" commit can be a strictly worse file than its parent. Consider
requiring `needs.test.result == 'success'` for the *commit* step (still uploading always).

### D9. `main` is a trigger branch that several jobs silently ignore (low)

**READ**: `ci.yml:5,7` trigger on `[master, main]`, but `merge-test-durations` is gated on
`github.ref == 'refs/heads/master'` (`:469`), `codecov-full.yml:208` and
`numba-coverage.yml:51` filter `branches: [master]`, `docs.yml:147` only `master`. Either the
repo has no `main` (then drop it from the triggers, it is dead config that reads as coverage)
or it does (then those jobs never fire there). One-line cleanup; listed because half a dozen
files disagree about it.

---

## 2. GAPS (things a mature Python-library CI has that this one does not)

### G1. No lockfile / no pinned resolution anywhere - the largest reliability gap

**READ**: every install is an unpinned resolve (`uv pip install "./pyutilz[...]" -e ".[all,dev]"`).
Only `pyutilz` is pinned (a git SHA, consistently, in all 7 call sites). For a ~150-dep,
35k-test suite this means any upstream release can turn the matrix red with no local change,
and no run is reproducible after the fact. `sklearn-matrix-ci.yml` pins exactly one library.
Cost: `uv lock` committed + `uv sync --frozen` in CI + a weekly "refresh the lock" job
(Dependabot already handles the PR side). This is the single change with the best
red-noise-per-hour return.

### G2. No lowest-supported-version resolution job

`pyproject.toml` presumably carries floors (`pyutilz>=1.0.0`, `numpy<2.5` are referenced in
comments). Nothing tests them; a floor that no longer installs or imports is invisible until a
user hits it. Cost: one ubuntu job, `uv pip install --resolution=lowest-direct ".[all,dev]"` +
a smoke import + a fast test subset. ~10 minutes weekly.

### G3. No aggregated test-failure report

30+ shard jobs, no `--junitxml`, no upload of results, no `$GITHUB_STEP_SUMMARY` write, no
annotations. Diagnosing a red push means opening N job logs of a 35k-test suite by hand.
Cost: add `--junitxml=junit-${{ matrix.os }}-${{ matrix.group }}.xml`, upload it, and one
aggregation job (`pytest-html`/`mikepenz/action-junit-report`, or 20 lines of Python writing
to `GITHUB_STEP_SUMMARY`). Near-zero minutes.

### G4. Build/install smoke is Linux-only

**READ** `ci.yml:597-642`: `build` runs on `ubuntu-latest` only, and the wheel smoke uses
`/tmp/smoke/bin/...` (POSIX paths). The suite *tests* on Windows and macOS but never proves
`pip install mlframe` works there. For a pure-Python wheel the risk is low (path handling in
package data, `py.typed`), but it is a stated goal of that job. Cost: make `build` a 3-OS
matrix or add a single `windows-latest` install-smoke job; the build itself is minutes.

### G5. No release provenance / SBOM

`release.yml` already uses Trusted Publishing (good). Missing: `actions/attest-build-provenance`
(one step, `attestations: write`, free) and any SBOM (`cyclonedx-py`). Both are becoming
table stakes for a published library.

### G6. Nothing verifies the workflows themselves

`.github/actionlint.yaml` exists and configures the `gpu` self-hosted label, but **no workflow
runs actionlint**. `zizmor` runs (via `lint-blocking` with `run-zizmor: true`, `ci.yml:550`),
which covers the security half but not the correctness half - actionlint is what catches D1/D2-class
typos, bad `if:` expressions and unknown contexts. Cost: one 30-second job, or a pre-commit hook.

### G7. `deep-nightly` produces durations no one consumes and cannot self-balance

**READ** `deep-nightly.yml:117-121` documents this as deliberate ("no merge-and-commit job
yet ... worth doing once this run is green"). Listed as a known gap, not a defect: until it
exists, the 20 shards split by file count and the header itself predicts a heavy shard may
need most of the 340 minutes. The artifacts (`deep-durations-*`, `:166`) are the only
never-consumed artifact set in the repo.

---

## 3. WASTE

### W1. `deep-nightly.yml` has no uv cache - 20 cold ~5 GB resolves every night

**READ** `deep-nightly.yml:83-84`:

```yaml
      - name: Set up uv
        uses: astral-sh/setup-uv@...        # no `with:` at all
```

Every other uv call site sets `enable-cache: true` + `cache-dependency-glob: pyproject.toml`
(`ci.yml:225-228, 419-423`, `update-test-durations.yml:357-360`). `ci.yml:206-211` documents
that this install is a "~5 GB, several-hundred-package graph". 20 shards x nightly, uncached.
One-line fix.

### W2. `deep-nightly.yml:78-79` and `numba-coverage.yml:124-125` set `cache: pip` on setup-python but install with uv

In `deep-nightly` the install goes through `install-pyutilz` (uv), so the pip cache is
populated/restored for nothing. (In `numba-coverage.yml` the installs *are* plain `pip`, so
there it is correct.) Harmless but misleading; drop it from `deep-nightly`.

### W3. `ci.yml` has no `paths-ignore` - a README edit runs 30 shards + 4 heavy-serial jobs

Given the documented concurrency ceiling ("~20 concurrent jobs shared across every repo on the
account", `ci.yml:58-60`), a docs-only push consuming the entire budget is the most expensive
avoidable spend here. Safe to add because the repo deliberately has no branch-protection
required checks (`dependabot-auto-merge.yml:85-87`), so a skipped `ci-required` blocks nothing.
Suggest `paths-ignore: ['**.md', 'docs/**', 'audits/**', 'LICENSE']`.

### W4. `update-test-durations.yml` is superseded, cannot finish, and races the job that replaced it

**READ** its own header (`:316-324`): "SUPERSEDED 2026-08-15 by ci.yml's merge-test-durations
... THIS workflow's own unsharded 2-worker run structurally cannot finish inside any
job-timeout ceiling ... so it never once reached the commit step." It still schedules a
300-minute weekly job (`:332, 345`) that the file itself predicts will be truncated, and its
push races D7. `.test_durations` in this worktree has **30131 entries** (measured), i.e. the
ci.yml merge path is working. Delete this workflow, or shard it the same way and keep it
quarterly.

### W5. Four near-identical install stanzas

`git clone pyutilz` + `checkout <sha>` + `pip install` is inlined in `numba-coverage.yml`
(twice), `gpu-matrix.yml:50-53`, `gpu-extras-install-matrix.yml:162-165`,
`sklearn-matrix-ci.yml:304-307`, plus `mypy-full.yml:216-219` and `ci.yml:546-549` as
install-commands - while the shared `install-pyutilz` composite action exists and is used by
`ci.yml`, `deep-nightly.yml`, `update-test-durations.yml`. The pinned SHA
`ec016b15e76e...` is currently identical in all of them (verified by grep), but keeping seven
copies in sync by hand is exactly the drift the action was extracted to prevent. Converting
`numba-coverage.yml`'s two copies is free (it is on hosted ubuntu like the rest).

### W6. `codecov-full.yml` has no `concurrency:` group

It triggers on both a daily cron (`:205`) and every `numba-coverage-nightly` completion
(`:206-209`). Two overlapping runs both `coverage combine` + upload to the same `combined`
flag. Cheap: add a group with `cancel-in-progress: true`.

---

## 4. NEW TOOLS WORTH ADDING

| Tool | Justification (one line) | Cost |
|---|---|---|
| `uv lock` + `uv sync --frozen` | Removes the whole class of "red today, unchanged code" failures and makes any run reproducible (G1). | One file, one weekly refresh job. |
| `actionlint` job or pre-commit hook | The config file already exists but nothing runs it; catches D1/D2-shaped mistakes before a 30-shard run does (G6). | ~30 s per push. |
| `--junitxml` + a JUnit summary step | Turns "N of 30 shards red" into one readable table instead of N log dives (G3). | ~0 minutes, one aggregation job. |
| `actions/attest-build-provenance` in `release.yml` | Signed provenance for a published PyPI package, alongside the Trusted Publishing already in place (G5). | One step, `attestations: write`. |
| `--resolution=lowest-direct` job | Proves the declared dependency floors actually install and import (G2). | ~10 min weekly. |
| `step-security/harden-runner` (audit mode) | Egress visibility on jobs that `pip install` several hundred packages and hold `CODECOV_TOKEN`; audit mode never blocks. | One step per job, no failures in audit mode. |
| `pytest-xdist --max-worker-restart` / `pytest-timeout method=signal` on Linux | `ci.yml:154-160` names the unkillable-native-call hang as the likely root cause of shards dying at the cap and explicitly says `signal` on Linux could preempt it - it is still not set. | Config-only. |

---

## Deliberate, documented - NOT reported as defects

Read and verified against the files' own comments, listed so they are not re-flagged:

* `cancel-in-progress: true` on `ci.yml` including pushes to master, despite the codecov
  incident - reverted with a full write-up (`ci.yml:22-30`).
* Windows/macOS legs blocking on a push even though they have never been green (`:116-119`).
* `continue-on-error` on 3.14 / non-Linux rows in the weekly sweep (`:107-110`).
* Coverage upload restricted to one OS + one version, driven from workflow `env` so the two
  cannot drift (`:35-50`).
* `merge-multiple: false` on every `download-artifact` (each shard's file is literally
  `.coverage`) - `codecov-full.yml:279-282`.
* `-n 4` rather than `-n auto` in `numba-coverage.yml` (nested joblib oversubscription).
* `--durations-path` kept separate per suite; the seeded `.test_durations_numba_disabled`.
* No `permissions:` restated at reusable-workflow call sites (callee is capped by caller) -
  `ci.yml:521-526`; and the converse in `docs.yml:157-163`.
* `docs.yml`'s `pages: write` + `id-token: write` at workflow level.
* Per-event `concurrency` keys on `numba-coverage.yml` / `deep-nightly.yml`.
* `workflow_run` gating on `conclusion == 'success' && event == 'push'` (the one job that
  omits it is D3).
* Unthrottled `workflow_run` chaining despite the minutes risk - accepted explicitly
  (`numba-coverage.yml:42-48`).
* `env`-passed `SHARD_SPLITS` / `SHARD_GROUP` / `REF_NAME` instead of `${{ }}` in `run:`
  (zizmor template-injection); the remaining `${{ matrix.* }}` splices
  (`numba-coverage.yml:171`, `sklearn-matrix-ci.yml:328`, `gpu-extras-install-matrix.yml:172`)
  are workflow-authored matrix values, not attacker-controlled.
* `include-hidden-files: true` on every dotfile upload, with the root-cause write-up.
* `persist-credentials: false` on every checkout except the three that must push.
* All actions SHA-pinned (the reusable-workflow exception is D6).
* Timeouts on every job; `gpu-matrix.yml` restricted to `workflow_dispatch`.
* `deep-nightly` running one OS / one Python (`deep-nightly.yml:17-20`).
* Missing deep-run artifacts degrading `codecov-full` to a two-way merge rather than failing
  (`codecov-full.yml:325-332`).

---

## What I could NOT check

* **The `fingoldo/py-ci-shared` reusable workflows and composite actions** (`ruff-blocking`,
  `lint-blocking`, `black-filtered`, `docs`, `mypy-full`, `install-pyutilz`, `upload-codecov`)
  are not in this checkout. Their internal `permissions:`, pinning, `fail_ci_if_error`
  setting, and whether any of them can pass vacuously are unverified. Given how much of the
  gating lives there, that repo deserves its own pass.
* **Any run history**: no job logs, timings, queue depths, artifact listings or codecov
  results. Every claim about what *happens* (D1's 409, D3's spend, D4's token downgrade) is
  marked INFERRED and is the cheapest thing to confirm from one recent run.
* **Repo settings**: branch protection (the files claim there is none), environment reviewers
  on `pypi`, Actions permissions/allowlist, secret names actually present
  (`CODECOV_TOKEN` is referenced in three files).
* **`ci.yml:407`** `python-version: ["${{ needs.matrix-setup.outputs.representative_python }}"]`
  where the output is `${{ env.REPRESENTATIVE_PYTHON }}` (`:81`) - workflow-level `env` in a
  job-level `outputs` value. Documentation says `env` is available there; there are known
  reports of workflow-level `env` resolving empty in job-level keys. I could not settle it
  statically. If it resolves empty, `test-heavy-serial` fails at setup-python and its job name
  renders with a blank version - visible in one run's job list. Cheap hardening either way:
  have the `matrix-setup` Python script `print("representative_python=" + REPRESENTATIVE)` to
  `$GITHUB_OUTPUT` instead of routing it through the job-level `outputs` mapping.
* **`pyproject.toml`** (pytest options, coverage `[paths]` used by `codecov-full`'s
  `coverage combine`, dependency floors) was out of scope and not read; a missing `[paths]`
  remap is a plausible silent failure mode for the combined report if any contributor ever
  runs on a different workspace path.
