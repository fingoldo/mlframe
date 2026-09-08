# mlframe public presentation review — README badges, structure, PyPI metadata

Date: 2026-09-08. Scope: `README.md`, `pyproject.toml [project]`, `docs/README.md`, `CHANGELOG.md`,
`.github/workflows/*`, `codecov.yml`, `mkdocs.yml`.

## What I could NOT check

I have no network access in this session. Everything below about **rendering** is inference from the
URL shape plus what the repo declares; I could not fetch a single badge, the codecov API, PyPI, or
`https://fingoldo.github.io/mlframe/`. Specifically unverified:

- whether each codecov flag actually has data on `master` right now (only that the workflows that
  publish those flags exist and pass `flags:`);
- whether the GitHub Pages site is live (see F3);
- whether `mlframe` exists on PyPI at all. The task brief says "published to PyPI"; the repo says the
  opposite in two places (`README.md:29-30` "Neither package is published to PyPI yet", and
  `release.yml`'s header comment: "a successful upload requires the runtime dependency `pyutilz` to
  be available on PyPI; until then this workflow builds and validates but the upload step will be the
  gating action"). I have treated **not on PyPI** as the working assumption and flagged every place
  that depends on it. If it *is* published, F2 and P4-b change (see the notes there).

---

## 1. Broken or misleading badges

### F1 — VERIFIED CLEAN: the three codecov flag badges are well-formed

All three carry the branch in the path, which is the exact thing that otherwise renders `unknown`:

```
https://img.shields.io/codecov/c/github/fingoldo/mlframe/master?flag=numba-disabled
https://img.shields.io/codecov/c/github/fingoldo/mlframe/master?flag=combined
https://img.shields.io/codecov/c/github/fingoldo/mlframe/master?flag=deep
```

`/master` is present in each. And each flag is really published (verified in the workflow sources):

| Badge | Flag | Published by | Trigger |
|---|---|---|---|
| codecov-numba | `numba-disabled` | `.github/workflows/numba-coverage.yml:215` | nightly cron `0 3 * * *` + `workflow_run` after CI on master |
| codecov-full | `combined` | `.github/workflows/codecov-full.yml:233` | nightly cron `0 10 * * *` + `workflow_run` after numba-coverage-nightly, `branches: [master]` |
| codecov-deep | `deep` | `.github/workflows/deep-nightly.yml:149` | nightly cron `40 1 * * *` |

The overall badge (`codecov.io/gh/fingoldo/mlframe/branch/master/graph/badge.svg`) also names the
branch, and the default CI upload (`ci.yml:345-354`, via `py-ci-shared/upload-codecov`) passes **no**
`flags:`, so it lands unflagged and feeds that badge. No badge points at a nonexistent flag.

Residual risk I cannot rule out: all three flag badges are fed by **scheduled-only** workflows. If a
nightly is red or skipped long enough that master has no report for that flag, shields renders
`unknown` — not because the URL is wrong, but because there is no data. That failure mode is invisible
until it happens, and this repo has an on-file history of exactly that (`codecov.yml` and `ci.yml:22-30`
both document a period where the badge sat at 0.00%).

### F2 — HIGH: four coverage badges, none of which a visitor can interpret

This is the single worst thing in the badge block. A first-time reader sees `codecov`, `codecov-numba`,
`codecov-full`, `codecov-deep` — four different percentages, with no indication which one is "the"
coverage number. `combined` and `deep` are internal flag names; "full" and "deep" are not
self-describing, and `codecov-full` sounds like it supersedes `codecov`, which is roughly true but
undocumented at the badge. The repo itself explains why they cannot be added together
(`ci.yml:356-362`), and none of that reaches the README.

Fix: keep **one** headline coverage badge in the header; move the breakdown into a short prose block
lower down where the labels can be explained. Concretely, replace lines 7-10 with:

```markdown
[![coverage](https://img.shields.io/codecov/c/github/fingoldo/mlframe/master?label=coverage)](https://codecov.io/gh/fingoldo/mlframe)
```

and add, in the `## Testing` section:

```markdown
### Coverage

Coverage is measured on four different runs, which measure different things and must not be added
together (they overlap on most of the codebase):

[![coverage](https://img.shields.io/codecov/c/github/fingoldo/mlframe/master?label=per-PR%20coverage)](https://codecov.io/gh/fingoldo/mlframe)
[![no-JIT coverage](https://img.shields.io/codecov/c/github/fingoldo/mlframe/master?flag=numba-disabled&label=coverage%3A%20numba%20off)](https://codecov.io/gh/fingoldo/mlframe/flags)
[![combined coverage](https://img.shields.io/codecov/c/github/fingoldo/mlframe/master?flag=combined&label=coverage%3A%20combined)](https://codecov.io/gh/fingoldo/mlframe/flags)
[![deep coverage](https://img.shields.io/codecov/c/github/fingoldo/mlframe/master?flag=deep&label=coverage%3A%20deep%20nightly)](https://codecov.io/gh/fingoldo/mlframe/flags)

- **per-PR** — the sharded CI run on every push (`ci.yml`).
- **numba off** — the nightly run under `NUMBA_DISABLE_JIT=1`; `@njit` bodies are invisible to
  coverage.py, so this is the only run that can see inside them (`numba-coverage.yml`).
- **combined** — the two above unioned via `coverage combine`, not summed (`codecov-full.yml`).
- **deep nightly** — the slow/integration tests excluded from per-PR CI (`deep-nightly.yml`).
```

I have not verified those four one-line descriptions against the full workflow bodies beyond their
names, triggers and flag names; check the "deep nightly" line in particular against
`deep-nightly.yml`'s actual test selection before pasting.

### F3 — HIGH (needs one URL check): the `docs` badge may link to a 404

`README.md:12` links to `https://fingoldo.github.io/mlframe/`. `mkdocs.yml` declares that same
`site_url`, and `pyproject.toml:488` declares it as the `Documentation` URL. But `docs.yml:22-23`
says, in the repo's own comment: *"deploy is skipped by its own PUBLISH_DOCS gate by default"*. If
that gate has never been flipped for this repo, all three of those point at a page that does not
exist — and the badge is a *static* `img.shields.io/badge/docs-mkdocs-blue.svg`, so it renders a
cheerful blue "docs | mkdocs" regardless. A static badge can never be broken; only its link can, which
is precisely why this one is worth checking by hand.

Action: open the URL. If it is live, no change. If it is not, either flip `PUBLISH_DOCS` (the site
builds on every PR already, so the content is known-good) or repoint the badge and
`pyproject.toml`'s `Documentation` URL at the in-repo docs until it is:

```markdown
[![docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://github.com/fingoldo/mlframe/tree/master/docs)
```

### F4 — MEDIUM: three workflow badges use the legacy form and are branch-agnostic

```
README.md:3  .../workflows/CI/badge.svg
README.md:5  .../workflows/Black/badge.svg
README.md:6  .../workflows/sklearn-matrix/badge.svg
```

VERIFIED: all three workflow names exist and match exactly (`ci.yml` → `name: CI`,
`black-filtered.yml` → `name: Black`, `sklearn-matrix-ci.yml` → `name: sklearn-matrix`), so none of
these is dead. But the legacy `/workflows/<name>/badge.svg` form takes no query parameters, so it
reports the **latest run on any ref and any event** — a failing dependabot PR, or a fork PR, turns the
CI badge red on the README of a healthy master. `README.md:4` (MyPy) already uses the modern path
form but also omits `?branch=`, so it has the same problem in different syntax. This repo has 20+
dependabot branches live on origin right now, so the exposure is real, not theoretical.

Replace lines 3-6 with:

```markdown
[![CI](https://github.com/fingoldo/mlframe/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/fingoldo/mlframe/actions/workflows/ci.yml?query=branch%3Amaster)
[![MyPy](https://github.com/fingoldo/mlframe/actions/workflows/mypy-full.yml/badge.svg?branch=master&event=push)](https://github.com/fingoldo/mlframe/actions/workflows/mypy-full.yml?query=branch%3Amaster)
[![Black](https://github.com/fingoldo/mlframe/actions/workflows/black-filtered.yml/badge.svg?branch=master&event=push)](https://github.com/fingoldo/mlframe/actions/workflows/black-filtered.yml?query=branch%3Amaster)
[![sklearn 1.6-1.8](https://github.com/fingoldo/mlframe/actions/workflows/sklearn-matrix-ci.yml/badge.svg?branch=master&event=push)](https://github.com/fingoldo/mlframe/actions/workflows/sklearn-matrix-ci.yml?query=branch%3Amaster)
```

Note the `sklearn-matrix` relabel: the raw workflow name tells a visitor nothing, while
"sklearn 1.6-1.8" states the actual guarantee (the range is verified — `pyproject.toml` caps
`scikit-learn<1.9`, and the `sklearn_matrix` marker doc at `pyproject.toml:556` names 1.6-1.8).

One caveat I could not verify: whether `mypy-full.yml` and `black-filtered.yml` actually run on
`push` to master. If either is `pull_request`-only or schedule-only, `&event=push` renders
`no status`. Check each workflow's `on:` block before adding that parameter to it.

### F5 — LOW: badge label/link mismatch on the codecov flag badges

All three flag badges link to `codecov.io/gh/fingoldo/mlframe/flags` — the right destination, but the
same one for all three, so clicking any of them lands on an undifferentiated flags list. Minor; the
F2 restructure supersedes it.

---

## 2. Missing badges

Only badges backed by something that verifiably exists in this repo. Ranked by what they tell a
visitor who is deciding whether to depend on this.

**a. Supported Python versions** — the single most-asked question, currently answerable only by
reading prose at `README.md:67`. Must be a *static* badge, not the PyPI-derived one, since the package
is not published:

```markdown
[![Python 3.9-3.14](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://github.com/fingoldo/mlframe)
```

Verified against `requires-python = ">=3.9"` and the six `Programming Language :: Python :: 3.x`
classifiers at `pyproject.toml:39-44`.

**b. Ruff** — VERIFIED: `ruff==0.16.1` pinned in `[dev]`, and `ci.yml:527` runs a blocking
`ruff-blocking` job (plus `tests-ruff-blocking` at :567). The official badge:

```markdown
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
```

**c. CodeQL** — VERIFIED: `.github/workflows/codeql.yml` exists, `name: CodeQL`. Tells a visitor the
project is scanned for security defects:

```markdown
[![CodeQL](https://github.com/fingoldo/mlframe/actions/workflows/codeql.yml/badge.svg?branch=master)](https://github.com/fingoldo/mlframe/actions/workflows/codeql.yml)
```

I did not open `codeql.yml`'s `on:` block — confirm it runs on push to master before adding
`?branch=master`.

**d. pre-commit** — VERIFIED: `.pre-commit-config.yaml` exists and `README.md:77` instructs
`pre-commit install`.

```markdown
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
```

**e. Typed** — VERIFIED: `src/mlframe/py.typed` exists (empty marker file), it is shipped
(`pyproject.toml:509` package-data), the `Typing :: Typed` classifier is declared, and `mypy-full.yml`
gates it. The MyPy workflow badge already covers "we run mypy"; this covers the different claim
"your editor will see our types":

```markdown
[![types: py.typed](https://img.shields.io/badge/types-py.typed-blue.svg)](https://peps.python.org/pep-0561/)
```

### Badges NOT to add

- **PyPI version / downloads / wheel** — the package is (per the repo's own statements) not on PyPI.
  `img.shields.io/pypi/v/mlframe` would render `not found` or `invalid`, which is worse than absent.
  Add these the same day the first `release.yml` upload succeeds, not before.
- **conda / conda-forge** — no feedstock anywhere in this repo.
- **OpenSSF Scorecard** — no `scorecard.yml` workflow; the badge requires the action to have published
  results.
- **Codacy / Code Climate / Snyk / DeepSource** — no configuration for any of them in the repo.
- **Contributor Covenant** — no `CODE_OF_CONDUCT.md` (VERIFIED absent; `SECURITY.md` is absent too).
- **"Made with Python" / "PRs welcome" / star count** — decoration, no information.

---

## 3. README structure: the 30-second test

I ran the five questions against the current file.

| Question | Answerable in 30s? | Where it currently is |
|---|---|---|
| What is this? | Yes, barely | `README.md:14-19`, one 6-line sentence |
| Why use it over sklearn / AutoGluon? | **No** | nowhere |
| How do I install it? | **No — actively misleading** | `README.md:25-38`, see S1 |
| Minimal example? | **No** | `README.md:118`, below a 25-row table |
| Where are the docs? | Weakly | `README.md:21-23`, buried in prose |

### S1 — HIGHEST-IMPACT ITEM IN THIS WHOLE REVIEW: the install section contradicts itself

`README.md:29-30` states the package is not on PyPI and that you must clone two repos. Then
`README.md:46` says:

```
pip install mlframe[all,dev]                     # full install (recommended)
```

...which is a PyPI install, marked *recommended*, of a package the same section just said is not on
PyPI. Every other line in that block uses `-e "./mlframe[...]"`, so line 46 is the odd one out. A
visitor's literal first action — copy the recommended line — fails with `No matching distribution
found`, with no recovery hint. Fix (assuming still-not-on-PyPI):

```
pip install -e "./mlframe[all,dev]"              # full install (recommended)
```

Also `README.md:63`'s `pip install -e "./mlframe[all]"` largely duplicates that line's purpose; and
the Installation section leads with a dependency-provenance paragraph about `pyutilz` before it says
what to type. Lead with the command, explain the `pyutilz` constraint under it.

### S2 — HIGH: no "why", and no runnable first example above the fold

The file is 715 lines. Between the intro and the first code block sit the whole Installation section
and a 25-row module table — roughly 100 lines of reference material before a reader sees mlframe run.
And the first example is ~30 lines with two imports, a custom extractor, an RNG-built frame and a
comment about return-value shape.

Recommended header order (content already exists in the file; this is reordering plus two new short
blocks):

1. Title + a **one-line** tagline. Something like: *Train, calibrate, ensemble and diagnose a dozen
   tabular model families on one dataset, from one function call.*
2. Badge block (F2/F4 versions).
3. **New: `## Why mlframe`** — 4-5 bullets, each a claim a competitor cannot make. Candidates already
   substantiated elsewhere in this README: one entry point across sklearn/CatBoost/LightGBM/XGBoost/
   HGB/Lightning; polars-native with no pandas round-trip on models that accept Arrow
   (`README.md:666-668`); calibration selected OOF-only with a bootstrap-CI tiebreak
   (`README.md:402-406`); diagnostics default-ON with the charts to match; scikit-learn 1.6-1.8
   pinned in CI (`README.md:669-672`).
4. `## Quickstart` — the *shortest* thing that produces a number. Move the current
   `train_mlframe_models_suite` block up here; consider trimming to fit + one printed diagnostic, and
   leaving the "`models` is keyed by target-type" explanation as one line under it.
5. `## Installation`.
6. `## Documentation` — a real section with three links, not a sentence: the site (F3 permitting), the
   [guide index](docs/README.md), the [gallery](docs/gallery/index.md).
7. Everything currently there, unchanged, starting with the module table.

### S3 — MEDIUM: no table of contents in a 715-line file

Add one right after the badges. GitHub renders a built-in outline behind a hamburger, but it is easy
to miss and does not survive to non-GitHub renderers (PyPI, and the mkdocs site — both of which
consume this same file).

### S4 — MEDIUM: `docs/README.md` mixes user guides with research notes under one index

`docs/README.md` is a good index, but "User-facing guides" (14 rows: calibration policy, feature
handling, error decoding) sits directly above "Internal / research notes" containing benchmark
pre-registrations, literature surveys and GPU roadmaps. A user looking for the calibration guide has
to scan past `WAVE5_GPU_ROADMAP.md` and `SHAP_PROXIED_FS_GAME_THEORY.md`. The section headings are
already there and honest ("These are working notes... not user API documentation") — the fix is
ordering and visual separation, not new content: put the research block behind a
`<details><summary>Research and design notes</summary>` fold so the user-facing table is the whole
visible page.

### S5 — LOW: CHANGELOG `[Unreleased]` has grown into the release notes

`CHANGELOG.md` opens well (Keep a Changelog + SemVer, plus an explicit "intentionally lean and
user-focused" statement). But `[Unreleased]` currently holds ~20 `### Added` entries, several of them
4-6 lines of dense prose about benchmark methodology, and the section has **two** `### Fixed`
headings — one before `### Added`, one after — so a reader scrolling sees "Fixed" twice under one
version. Merge the duplicate heading. The prose length is a judgement call, but it is at odds with the
file's own stated contract. Also worth confirming: `version.py` says `0.9.0`, and no `## [0.9.0]`
section appears in the first 40 lines — check that released versions actually have sections.

---

## 4. `pyproject.toml [project]` metadata

The metadata here is in unusually good shape: `keywords` (17, well-chosen), `classifiers` (all six
Python versions, three `Topic ::` entries, `Typing :: Typed`, `Development Status :: 4 - Beta`),
`license`/`license-files` in the modern PEP 639 form, and all five useful `[project.urls]`
(Homepage / Documentation / Repository / Issues / Changelog). Three real gaps:

### P4-a — HIGH: the README's relative links and images will all break on the PyPI page

`readme = "README.md"` means this exact file becomes the PyPI long description. PyPI does **not**
resolve repo-relative paths. Verified relative references that would break there:

- six images: `docs/gallery/binary/binary_full.png`, `.../regression/regression_full.png`,
  `.../drift/psi_heatmap.png`, `.../binary/calibration_reliability.png`,
  `.../shap_panels/shap_shap_beeswarm.png`, `.../model_comparison/model_comparison.png`
  (`README.md:581-585`) — these render as broken-image icons, the most visible possible defect on a
  project page;
- links to `CHANGELOG.md`, `docs/README.md`, `docs/examples/composite_targets.md`,
  `docs/composite_targets_tutorial.ipynb`, `docs/visualization.md`, `docs/gallery/index.md`,
  `docs/ENVIRONMENT_VARIABLES.md`, `CONTRIBUTING.md`, `LICENSE`, and
  `.github/workflows/sklearn-matrix-ci.yml` — all 404 from PyPI.

Fix: make image sources and cross-file links absolute. Images:

```markdown
![binary_full](https://raw.githubusercontent.com/fingoldo/mlframe/master/docs/gallery/binary/binary_full.png)
```

Links:

```markdown
[docs/visualization.md](https://github.com/fingoldo/mlframe/blob/master/docs/visualization.md)
```

Both forms still work on GitHub, so there is no dual-maintenance cost. This matters the moment
`release.yml` first succeeds, and it is cheap to fix before then. (Caveat: I verified `docs/gallery/`
exists as a directory but did not confirm each of the six PNGs at its exact path.)

### P4-b — MEDIUM: `dependencies` contains `pyutilz>=1.0.0`, which (per the repo) is not on PyPI

`pyproject.toml:151` declares a hard runtime dependency on `pyutilz>=1.0.0`, with an adjacent comment
(:142-150) saying it is not yet published and that pip must resolve it from the user's already-installed
environment. If mlframe is ever uploaded before pyutilz is, every `pip install mlframe` fails at
resolve time on a clean machine — exactly what `release.yml`'s own header comment warns about. Nothing
to fix in the metadata itself; flagged because it gates every PyPI-facing recommendation in this
report. **Verify pyutilz's PyPI status before the first release.**

### P4-c — LOW: `description` is long for a PyPI summary

`pyproject.toml:8` is 178 characters. PyPI shows the summary under the project name and in search
results, where the tail gets cut. Suggested replacement, keeping the distinguishing part first:

```toml
description = "One-call training, calibration, ensembling and diagnostics for tabular ML across sklearn, CatBoost, LightGBM, XGBoost and PyTorch."
```

(129 chars.) The dropped terms — feature-engineering, feature-selection, ranking, quantile-regression —
are all already in `keywords`, so nothing becomes unsearchable.

### P4-d — informational, do NOT "fix"

There is no `License :: OSI Approved :: MIT License` classifier, and there should not be: with the
PEP 639 `license = "MIT"` SPDX expression at :10, modern setuptools **rejects** a license classifier
alongside it. The current form is correct.

---

## Suggested badge block, all fixes applied

Assumes F3 resolves in favour of the live site, and that the `event=push` caveat in F4 checks out for
each workflow.

```markdown
[![CI](https://github.com/fingoldo/mlframe/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/fingoldo/mlframe/actions/workflows/ci.yml?query=branch%3Amaster)
[![MyPy](https://github.com/fingoldo/mlframe/actions/workflows/mypy-full.yml/badge.svg?branch=master&event=push)](https://github.com/fingoldo/mlframe/actions/workflows/mypy-full.yml?query=branch%3Amaster)
[![CodeQL](https://github.com/fingoldo/mlframe/actions/workflows/codeql.yml/badge.svg?branch=master)](https://github.com/fingoldo/mlframe/actions/workflows/codeql.yml)
[![sklearn 1.6-1.8](https://github.com/fingoldo/mlframe/actions/workflows/sklearn-matrix-ci.yml/badge.svg?branch=master&event=push)](https://github.com/fingoldo/mlframe/actions/workflows/sklearn-matrix-ci.yml?query=branch%3Amaster)
[![coverage](https://img.shields.io/codecov/c/github/fingoldo/mlframe/master?label=coverage)](https://codecov.io/gh/fingoldo/mlframe)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Python 3.9-3.14](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://github.com/fingoldo/mlframe)
[![types: py.typed](https://img.shields.io/badge/types-py.typed-blue.svg)](https://peps.python.org/pep-0561/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://fingoldo.github.io/mlframe/)
```

The `Black` badge is dropped from the header on purpose — Ruff already signals "this codebase is
linted", and the header is at eleven badges as it stands. Keep it if you want the formatting claim
explicit; it is not broken, only redundant.

---

## Incidental finding (outside the requested scope)

`codecov.yml`'s comment says the 10 uploads come from "the single python-3.11 leg", but `ci.yml:50`
sets `REPRESENTATIVE_PYTHON: "3.12"`. The `after_n_builds: 10` value itself is still correct (it
tracks the shard count, not the version), so this is a stale comment, not a live misconfiguration —
but the comment is what a future reader will trust when the shard count next changes.
