# Cross-cutting: open-source project hygiene & packaging -- mlframe audit

## Scope

Files actually opened and read this session (relative to repo root):

- `README.md` (706 lines, full read)
- `CONTRIBUTING.md` (97 lines, full read)
- `LICENSE` (21 lines, full read)
- `CHANGELOG.md` (163 lines, full read)
- `MANIFEST.in` (23 lines, full read)
- `.gitattributes` (4 lines, full read)
- `.gitignore` (220 lines, full read)
- `mkdocs.yml` (87 lines, full read)
- `src/mlframe/version.py` (8 lines, full read)
- `.github/PULL_REQUEST_TEMPLATE.md` (17 lines, full read)
- `.github/ISSUE_TEMPLATE/bug_report.md` (34 lines, full read)
- `.github/ISSUE_TEMPLATE/feature_request.md` (23 lines, full read)
- `requirements-dev.txt` (16 lines, full read)
- `pyproject.toml` (959 lines total -- lines 1-436 read in full, covering `[project]` metadata, `dependencies`, `[tool.uv]`, all `[project.optional-dependencies]` extras, `[project.scripts]`, `[project.urls]`, `[tool.setuptools]`; lines 437-959, which are `tool.ruff`/`tool.mypy`/`tool.pytest`/other build-tool configuration outside this cluster's "packaging metadata" scope, were not read)
- `docs/README.md` (40 lines, full read)
- `docs/examples/README.md` (19 lines, full read)
- `docs/gallery/index.md` (306 lines total -- lines 1-40 read directly; the rest of the category list was cross-checked structurally against the on-disk `docs/gallery/*/` folders and PNG counts via directory listing, not read line-by-line)
- `docs/SELECTION_BIAS.md` (212 lines total -- lines 1-60 read)
- `docs/honest_diagnostics_guide.md` (113 lines total -- lines 1-60 read)
- `docs/calibration_policy.md` (74 lines, full read)
- `docs/sidecar_verification_guide.md` (67 lines, full read)
- `docs/dummy_baselines_guide.md` (474 lines total -- lines 1-40 read)

Also inspected (directory listings only, not file-by-file content review): `docs/` top-level directory listing, `docs/examples/` directory listing, `docs/gallery/` directory listing (all ~27 subfolders + asset files), `.github/` and `.github/workflows/` directory listings (workflow *file contents* were not read -- out of this cluster's scope -- only used to confirm README badge links point at files that exist), `.github/ISSUE_TEMPLATE/` directory listing.

For two doc-accuracy cross-checks, briefly grepped (not fully read) `src/mlframe/calibration/policy.py` (confirmed `pick_best_calibrator`'s signature matches `docs/calibration_policy.md`) and `src/mlframe/utils/safe_pickle.py` (confirmed `MLFRAME_ALLOW_UNVERIFIED_PICKLE` / `safe_load` match `docs/sidecar_verification_guide.md`) -- these are src files outside this cluster's normal scope, read only for the specific purpose of verifying a documentation claim, per the audit brief's allowance to read a referenced symbol for context.

Real total: **22 files reviewed** (13 fully, 1 mostly-full with an explicitly out-of-scope tail unread, and 5 docs partially read as noted), **2255 lines actually read**, out of the larger byte totals of the partially-read files.

## Findings

| ID | Severity | Category | File:Line | Summary |
|---|---|---|---|---|
| F1 | P1 | docs-hygiene | CHANGELOG.md:8-119 | CHANGELOG.md's own stated policy ("intentionally lean and user-focused... for the full engineering record see git history") is violated by its own `[Unreleased]` section, a ~9,000-word wall of per-round production-profiling narrative. |
| F2 | P2 | packaging-metadata | MANIFEST.in:5-6 | `MANIFEST.in` `include`s `SECURITY.md` and `CODE_OF_CONDUCT.md`, neither of which exists anywhere in the repo. |
| F3 | P2 | packaging-metadata | pyproject.toml:86 vs pyproject.toml:281 | `antropy` is declared twice with two different version floors: as a hard core dependency (`antropy>=0.1`, line 86, with a comment explaining it is imported unconditionally) and again inside the optional `signal` extra (`antropy>=0.1.4`, line 281), which misleadingly implies it is opt-in. |
| F4 | P2 | docs-accuracy | README.md:39-41 | README's "core install pulls only the lightweight stack (numpy, pandas, polars, scipy, scikit-learn, pyarrow, joblib, tqdm, pydantic, numba)" omits 11 other hard/hard-adjacent core dependencies declared in `pyproject.toml`'s `[project.dependencies]`, including a full plotting stack (matplotlib + pillow). |
| F5 | P2 | docs-coverage | docs/README.md:1-40 | `docs/README.md`'s own "documentation index" tables never mention `visualization.md`, `SHAP_PROXIED_FS_GAME_THEORY.md`, or `gallery/index.md`, even though all three are live docs wired into `mkdocs.yml`'s nav under "Topics". |
| F6 | P2 | docs-staleness | docs/gallery/index.md:6-28 | The gallery's own "Total images: 40 across 19 categories" summary and "Contents" list are stale: 7 gallery subfolders holding 8 PNGs (`calibration_by_feature`, `calibration_heatmap_2d`, `fairness_calibration`, `pdp_2d`, `risk_coverage`, `shap_interactions` [2 images], `shap_per_instance`) exist on disk and are neither listed nor counted; the real total is 49 PNGs, not 40. |
| F7 | P2 | docs-quality | docs/calibration_policy.md:1, docs/honest_diagnostics_guide.md:1 | Both doc titles carry a bare internal tag ("Calibration policy guide (AP12)", "Honest diagnostics guide (AP13)") that is never defined anywhere in `docs/` -- meaningless to an external reader. |

### F1 -- CHANGELOG.md contradicts its own stated policy

`CHANGELOG.md` opens with: "This file is intentionally lean and user-focused. For the full engineering record (per-commit kernel tuning, profiling numbers, audit notes) see the git history." (lines 8-10). The very next section, `[Unreleased]`, is exactly the opposite: individual bullets run to multiple paragraphs of round-by-round A/B wall-clock numbers ("round 14... 726.1s", "round 18... 743.3s"), raw cProfile tottime/cumtime figures, kernel-launch counts, ncu/nsys profiler readouts, and references to internal audit-finding IDs ("MRMR audit finding #5"). This is precisely the "per-commit kernel tuning, profiling numbers, audit notes" the file's own header says belongs in git history instead. It also departs from the "Keep a Changelog" convention the file cites (a changelog entry should be a short, human-scannable, user-facing description of what changed) -- a downstream consumer trying to answer "what's different for me in this release" has to wade through engineering narrative to find it. Suggested fix: trim `[Unreleased]` entries to one or two user-facing sentences each (what changed, why it matters to a caller), and move the detailed profiling/round-by-round narrative to the referenced git history (which the header already says is the right home for it) or an audit doc under `audit`/`audits/`.

### F2 -- MANIFEST.in references two files that don't exist

`MANIFEST.in` lines 5-6 read `include SECURITY.md` and `include CODE_OF_CONDUCT.md`. Neither file exists anywhere in the repository (`ls SECURITY.md CODE_OF_CONDUCT.md` at repo root: both "No such file or directory"). Running `python -m build --sdist` (or classic `setup.py sdist`) with this MANIFEST.in will emit a "warning: no files found matching 'SECURITY.md'" / same for CODE_OF_CONDUCT.md for each entry -- noise at best, and a false signal to anyone reading MANIFEST.in as documentation that these governance files exist. `CONTRIBUTING.md`'s own "Security" section (lines 91-93) inlines vulnerability-reporting instructions instead of pointing at a `SECURITY.md`, so the file was evidently planned but never created (or was later deleted without updating MANIFEST.in). Suggested fix: either write the two missing files (a `CODE_OF_CONDUCT.md`, e.g. Contributor Covenant, is a common gap for a project this size and worth adding; `SECURITY.md` could just promote the existing inline text from `CONTRIBUTING.md`) or drop the two dead `include` lines from `MANIFEST.in`.

### F3 -- antropy declared as both a hard dependency and an optional extra with a different floor

`pyproject.toml` line 86 lists `antropy>=0.1` inside `[project.dependencies]` (the unconditional core install), with an adjacent comment (lines ~83-85) explicitly stating it "would otherwise raise ImportError on `import mlframe`" if omitted -- i.e. it is mandatory. Line 281, inside the `signal` optional extra, lists `antropy>=0.1.4` again. Because it's already a hard dependency, listing it a second time inside an "extra" is misleading: a user reading the extras table (or README's mirrored extras table, `README.md:56`, `# antropy + astropy + pywavelets + ruptures`) will believe `antropy` only arrives with `pip install mlframe[signal]`, when in fact it's always installed and `[signal]` only adds the other three packages plus bumps antropy's floor slightly. This doesn't break installs (pip resolves the union), but it's an inaccurate, self-contradicting metadata declaration. Suggested fix: drop the duplicate `antropy` entry from the `signal` extra (or, if the higher floor is genuinely needed for the signal-processing code paths, bump the core dependency's floor to `>=0.1.4` instead and remove it from the extra).

### F4 -- README's "lightweight core install" list omits 11 hard dependencies

`README.md:39-41` states: "The core install pulls only the lightweight stack (numpy, pandas, polars, scipy, scikit-learn, pyarrow, joblib, tqdm, pydantic, numba)." Cross-checking against `pyproject.toml`'s actual `[project.dependencies]` (lines 55-107) shows the real hard-dependency set additionally includes `portalocker`, `dill`, `orjson`, `psutil`, `matplotlib`, `pillow`, `antropy`, `category-encoders`, `expiringdict`, `pympler`, `xxhash`, and `pyutilz` (the last is separately called out earlier in the same README section, so its omission from this particular list is not itself misleading -- but the other 11 are not mentioned anywhere near this claim). Notably `matplotlib`+`pillow` is a full plotting stack that most readers would expect to be gated behind the `viz` extra (which itself additionally pulls `plotly`/`seaborn`/`altair`/etc.), not silently present in "just the lightweight stack." Suggested fix: either update the enumerated list to the real core set, or soften the claim to "a compact stack (see `pyproject.toml`'s `[project.dependencies]` for the exact list)" so it can't drift out of sync again.

### F5 -- docs/README.md's index omits three live docs

`docs/README.md` presents itself as the documentation index ("Conceptual guides and design notes for mlframe... this index covers the deeper material in the docs root") with two tables ("User-facing guides", "Internal / research notes") that between them list 19 files. Grepping the file for `visualization`, `SHAP_PROXIED`, and `gallery` returns zero matches: `visualization.md`, `SHAP_PROXIED_FS_GAME_THEORY.md`, and `gallery/index.md` all exist in `docs/`, are all wired into `mkdocs.yml`'s nav under "Topics" (`mkdocs.yml:84-87`), and are all reachable from the top-level `README.md`'s "Visualization & Diagnostics" section -- but none of them appear in `docs/README.md`'s own index tables, so a reader arriving at `docs/README.md` (the promised "Full guide index" per `README.md:20-22`) and scanning only that page would not discover them. Suggested fix: add the three missing rows to `docs/README.md`'s "User-facing guides" table.

### F6 -- gallery index.md summary/contents are stale vs the actual gallery directory

`docs/gallery/index.md` states "Total images: 40 across 19 categories" and lists exactly 19 category anchors in its "Contents" section (lines 10-28). The actual `docs/gallery/` directory contains 27 subfolders; 7 of them (`calibration_by_feature`, `calibration_heatmap_2d`, `fairness_calibration`, `pdp_2d`, `risk_coverage`, `shap_interactions`, `shap_per_instance`) hold PNGs (1 each except `shap_interactions`, which holds 2) that are absent from both the Contents list and the total count. A direct count of every `.png` under `docs/gallery/` (excluding the `.benchmarks` cache folder) returns 49, not 40. The page's own instructions say it is "regenerated with `python scripts/render_gallery.py`" -- it simply hasn't been re-run since these 7 categories' charts were added. Suggested fix: re-run `scripts/render_gallery.py` to regenerate `docs/gallery/index.md` from the current on-disk set.

### F7 -- cryptic "(AP12)"/"(AP13)" tags in public doc titles

`docs/calibration_policy.md:1` reads `# Calibration policy guide (AP12)` and `docs/honest_diagnostics_guide.md:1` reads `# Honest diagnostics guide (AP13)`. Grepping all of `docs/` for the pattern `\(AP\d+\)` finds only these two occurrences -- the tag is not a recurring, explained taxonomy (no "what is AP12" glossary entry anywhere in `docs/`), just an internal work-item label that leaked into the public-facing doc title. Low-severity, but it's the kind of thing an external reader stumbles on and can't resolve. Suggested fix: drop the parenthetical from both titles (or move it into an HTML comment if it's useful for internal cross-referencing).

## Proposals

| ID | Category | File:Line | Summary |
|---|---|---|---|
| PR1 | governance | (repo root) | Add a `CODE_OF_CONDUCT.md` (e.g. Contributor Covenant) -- currently missing, which is common for a project this size, but `MANIFEST.in` already expects one (see F2), suggesting it was on the roadmap. |
| PR2 | test-coverage | (repo root docs) | No automated check verifies `mkdocs.yml`'s `nav`/`not_in_nav` entries against the actual `docs/` file listing, or that `docs/README.md`'s hand-maintained index stays in sync with `docs/`'s contents (F5) -- a small script (glob `docs/**/*.md`, diff against the union of `nav` + `not_in_nav` + `docs/README.md` links) run in CI (or as a pre-commit hook, given `.pre-commit-config.yaml` already exists) would catch both F5-style and orphan-nav-entry regressions automatically. |
| PR3 | packaging | pyproject.toml | Consider a `pip-compile`/`uv lock`-driven or scripted consistency check that flags a dependency appearing in both `[project.dependencies]` and any `[project.optional-dependencies]` entry (the class of bug in F3) -- cheap to add given `import-linter`/`pydoclint`/`semgrep` are already wired into the `dev` extra and pre-commit setup for other structural checks. |
| PR4 | release-process | CHANGELOG.md, src/mlframe/version.py | The project has been on `0.9.0` (per `version.py`) with a large, still-growing `[Unreleased]` section for a long stretch of active, production-validated work (the CHANGELOG's own entries describe results already deployed against real production runs). Worth considering cutting a `0.10.0` (or `1.0.0`, given the maturity implied by the fix log) release and archiving the current `[Unreleased]` content, both to give users a stable reference point and to make F1's proposed trim-down natural to do at the same time. |
| PR5 | gitattributes | .gitattributes | Only `*.sh` gets a forced `eol=lf` rule. Given this codebase's own memory/CLAUDE.md rules flag CRLF-mangling as a recurring pain point on this Windows dev box, and CI runs on Linux runners (per the workflow badges), extending the same `eol=lf` treatment to `*.py`/`*.yml`/`*.yaml` (or a blanket `* text=auto eol=lf`) would close off the same class of "CRLF shebang/script breaks on Linux CI" bug the existing `.sh` rule was added for, before it recurs on another file type. |

## Coverage notes

- `pyproject.toml` lines 437-959 (the `tool.ruff` / `tool.mypy` / `tool.pytest` / other build-tooling configuration) were not read -- the cluster brief scopes this file to "packaging metadata: name/version/description/classifiers/dependency declarations/extras/entry points," all of which live in lines 1-436, and the remainder is general repo tooling config outside that definition, not packaging metadata.
- `docs/gallery/index.md` was read for its header/summary/contents (lines 1-40) and then cross-checked structurally against the real directory tree rather than read line-by-line for all 306 lines -- the remaining ~266 lines are a long, repetitive list of `### <chart_name>` / one-line caption / `![...]()` triples per category, which the F6 finding already establishes is stale in aggregate; reading every remaining entry individually would not change that verdict.
- `docs/SELECTION_BIAS.md` (212 lines), `docs/honest_diagnostics_guide.md` (113 lines), and `docs/dummy_baselines_guide.md` (474 lines) were each spot-checked on their opening section (~40-60 lines) rather than read in full, per the cluster brief's "spot-check 5-8 of the topic docs for staleness" instruction (this session spot-checked 6: these three plus `calibration_policy.md`, `sidecar_verification_guide.md`, and `gallery/index.md`, the latter two read in full/near-full). The unread tails of these three files were not cross-checked against current source code line-by-line; a deeper pass on those specific files (rather than the sample taken here) could surface additional staleness this session did not catch.
- `.github/workflows/*.yml` file *contents* were not read (out of this cluster's declared scope, which lists only `.github/ISSUE_TEMPLATE/*` and `.github/PULL_REQUEST_TEMPLATE.md`) -- only the directory listing was used, to confirm the workflow files README.md's badges link to (`ci.yml`, `mypy-full.yml`, `black-filtered.yml`, `sklearn-matrix-ci.yml`, `numba-coverage.yml`) actually exist on disk. Badge *behavior* (whether they currently pass) was not and could not be verified from a local, offline read.
- `.pre-commit-config.yaml` (39,679 bytes) is referenced by `CONTRIBUTING.md` and `MANIFEST.in`'s neighborhood but is not itself in this cluster's file list, so it was not opened.
- No `docs/_build/` or built mkdocs site was available to actually run `mkdocs build --strict` and get a tool-verified list of dead links/orphans; all link-checking in this report (F5, F6) was done by direct grep/diff against the file tree, which catches missing-from-index and stale-count problems but would not catch, e.g., a broken relative link inside prose text that a real `mkdocs build --strict` run would flag.
