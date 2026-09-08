# Static-analysis & quality-gate review — mlframe

Date: 2026-09-08. Read-only audit. Every claim below is tagged **MEASURED** (I ran it / read the exact
line) or **INFERENCE** (reasoning from what I read, not executed).

Environment used for measurements: this worktree, `python` 3.14.3, ruff 0.15.22, mypy 2.1.0,
black 26.5.1, vulture 2.16, interrogate 1.7.0, deptry 0.25.1, codespell 2.4.2, yamllint 1.38.0.

Files read in full or in the relevant part: `pyproject.toml` (all `[tool.*]`), `.pre-commit-config.yaml`,
`.yamllint`, `codecov.yml`, `tests/ruff.toml`, `tests/interrogate.toml`, `.github/workflows/ci.yml`,
`mypy-full.yml`, `deep-nightly.yml`, `py-ci-shared/configs/ruff-base.toml`,
`py-ci-shared/.github/workflows/lint-blocking.yml`, `tests/test_meta/` (listing + ~12 gates in detail,
`conftest.py`, `regen_baselines.py`, `BASELINES_README.md`).

This setup is well above average — the density of recorded rationale is unusual and most odd-looking
settings are correct and already explained. The findings below are the ones that survived reading the
comments.

---

## 1. MISCONFIGURATION

### 1.1 Every CI lint tool runs unpinned via `uvx`, while the repo pins ruff/black obsessively — HIGH

**MEASURED.** `py-ci-shared/.github/workflows/lint-blocking.yml` (which `ci.yml`'s `lint-blocking` and
`tests-lint-blocking` jobs call) invokes, with no version specifier:

- L88 `uvx --from actionlint-py actionlint`
- L97 `uvx zizmor .`
- L105 `uvx codespell --toml pyproject.toml ...`
- L110 `uvx yamllint .github/workflows .pre-commit-config.yaml`
- L115 `uvx bandit -r "${INPUTS_SOURCE_PATH}" -ll`
- L125 `uvx vulture ... --min-confidence 80`
- L134/136 `uvx interrogate ...`
- L165 `uv pip install --system deptry`

All eight are **blocking** gates. `uvx` resolves the newest release each run, so any of these can turn a
green master red overnight on an unrelated upstream release, or (worse for a gate) silently *drop* a
check when upstream re-classifies a rule. This is precisely the drift class the repo already wrote a
meta-test for (`tests/test_meta/test_ruff_pin_consistency.py`, whose docstring calls it "the exact
stale-pin-drift class the prior audit's F1 finding already fixed once") — but only ruff and black got
the treatment.

**Do:** pin every one (`uvx zizmor==X.Y.Z`, `uvx codespell==2.4.2`, ...) as reusable-workflow inputs with
defaults, and extend `test_ruff_pin_consistency.py` into a general "every tool version pin agrees across
pyproject dev extra / .pre-commit-config.yaml / the CI workflow inputs" gate.

### 1.2 The pinned ruff version is not the ruff that actually runs locally — HIGH

**MEASURED.** `pyproject.toml:440` pins `ruff==0.16.1`; `.pre-commit-config.yaml` has `rev: v0.16.1` on
all three `astral-sh/ruff-pre-commit` entries; `test_ruff_pin_consistency.py` asserts those texts agree.
But the blocking hooks override the upstream hook with `entry: python -m ruff check --ignore C901` +
`language: system` (lines 98-122 and 432-441), so pre-commit never builds the pinned isolated env — it
runs the ambient interpreter's ruff. In this checkout that is **ruff 0.15.22**, not 0.16.1.

The `rev:` is therefore decorative for the blocking hooks, and the meta-test verifies agreement between
two strings while the executing binary is a third, unchecked value. Net effect: a commit can pass locally
under 0.15.22 and fail CI's `uvx ruff==0.16.1`, i.e. exactly the failure mode the pin exists to prevent.

**Do:** have the meta-test (or a pre-commit hook) additionally assert
`importlib.metadata.version("ruff") == <the pin>`, and/or reinstall dev extras. A `language: system`
hook needs a runtime version assertion, not a `rev:`.

### 1.3 `mypy` is a blocking 0-error gate with an open version range — HIGH

**MEASURED.** `.github/workflows/mypy-full.yml:33` pins `mypy-version: "2.1.0"`. The local blocking hook
`mypy-full` (`.pre-commit-config.yaml:288`) runs `python -m mypy src/mlframe` from the checkout's own env,
and the dev extra declares only `"mypy>=1.0"` (`pyproject.toml:461`). mypy 1.x to 2.x changed defaults and
the error set substantially; a range this wide on a whole-project zero-error gate is the same drift the
ruff pin comment spends eight lines warning about. (Locally it happens to resolve to 2.1.0 today —
coincidence, not constraint.)

**Do:** `"mypy==2.1.0"` in the dev extra, covered by the pin-consistency meta-test.

### 1.4 Five blocking local hooks invoke tools that are declared nowhere — MEDIUM

**MEASURED.** Searching `pyproject.toml` for `"interrogate` / `"vulture` / `"codespell` / `"deptry` /
`"yamllint` returns **0 hits each**. Yet `.pre-commit-config.yaml` runs, as `language: system` blocking or
manual hooks: `python -m interrogate` (L390, L469), `python -m vulture` (L367), `python -m codespell_lib`
(L133, L478), `python -m deptry` (L415), `python -m yamllint` (L491). `uvx zizmor .` (L507) additionally
requires `uv` on PATH.

A fresh `pip install -e ".[all,dev]"` therefore does **not** give a contributor a working hook set (it
fails closed with ModuleNotFoundError, so nothing passes silently — but the documented onboarding path is
broken), and none of these five has any version constraint anywhere.

**Do:** add pinned entries to the `dev` extra (they also then get real pins for 1.1). The `deptry`
`DEP002` ignore list already anticipates this pattern for ruff/black/mypy/bandit/pre-commit/pip-audit;
extend it with the five names.

### 1.5 `detect-secrets`, `shellcheck` and the file-hygiene hooks exist only locally, never in CI — MEDIUM

**MEASURED.** `grep -rl 'shellcheck|detect-secrets' .github/workflows/` returns **no matches**;
`py-ci-shared/.github/workflows/lint-blocking.yml` mentions shellcheck only inside `# shellcheck disable`
directives, and detect-secrets not at all. So `detect-secrets` (`.secrets.baseline`, 27 plugins, 8 files
with results), `shellcheck`, `check-merge-conflict`, `check-added-large-files`, `mixed-line-ending`,
`check-yaml/toml/json`, `end-of-file-fixer`, `trailing-whitespace` are enforced **only** on a machine with
hooks installed.

For a public repo taking outside PRs, secret scanning that a `--no-verify` or a fork PR bypasses is not a
gate. This also inverts the config header's own stated policy (CI-blocking implies local hook); the
converse direction was never checked.

**Do:** add one cheap CI job running exactly the hooks with no CI counterpart —
`pre-commit run --all-files detect-secrets shellcheck check-merge-conflict check-added-large-files
mixed-line-ending check-yaml check-toml check-json`. (Cannot verify from the repo whether GitHub
push-protection is enabled on the org — **INFERENCE** that it is not a substitute for the baseline-aware
detect-secrets run.)

### 1.6 No coverage gate exists anywhere — MEDIUM

**MEASURED.** `codecov.yml` is 24 lines and sets `coverage.status.project: off` and `patch: off`, with no
comment explaining either. `[tool.coverage.report]` (pyproject L615-632) has **no `fail_under`**. So
coverage is collected across 10 shards, merged, uploaded, badge-rendered — and can drop to any value
without failing anything.

Given the amount of machinery invested in *collecting* it (a whole `codecov-full.yml`, a
`numba-coverage.yml`, the `after_n_builds: 10` lockstep note), having zero threshold is the largest
"measured but not gated" item here. `patch` status in particular is the high-signal, low-noise one: it
only asks "is the code this PR added tested", which is exactly the regression this suite cannot otherwise
see.

**Do:** turn on `coverage.status.patch` with an explicit target (start `informational: true` for one
release, then flip) and set a `fail_under` in `[tool.coverage.report]` slightly below the current measured
total. The target must come from a measured full-suite number — I did not run the suite, so I have no
number to propose.

### 1.7 `vulture --min-confidence 80` currently reports nothing at all — MEDIUM

**MEASURED.** `python -m vulture src/mlframe scripts/vulture_whitelist.py --min-confidence 80` produces
**0 lines of output**. The same scan at `--min-confidence 60` produces **1071 findings**.

Vulture assigns 90% to unused imports and 100% to unreachable code; unused
functions/classes/methods/attributes/variables sit at 60%. At threshold 80 the gate therefore only ever
fires on unused imports and unreachable code — it is a live tripwire (not a gate that *cannot* fail), but
its stated purpose ("dead-code scan") is almost entirely delegated to the 60%-confidence band it excludes.
The uncalled-function class *is* separately covered by `test_shared_uncalled_functions.py` (baseline: 929
entries), so the real uncovered residue is **unused attributes / unused class-scope variables** — a
genuine ML-config bug class (a config field set on `self` that nothing ever reads).

**Do:** either accept this and rename the hook so it stops implying broader coverage, or run a second
vulture pass at `--min-confidence 60` restricted to the *attribute* findings with its own baseline. Do not
simply lower the threshold — 1071 findings need a baseline first.

### 1.8 The two vulture runs contradict the comment that justifies them — LOW/MEDIUM

**MEASURED.** `.pre-commit-config.yaml:352-361` argues at length that vulture must be whole-repo because
"scoping its invocation to only the touched files hides every OTHER file that references a symbol", and
L425-431 states vulture "must scan src/mlframe and tests together to be meaningful". But the actual entry
is `python -m vulture src/mlframe scripts/vulture_whitelist.py` (no `tests`), and CI runs vulture **twice,
separately** — `lint-blocking` with `source-path: src/mlframe`, `tests-lint-blocking` with
`source-path: tests` — each blind to the other tree. The stated cross-file blind spot is reintroduced by
the split. It does not bite today only because both runs are empty at threshold 80 (see 1.7).

**Do:** make it one invocation over `src/mlframe tests` with both whitelists, or delete the two comments
that claim a property the config does not have.

### 1.9 `doctest_optionflags` is configured but no doctest ever runs — LOW

**MEASURED.** `pyproject.toml:536` sets `doctest_optionflags = "NORMALIZE_WHITESPACE ELLIPSIS"`. Searching
`doctest` across `pyproject.toml` and every workflow finds that line and nothing else. No
`--doctest-modules`, no `--doctest-glob`, no doctest job. This is a setting for a gate that does not
exist; docstring examples in a ~1550-file library are unverified.

**Do:** either drop the line, or add a `--doctest-modules` leg over a scoped subpackage (see 3.4).

### 1.10 The mccabe debt comment's number has drifted 70 to 92 — LOW

**MEASURED.** `pyproject.toml:975-989` states threshold 40 "keeps 70 findings ... real, tracked debt".
`ruff check src/mlframe --statistics` reports **92 C901**. The tracked debt grew ~31% since the note was
written and nothing notices, because C901 is `--ignore`d in every blocking invocation (correctly, per the
recorded decision) and the advisory job's output is read by nobody in particular.

**Do:** if the debt is genuinely tracked, ratchet it — a meta-test pinning the C901 count at its current
value (the same baseline pattern already used 25+ times in `tests/test_meta/`) turns a silently-growing
number into a visible one, at zero false-positive cost.

### 1.11 `LOC_BUDGET_EXEMPT` has no staleness check, and its own docstring is wrong — LOW

**MEASURED.** `tests/test_meta/test_no_file_over_1k_loc.py:9` says "The exempt list is empty by design" —
the set has 13 entries. More importantly, nothing asserts that each exempt path (a) still exists and
(b) still exceeds `LOC_LIMIT`. A file that gets renamed, deleted, or successfully carved leaves a dead
exemption behind, and a *new* file created at that exact path inherits the exemption silently.

**Do:** add two assertions to the same test — every exempt path resolves to a real file, and every exempt
file is actually over budget (otherwise: "drained, remove from the exempt set").

### 1.12 Baselines drain-warn but never require draining — MEDIUM (structural)

**MEASURED.** `test_no_mutable_defaults.py:134-146` (and the identical shape in
`test_public_docstrings.py:89`, `test_public_annotations.py:129`, `test_no_bare_except.py:167`) computes
`fixed = baseline - current` and writes it to **stderr only** — the test still passes. So a baseline entry
that has been fixed stays in the file forever, and a later *regression at that same `file:line` key* is
invisible to the gate.

Current sizes (**MEASURED**, entry counts): `_code_audit_tests_baseline.json` 1757,
`_nondiscriminating_assert_baseline.json` 1117, `_audit_metadata_baseline.json` 944,
`_uncalled_functions_baseline.json` 929, `_code_audit_baseline.json` 753, `_annotation_baseline.json` 702,
`_source_proxy_baseline.json` 475. Nine other baselines are at 0 entries (good — those are true
zero-tolerance gates).

The keys are `file:line`-ish strings, so line drift already forces some churn; but a genuine
re-introduction at a stale key is a real, if narrow, blind spot, and the large baselines have no downward
pressure at all.

**Do (cheap):** make `fixed` a **failure** with the message "N sites drained — run
`--refresh-<x>-baseline`". That is how a ratchet becomes monotone. (**INFERENCE**: this will cause churn
on line-number-keyed baselines; if keys are line-sensitive, apply the strict form to the zero/small
baselines first.)

### 1.13 `regen_baselines.py` covers 7 of 27 baselines — LOW

**MEASURED.** `_BASELINES` in `tests/test_meta/regen_baselines.py` lists 7 files; the directory holds 27
`_*baseline.json`. `BASELINES_README.md`'s "Files" section lists 7 as well. The other 20 are refreshed only
via their individual `--refresh-*` flags (registered in `conftest.py`, which lists 21 flags — itself a
third, differently-sized list). Three hand-maintained lists of the same thing, none checked against each
other.

**Do:** one meta-test asserting `{baseline files on disk} == {regen_baselines entries} == {refresh flags}
== {README entries}`. This is the same class the repo already gates for pre-commit/CI scope parity
(`test_precommit_ci_scope_parity.py`) and mypy beachhead coverage
(`test_precommit_mypy_beachhead_coverage.py`) — it just never got applied to the baselines themselves.

### 1.14 `test_precommit_ci_scope_parity.py` covers 4 of the paired hooks, not all — LOW

**MEASURED.** `PAIRED_HOOKS` = black-filtered / bandit / interrogate / codespell. The `ruff` src-vs-tests
pair (`.pre-commit-config.yaml:98` vs L432) is not in the list, despite being the same shape and the
highest-traffic gate.

**Do:** add it — needs disambiguation since both share the hook id `ruff`; key on `(repo, exclude/files)`
instead of on id.

### 1.15 `bandit -ll` discards all LOW-severity findings — LOW

**MEASURED (config), INFERENCE (impact).** `-ll` raises the reported severity floor to MEDIUM, so LOW
checks never fire — including `B110 try_except_pass`, `B112 try_except_continue`, `B404 subprocess
import`, `B603/B607 subprocess call`. `B101` is separately (and correctly, per the recorded 85/89
rationale) skipped in `[tool.bandit]`. The try/except/pass class is partly covered by the bare-except
meta-gates, so the residual is mostly subprocess handling in scripts and `_benchmarks`.

**Do:** low priority. If you want it, `-l` (LOW+) combined with `-i` (HIGH confidence only) is usually a
better signal/noise pair than `-ll`.

---

## 2. COVERAGE GAPS IN THE CHECKING ITSELF

### 2.1 Numeric warnings never fail anything — highest-value gap for this codebase

**MEASURED.** `[tool.pytest.ini_options].filterwarnings` errors on exactly one thing:
`error::DeprecationWarning:mlframe`. Everything else is an `ignore`. **`RuntimeWarning` appears nowhere**
in the filter list; searching `RuntimeWarning` in `tests/conftest.py` and `pyproject.toml` finds only an
unrelated `warnings.warn(..., RuntimeWarning)` at `tests/conftest.py:664`.

For a numerics library this is the biggest single blind spot in the whole setup. The concrete defect
class: `RuntimeWarning: invalid value encountered in divide` / `overflow encountered in exp` /
`Mean of empty slice` / `Degrees of freedom <= 0` emitted from an FE kernel or an MI estimator on a
degenerate column. The function returns NaN or +/-inf, a downstream `np.nanmax` or comparison silently
absorbs it, and the only symptom is a slightly worse feature ranking. Nothing in ~35k tests, 99 code_audit
scanners, or ~40 meta-gates observes it — an AST gate structurally cannot, because this is a runtime
property.

The codebase is already positioned for this: **MEASURED** 144 `np.errstate(...)` uses across src/mlframe,
i.e. the sites that legitimately expect these warnings are already annotated.

**Do:** add `"error::RuntimeWarning:mlframe"` to `filterwarnings`, or (lower blast radius) an autouse
fixture enabling `np.errstate(all="raise")` on one dedicated CI leg. Start it as a non-blocking nightly
leg to size the failure count before making it blocking — **INFERENCE**: I could not run the suite, so I
have no estimate of how many tests this trips.

### 2.2 The tool-version-to-gate-result coupling is unchecked

Generalisation of 1.1-1.3: no gate anywhere asserts "the tool that produced this green result is the
version we intend". Every version guarantee in this repo is a *text-to-text* comparison between config
files. Defect class: a gate that appears to run and passes, on a different tool than the one reviewed.

### 2.3 Wheel package-data is only half-verified

**MEASURED.** `ci.yml:637` asserts `mlframe/py.typed` is present in the built wheel. But
`[tool.setuptools.package-data]` declares two more entries whose own comments say they are consulted at
runtime: `"mlframe.feature_selection.filters" = ["default_kernel_tuning.json"]` (comment: "MUST ship in
the wheel") and `"mlframe.feature_selection.filters._vendored.infonet" = ["configs/*.yaml"]`. Neither is
asserted, and the smoke import (`mlframe.metrics.core`, `training`, `models.ensembling`,
`calibration.quality`) never touches the kernel-tuning cache or infonet inference.
`include-package-data = false` makes an accidental drop of either entry perfectly silent.

**Do:** one-line extension of the existing `zipfile` assertion in `ci.yml` to cover all three declared
data paths.

### 2.4 Public typing surface is never checked from a consumer's position

`py.typed` ships, and mypy runs over `src/mlframe` from inside the repo — but nothing runs mypy against
`import mlframe` from a *clean env with the wheel installed*. Defect class: a public signature annotated
with a name that is not exported (or that only resolves under `TYPE_CHECKING`), which type-checks in-repo
and breaks every downstream user's mypy run. Cost: ~5 lines appended to the existing `build` job.

### 2.5 Docstring *examples* are unverified and interrogate's denominator is narrow

See 1.9 for the doctest half. On interrogate: `ignore-init-method`, `ignore-init-module`, `ignore-magic`,
`ignore-private` are all true, so "100% docstrings" means 100% of public non-dunder non-`__init__`
objects. That is a defensible scope (and `test_complex_private_functions_documented.py` covers part of the
rest) — but 100% presence says nothing about correctness, and pydoclint, the tool that *would*, is
advisory-only with ~200 residual findings recorded in its own config comment.

---

## 3. NEW TOOLS / RULES WORTH ADDING

Ordered by value-per-cost. Deliberately excludes anything duplicating
ruff/bandit/vulture/deptry/interrogate/codespell/mypy/import-linter/pydoclint/semgrep/zizmor/actionlint/
shellcheck/detect-secrets.

### 3.1 Enable four ruff rules already available, currently off — do this first

All counts **MEASURED** via `ruff check src/mlframe --extend-select ... --statistics` with the project
config active:

| Rule | Count | What it catches that nothing here catches | FP risk |
|---|---|---|---|
| `PLW0127` self-assigning-variable | **2** | dead statement where a real operation was intended | ~0 |
| `PGH003` blanket-type-ignore | **38** | `# type: ignore` with no code — silences *every* future error on that line, defeating the `warn_unused_ignores` ratchet | ~0 |
| `PGH004` blanket-noqa | **8** | same, for ruff | ~0 |
| `PLE2515` zero-width-space in source | **1** | invisible character in a string literal | 0 |

`PLW0127` found a probable live bug — **MEASURED** at `src/mlframe/models/ensembling/predict.py:489`:

```python
    # Restore (N,) shape if original was 1-D
    if first.ndim == 1 and ensembled_predictions.shape[1] == 1:
        ensembled_predictions = ensembled_predictions[:, 0]
        if uncertainty is not None and uncertainty.shape:
            uncertainty = uncertainty          # <-- no-op
```

**INFERENCE** (not verified against a test): the guarded branch was clearly meant to squeeze `uncertainty`
the same way as the line above (`uncertainty = uncertainty[:, 0]`), so `predict()` returns predictions
shaped `(N,)` alongside uncertainty shaped `(N, 1)`. Worth checking by hand regardless of whether the rule
is adopted. The second hit, `src/mlframe/reporting/charts/shap_panels.py:427` (`root = root`), is dead but
harmless.

Note `PLE0605` (10 hits) is a **false positive** here — every hit is `__all__ = sorted([...])`, valid at
runtime, statically unresolvable. Do not enable it.

Cost: `extend-select = ["PLW0127", "PGH", "PLE2515"]` plus ~46 mechanical edits. Everything else I
measured is too noisy for this codebase and should stay off, consistent with the recorded SIM/RET/TRY
decision: `PLR0913` 1912, `TID252` 1529, `FBT001/002` 2090, `ARG001/002` 1346, `SLF001` 451, `ERA001` 237,
`C408` 683, `PLW0603` 275, `PIE790` 212, `PLW2901` 53, `FURB*` ~130.

### 3.2 `mypy: strict_equality = true` (project-wide)

**Catches, that nothing here does:** comparisons between non-overlapping types —
`if dtype == "float64"` against an `np.dtype`, `if mode == SomeEnum.X` where `mode: str`, bytes-vs-str.
Always-False conditions are a silent-correctness class no AST gate here covers.
**FP risk:** low; the usual noise is comparing a `Literal` to a wider `str`, which is generally a real
finding. **Cost:** one line in `[tool.mypy]`, plus whatever the first run surfaces (I did not run
whole-project mypy — it is the slow blocking gate and long runs were out of scope).

Consider `warn_unreachable = true` too, but **evaluate before adopting**: this codebase's
`try/except ImportError` optional-dep gates and version-guarded branches are exactly what that rule
over-reports on.

### 3.3 `pip-audit` as a **blocking** job — it already exists, only advisory

`pip-audit>=2.7` is in the dev extra and runs in the advisory bundle. Given how much of
`[project.dependencies]` consists of hand-written CVE floor pins with PYSEC/GHSA identifiers in the
comments (pillow, aiohttp, cryptography, starlette, tornado, setuptools), those floors are maintained by
hand and go stale silently. Making pip-audit blocking on the *declared* set (`pip-audit -r pyproject.toml`,
non-transitive) turns "someone remembered to bump the floor" into a gate.
**FP risk:** medium — a new advisory with no patched release yet will hard-block; mitigate with an explicit
`--ignore-vuln` list, which doubles as the machine-readable record of "known-open on py3.9" that the
comments already maintain in prose. **Cost:** one CI job, ~30 s. Not a duplicate of
Dependabot/dependency-review (those gate *changes*; this gates the *current* state).

### 3.4 `pytest --doctest-modules` on a scoped subpackage

**Catches:** docstring examples that no longer match behaviour — the "documentation drifted from code"
class the meta-suite polices *structurally* (`test_config_docstring_drift`,
`test_no_stale_not_wired_docstrings`) but never *executionally*.
**FP risk:** high if pointed at the whole tree (import side effects, GPU, randomness). Scope it to the
same beachhead already chosen for strict mypy — `mlframe.calibration`, `mlframe.metrics` — where
`doctest_optionflags` is already configured for it. **Cost:** one addopts variant in a small CI leg.

### 3.5 `hypothesis` is already a dev dep — spend it on the numeric invariants

**MEASURED:** `hypothesis>=6.80` is in the `dev` extra, and `tests/test_meta/test_utility_fuzz.py` exists,
so it is used somewhere. The numeric-kernel surface (MI estimators, discretization, rank correlation) is
the natural target: property tests asserting monotonicity, permutation invariance, and NaN/constant-column
handling catch the same defect class as 2.1 from the other side. This is not a new tool — it is unspent
capacity. **INFERENCE**: I did not survey how much of the suite already uses it.

### 3.6 Explicitly NOT recommended

`pylint` (ruff covers the useful subset, measured above and mostly noise here), `flake8` + plugins
(superseded), `pyright` (a second whole-project type checker with a different error set, on a codebase
already at mypy-zero, is a large migration for marginal gain), `xenon`/`radon` (C901 already measures it),
`safety` (pip-audit), `refurb` (measured `FURB*` above: ~130 hits, all cosmetic).

---

## 4. CEREMONY THAT COULD BE REMOVED

Ranked by how little would be lost.

1. **`.semgrep.yml` — 2 rules, advisory-only.** **MEASURED:** the file defines exactly two rules
   (`broad-except-silent-swallow`, `module-global-write-via-reexport-alias`), both `severity: WARNING`,
   run only in `lint-advisory` and a `stages: [manual]` hook. The first duplicates the bare-except /
   `verbose_gated_except` meta-gates, which are *blocking*. Carrying `semgrep>=1.70` (a heavy dep) plus a
   CI step for two advisory rules, one of which a blocking gate already covers, is the clearest
   net-negative item here. **Do:** port `module-global-write-via-reexport-alias` to a meta-test (that
   suite is where this repo's real rules live) and drop semgrep entirely.

2. **`[tool.importlinter]` — 2 contracts, advisory, and the config says so itself.** Its own header states
   "Contracts here are therefore documentation of intent -- a rule that must actually hold needs its own
   meta-test", and the second contract's trailer says "DOCUMENTATION ONLY ... The enforcing gate is the
   blocking meta-test `test_no_inbound_edge_to_benchmarking.py`". So one of the two contracts is
   explicitly redundant with a blocking test, and the other (kernel-sibling back-imports) could be the
   same ~15-line AST meta-test. **Do:** convert contract #1 to a meta-test, delete the section and the
   `import-linter` dev dep. This is honest de-duplication, not weakening — the advisory version enforces
   nothing today.

3. **The empty baselines' machinery.** Nine baseline files hold 0 entries (`_docstring_`, `_logger_lazy_`,
   `_mutable_defaults_`, `_tick_isinstance_`, `_readonly_to_numpy_mutation_`, `_fe_noop_copy_`,
   `_module_level_logging_disable_`, `_numba_config_env_mutation_`, `_unprotected_treeexplainer_`). Each
   carries a JSON file, a `--refresh-*` flag in `conftest.py`, and the load/diff/stderr-warn dance — to
   compare against `set()`. **Do:** for these, `assert not current` with the same actionable message.
   Keeps the gate identical, drops ~9 files and ~9 flags. (Counter-argument, and it is a fair one: the
   uniform shape means the next regression can be baselined in one command. If you value that, keep them
   — but then 1.13's consistency meta-test becomes more important, not less.)

4. **`[tool.pydoclint]` — advisory with ~200 known-residual findings.** The config comment records that the
   residual set was hand-audited once and yielded 4 real fixes. That is good ROI *for one audit*, not for
   a permanent advisory CI job nobody gates on. **Do:** either baseline the ~200 and make it blocking
   (then it has teeth), or run it as a periodic manual sweep and drop the CI job. As-is it is a job that
   emits ~200 lines every run, forever, that no one is required to read.

5. **`rev:` pins on `language: system` hooks** (see 1.2) — pure noise once the `entry` is overridden.
   Replace with a runtime version assertion, or drop the upstream repo entry and make it a plain `local`
   hook so the pin does not imply a guarantee it cannot give.

---

## What I could NOT check

- **I did not run the test suite, whole-project mypy, or the full pre-commit set** (instructed not to; the
  suite takes hours). So: no measured error count for 3.2, no measured failure count for 2.1, no
  verification that the ~40 meta-gates currently pass, and no measured coverage number for 1.6.
- **`py-ci-shared`'s reusable workflows were read at the local checkout** (`$PY_CI_SHARED_DIR` =
  `C:\Users\Admin\Machine learning\py-ci-shared`), whose HEAD may differ from the `@v1` tag / pinned SHA
  that CI actually resolves. Finding 1.1 should be re-confirmed against the exact ref `ci.yml` calls.
- **I read ~12 of the ~40 meta-tests in detail** and covered the rest by name plus targeted greps for the
  scan/baseline/fail patterns. Findings 1.12 and the empty-scan note below rest on that sample, not an
  exhaustive read.
- **Empty-scan fail-closed (partial, INFERENCE):** `grep -l 'rglob|glob('` matches **56** meta-test files;
  only **5** contain any "I actually scanned something" sanity guard. `test_no_file_over_1k_loc.py`
  `pytest.skip`s if `src/mlframe` is missing — and a skip is green in CI. I did not verify the other ~50
  individually, so I state this as a *likely* class rather than a confirmed defect: a gate whose root-path
  derivation breaks (a directory rename, a `parents[N]` off-by-one, running from an installed wheel) scans
  zero files and passes green forever. **Cheap universal fix:** one shared helper asserting the scanned
  file count is above a floor (e.g. `> 1000` for src-wide gates), used by every AST gate.
- **Whether GitHub org-level secret push protection is enabled** (relevant to 1.5).
- **Branch protection / required-check configuration.** `ci-required` aggregates
  `[test, test-heavy-serial, ruff-blocking, lint-blocking, tests-ruff-blocking, tests-black-filtered,
  tests-lint-blocking, build]` — note `black-filtered` (src) and `mypy-full` live in *separate workflows*,
  so they must be required checks in their own right. I cannot see branch-protection settings to confirm
  they are. If they are not, both are effectively advisory despite the config comments calling them
  blocking. **Worth verifying first — it would be the highest-impact item in this document.**
