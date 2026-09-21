"""Wire the cross-project checks py-ci-shared ships that mlframe never ran.

Every rule below is implemented and unit-tested in ``py_ci_shared``; this module is purely the consumption
point plus this repo's own allowlists. mlframe consumed three of roughly forty-six shared checks, so
maintained and directly-applicable rules sat unused while the findings they cover kept landing here -- the
same gap pyutilz closed with its own `test_shared_checks_wired.py`, which this mirrors.

Keeping them in ONE file makes "which shared checks does this repo actually run?" answerable by reading a
single import block, rather than by grepping for `py_ci_shared` across a hundred-odd meta modules.

Checks already covered by a first-party mlframe meta test are deliberately NOT re-wired here: the LOC budget
(`test_no_file_over_1k_loc.py`), import cycles (`test_no_import_cycles.py`), the code-audit ratchet
(`test_code_audit_baseline.py`, which also catches mutable default arguments), README env-var
parity (`test_readme_env_var_parity.py`) and per-job CI timeouts (`test_x_cicd_dependencies_fixes.py`).
Duplicating those would give two baselines for one rule.

Runtime: file reads plus a TOML parse. No network, no imports of mlframe itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

py_ci_shared = pytest.importorskip("py_ci_shared", reason="py-ci-shared is a dev-only git dependency")

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# The prose files this repo maintains at the root. `docs/` is out of scope: mkdocs resolves relative links
# against the BUILT site, not the repo root, so a valid mkdocs link reads as dead to a filesystem resolver.
PROSE_FILES = ("README.md", "CONTRIBUTING.md", "CHANGELOG.md", "TESTING.md")

# Deliberately empty: a `continue-on-error: true` inside a BLOCKING workflow turns a gate into a green no-op,
# and this repo keeps its advisory lint bundle in separate warn-only hooks rather than inside blocking jobs.
# An empty allowlist means any such step has to be justified here rather than appearing silently.
_REVIEWED_ADVISORY_STEPS: set[str] = {
    # Depends on stub availability in a bare environment holding only the wheel and py.typed, which is not
    # this repo's to guarantee: a missing third-party stub is a fact about that package, not a defect here.
    # It exists to surface a public signature annotated with a name that is not exported -- which
    # type-checks in-repo and breaks every consumer -- so its output is worth reading and its verdict is
    # not worth gating on.
    "Type-check the public surface from a consumer position",
    # Sizing a warning class, not enforcing one. The RuntimeWarning filter was measured clean on
    # tests/metrics + tests/calibration + tests/evaluation (1592 tests); this job produces the same number
    # for the feature-engineering and feature-selection trees, where the kernels that emit those warnings
    # live. It becomes blocking by moving the filter into pyproject's filterwarnings once it reads zero,
    # at which point this job and this entry both go away.
    "RuntimeWarning census (advisory)",
    # Several dependency floors are known-optimistic and nobody has measured how many. Blocking on the
    # first run would red the weekly schedule on pre-existing debt rather than on a regression. Flip it
    # once the count is zero.
    "install + smoke at declared floors",
    # Three of these hooks are auto-fixers whose failure mode on a fresh checkout is "I rewrote your
    # files", and nobody has measured how many files that touches. The job exists to close the gap that
    # detect-secrets and shellcheck were enforced only on a machine with hooks installed -- bypassed by
    # --no-verify and by any fork PR. Flip it off once a run reads clean.
    "hooks with no CI counterpart",
    # The declared CVE floors are behind by several advisories and the fixes are sitting in open dependabot
    # PRs, so this reports known debt, not a regression. Flip it off once that backlog is merged.
    "known CVEs in the locked graph (advisory)",
}

# pyutilz and py-ci-shared are first-party upstreams owned by this repo's own maintainer, so the supply-chain
# threat a commit-SHA pin defends against does not apply -- whoever could move those refs could push here
# directly. They are pinned anyway (see the twelve pin sites across the workflows), for reproducibility
# rather than for security; this prefix list only exempts them from the THIRD-PARTY pinning rule.
_FIRST_PARTY_GIT_PREFIXES = (
    "git+https://github.com/fingoldo/py-ci-shared",
    "git+https://github.com/fingoldo/pyutilz",
)


def _workflow_names() -> list[str]:
    """Every workflow file, so a newly added one is covered without editing this module."""
    return sorted(p.name for p in WORKFLOWS_DIR.glob("*.yml"))


@pytest.mark.parametrize("workflow_name", _workflow_names())
def test_no_undeclared_continue_on_error(workflow_name: str):
    """`continue-on-error: true` turns a blocking gate into a green no-op.

    Parametrised per workflow so the failure names the file rather than handing back one combined blob.
    """
    from py_ci_shared.ci_workflow_gate import assert_continue_on_error_is_reviewed

    assert_continue_on_error_is_reviewed(WORKFLOWS_DIR / workflow_name, reviewed_advisory_steps=_REVIEWED_ADVISORY_STEPS)


def test_declared_entry_points_resolve():
    """Every console script / entry point imports and exposes the attribute it names.

    A broken entry point is invisible until someone installs the package and runs the command, which no test
    in this repo otherwise does.
    """
    from py_ci_shared.entry_points_resolvable import assert_all_entry_points_resolvable

    assert_all_entry_points_resolvable(PYPROJECT)


def test_no_phantom_markdown_links():
    """Every markdown link in the maintained root prose resolves to a real file.

    A dead link in README/CONTRIBUTING is the first thing a new reader hits and the last thing anyone checks.
    """
    from py_ci_shared.phantom_markdown_links import assert_no_phantom_markdown_links

    existing = [REPO_ROOT / name for name in PROSE_FILES if (REPO_ROOT / name).exists()]
    assert existing, "none of the expected prose files exist, so this check would pass by looking at nothing"
    assert_no_phantom_markdown_links(md_files=existing, repo_root=REPO_ROOT)


def test_pyproject_declares_no_unpinned_git_dependency():
    """A `git+https` direct reference in [project] makes the sdist/wheel unpublishable on PyPI.

    pyutilz is documented in a comment there rather than declared as a git URL for exactly that reason; this
    fails on the commit that turns the comment into a real dependency.
    """
    from py_ci_shared.git_dependency_pins import assert_all_git_dependencies_pinned

    assert_all_git_dependencies_pinned(PYPROJECT, allow_unpinned_url_prefixes=_FIRST_PARTY_GIT_PREFIXES)


def test_dev_requirements_git_dependencies_are_pinned_or_first_party():
    """A THIRD-PARTY git dependency must carry a full commit SHA; only the maintainer's own upstreams float."""
    from py_ci_shared.git_dependency_pins import assert_all_git_dependencies_pinned

    req = REPO_ROOT / "requirements-dev.txt"
    if not req.exists():
        pytest.skip("no requirements-dev.txt in this repo")
    assert_all_git_dependencies_pinned(req, allow_unpinned_url_prefixes=_FIRST_PARTY_GIT_PREFIXES)


def test_every_from_import_resolves():
    """A `from X import Y` naming something X does not define is invisible until that code loads.

    At module scope it is an ImportError at COLLECTION, and pytest-split collects the whole tree in every
    shard -- one removed helper took 39 of 40 shards red here. Inside a function it waits for the branch: a
    `from ..linear_model import LinearRegression` in the `if self.regressor is None:` arm of
    `ESTransformedTargetRegressor.fit` broke the DOCUMENTED DEFAULT while every test passing an explicit
    regressor stayed green, and `mlframe.linear_model` never existed at all.

    Static resolution -- the target is parsed, never imported -- so this costs no side effects and covers
    modules whose imports are expensive or hardware-dependent.
    """
    from py_ci_shared.unresolved_imports import assert_all_from_imports_resolve

    assert_all_from_imports_resolve(
        # scripts/, profiling/ and benchmarks/ are scanned too. They sit outside `testpaths`, so nothing
        # collects them and CI stays green while they rot: a package reorganisation left 9 dead module paths
        # across 23 import sites there, and every one of those reproducibility benchmarks was silently
        # unrunnable -- anyone returning to re-measure a perf claim got a ModuleNotFoundError, not a number.
        scan_roots=[REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "scripts", REPO_ROOT / "profiling", REPO_ROOT / "benchmarks"],
        package_roots=[REPO_ROOT / "src"],
        resolvable_prefixes=("mlframe",),
        allowlist=(),
    )


def test_no_epsilon_padded_power_denominators():
    """`a / (b + 1e-12)` is safe while b's scale sits near 1, and unsafe the moment b is a power.

    A power falls off geometrically, so a fixed pad stops being negligible at ordinary inputs and starts
    deciding the result -- with nothing raising. Both instances this caught lived in spatial.py and passed
    their own suites, whose fixtures used coordinates of order 1 where the pad genuinely is negligible:
    `k / (r**d + 1e-12)` returned 9.999e12 for a true 1e17 at d=8, r=0.01, and `1.0 / (dist**power + 1e-12)`
    turned inverse-distance weights into [0.282, 0.282, 0.266, 0.170] where the true weights were
    [0.940, 0.059, 0.0015, 0.0001] -- an almost unweighted average.

    `_benchmarks` is excluded: a frozen bench copy is meant to keep the shape it was frozen with, which is
    the whole point of comparing against it.
    """
    from py_ci_shared.epsilon_padded_denominators import assert_no_epsilon_padded_power_denominators

    assert_no_epsilon_padded_power_denominators(
        [REPO_ROOT / "src"],
        exclude=("_benchmarks", "_cpx36_baseline"),
    )

def test_no_hash_is_fed_by_an_array_copy():
    """`h.update(a.tobytes())` allocates a second copy of the whole array purely to be hashed.

    `mlframe._array_buffer.array_buffer(a)` hands the hash the existing buffer and produces the identical
    digest. The sites this replaced hashed whole training frames -- a KeyBank fingerprint over X_train, a
    collinearity cache key over the feature matrix, an RFECV signature over X and y -- on data this package
    sizes in the tens of gigabytes, with the copy paid on every cache lookup.

    The rewrite originally landed as `np.ascontiguousarray(a).data`, spelled out at each site, and that form
    RAISES on datetime64 and timedelta64: they have no buffer-protocol format. `data_signature` crashed on any
    pandas frame carrying a datetime column until the sites were routed through the leaf helper instead. See
    tests/test_meta/test_array_buffer_is_the_one_way_to_feed_a_hash.py, which also gates the form from
    reappearing.

    Sites where the rewrite does not apply are not reported: `hash()` and dict keys need a hashable object
    and a memoryview is not one, and a `+`-joined payload has to be restructured rather than substituted.
    """
    from py_ci_shared.hash_fed_by_array_copy import assert_no_hash_fed_by_array_copy

    assert_no_hash_fed_by_array_copy([REPO_ROOT / "src"], exclude=("_benchmarks", "_cpx36_baseline"))

def test_repo_hygiene():
    """No tracked generated files, and no numeric CI gate that passes when its own input broke.

    This one paid for itself immediately: `codecov-full.yml` compared `$count` numerically without proving it
    non-empty, so a failed `gh api` query printed "Skipping run N (0 artifacts)" -- reporting a BROKEN QUERY as
    a confirmed absence and walking past a run that may have had the data.
    """
    from py_ci_shared.repo_hygiene import assert_repo_hygiene

    assert_repo_hygiene(REPO_ROOT, workflows_dir=WORKFLOWS_DIR)


def test_workflow_paths_exist():
    """Every path a workflow references resolves; a renamed script silently stops being run otherwise."""
    from py_ci_shared.ci_workflow_paths import assert_workflow_paths_exist

    assert_workflow_paths_exist(WORKFLOWS_DIR, REPO_ROOT)


def test_no_count_claim_mismatches():
    """A prose sentence claiming "N of them" must match the list it introduces."""
    from py_ci_shared.phantom_code_references import assert_no_count_claim_mismatches

    assert_no_count_claim_mismatches([REPO_ROOT / name for name in PROSE_FILES if (REPO_ROOT / name).exists()])


# Nothing here is intentionally excluded from CI: the main run is PATHLESS (`pytest -m "not slow and not
# gpu ..."` collects from rootdir), so every tests/<subdir> is reached. An empty set therefore means any
# NEW subdir that no job collects has to be justified here rather than silently gating nothing.
_INTENTIONALLY_UNREACHED_TEST_DIRS: set = set()


def test_every_test_subdir_is_reachable_from_ci():
    """A new tests/<subdir> that no CI job collects gates nothing.

    Wiring this needed an upstream fix first (py-ci-shared 407cc90): the pathless-invocation detector read
    pytest-split's `--splits 10 --group 1` values as positional test paths, so this repo's pathless run
    looked targeted and all 20 subdirs came back unreached. Whitelisting them would have recorded a
    falsehood -- they are collected -- so the check was fixed rather than silenced.
    """
    from py_ci_shared.ci_test_dir_reachability import assert_every_test_subdir_reachable

    assert_every_test_subdir_reachable(
        repo_root=REPO_ROOT,
        workflows_dir=WORKFLOWS_DIR,
        intentionally_unreached=_INTENTIONALLY_UNREACHED_TEST_DIRS,
    )


# Stale comments are RATCHETED rather than gated outright: `src/` carries 23 of them, and a hard gate would
# simply be red from the day the check is wired, which teaches everyone to ignore it. The baseline freezes
# what exists so no NEW stale TODO or commented-out call can appear, and the set can only shrink.
_STALE_COMMENT_BASELINE = Path(__file__).resolve().parent / "_stale_comment_baseline.json"


def _stale_comment_keys() -> dict:
    """Current stale comments as {stable key: description}.

    Keyed on `path::<comment text>` rather than `path:line`, because the age check reports a line number and
    ANY edit above a comment would otherwise present it as a brand-new finding. The comment text is what
    identifies it; where it sits in the file is not.
    """
    from py_ci_shared.stale_comment_age import find_stale_comments

    out: dict = {}
    for problem in find_stale_comments(REPO_ROOT, ["src"], max_age_days=30, require_issue_ref=True):
        location, _, description = problem.partition(": ")
        path = location.rsplit(":", 1)[0]
        snippet = description.split("`")[1] if "`" in description else description
        out[f"{path}::{snippet.strip()}"] = description.strip()
    return out


def test_no_new_stale_todos():
    """A TODO that outlives its own deadline is a decision nobody made, not a plan.

    Refresh via `python tests/test_meta/regen_baselines.py` -- and only after confirming the new entries are
    genuinely accepted, never to get past a failure.
    """
    import orjson

    found = _stale_comment_keys()
    accepted = orjson.loads(_STALE_COMMENT_BASELINE.read_bytes()) if _STALE_COMMENT_BASELINE.exists() else {}

    new = {k: v for k, v in found.items() if k not in accepted}
    assert not new, "new stale comment(s) -- do it, delete it, or reference an issue:\n  " + "\n  ".join(f"{k}: {v}" for k, v in sorted(new.items()))

    drained = [k for k in accepted if k not in found]
    if drained:
        print(f"\n[stale-comments] {len(drained)} baseline entr(y/ies) DRAINED; refresh to lock the smaller set in.")


def regenerate_baseline() -> None:
    """Rewrite the stale-comment baseline from the current tree. Called by `regen_baselines.py`."""
    import orjson

    payload = orjson.dumps(dict(sorted(_stale_comment_keys().items())), option=orjson.OPT_INDENT_2).decode("utf-8")
    _STALE_COMMENT_BASELINE.write_bytes((payload + chr(10)).encode("utf-8"))  # bytes: text mode on Windows writes CRLF


def test_no_inert_patch_targets():
    """A test that patches an attribute the target module does not have is patching nothing.

    The assignment CREATES the attribute instead of replacing anything the code reads, so the
    production path runs unpatched while the test's own assertions read back whatever the test just
    wrote -- and the save/restore leaves the invented attribute on the module for the rest of the
    process, which is the module-pollution class CLAUDE.md already calls out.

    Four were found when this was first run: two patched
    `mlframe.training.core.get_pandas_view_of_polars_df` (every real call site imports it lazily from
    `training.utils` inside a function body, so only the utils patch ever did anything), one reset a
    `_fallback_logged` latch that had been deliberately replaced by a time-based rate limit, and one
    reset the RawKernel singleton on the discretization facade rather than on the module that owns
    the global -- so `k1 is k2` could pass on a kernel an earlier test had already built.
    """
    from py_ci_shared import inert_patch_targets

    index = inert_patch_targets.module_index([REPO_ROOT / "src"], package_root=REPO_ROOT / "src")
    findings = inert_patch_targets.scan(sorted((REPO_ROOT / "tests").rglob("test_*.py")), index)

    assert findings == [], "patched attributes that do not exist on their target module:\n  " + "\n  ".join(
        f"{f.path.relative_to(REPO_ROOT).as_posix()}:{f.lineno} {f.target}" for f in findings
    )


def test_every_database_effect_is_asserted_by_an_importing_test():
    """A module that commits or executes, whose importing tests never look at that call.

    Zero here, so this is a gate rather than a ratchet and `accepted` is empty on purpose. It is
    wired now because the count being zero is the cheap moment: the same check found fifty-five in a
    sibling repo, and closing those turned up a resume cache that could serve an empty response as a
    model's answer and a readiness probe that answered 200 without reaching the database.

    The population assertion is not decoration. On a src layout the scan resolved no modules at all
    until recently, so the check passed having measured nothing -- green for the one reason a gate
    must never be green. A count is the cheapest way to notice that.
    """
    from py_ci_shared.effect_assertion_parity import assert_effects_are_asserted, build_import_map

    import_map = build_import_map(REPO_ROOT)
    assert len(import_map) > 1000, f"only {len(import_map)} modules resolved -- the scan lost its subject and this gate would pass vacuously"

    assert_effects_are_asserted(REPO_ROOT, import_map, ())


def _src_files() -> list[Path]:
    """Production modules, minus frozen bench copies (kept in the shape they were measured with)."""
    return sorted(p for p in (REPO_ROOT / "src").rglob("*.py") if "_benchmarks" not in p.parts and "_cpx36_baseline" not in p.parts)


# `path::function` -> why the reported timer does not time GPU work. Both are loops the checker reads too widely.
_GPU_TIMING_NOT_GPU_WORK: dict[str, str] = {
    "src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_score.py::_score_one_pair": "times numpy transforms and the CPU numba discretizer; the stop sits in a nested else, so the checker treats the rest of the loop as timed",
    "src/mlframe/feature_selection/filters/_screen_predictors.py::screen_predictors": "start_time is a runtime-budget origin passed down to callees; nothing is timed here, the only GPU call is cp.random.seed()",
}


def test_gpu_timings_synchronize_the_device():
    """A timer stopped right after a CUDA launch measures the launch, not the work.

    The first run found 46 such timings: benchmark scripts, and the kernel-tuning-cache dispatch timings whose
    numbers are persisted and pick backends on every later run. `mlframe.utils.gpu_sync.synchronize_gpu_if_available`
    is the fix at each of them.
    """
    from py_ci_shared.gpu_timing_sync import assert_no_unsynchronized_gpu_timings

    files = sorted(p for d in ("src", "tests", "benchmarks", "profiling") for p in (REPO_ROOT / d).rglob("*.py"))
    assert len(files) > 1000, f"only {len(files)} files scanned -- the walk lost its subject"
    assert_no_unsynchronized_gpu_timings(files, root=REPO_ROOT, allowlist=frozenset(_GPU_TIMING_NOT_GPU_WORK))


def test_no_identity_comparison_of_string_constants():
    """`x is SOME_STRING` holds only while CPython happens to intern both sides.

    The sentinel in polynom_pair_fe is a string on purpose -- it crosses the loky process boundary, where an
    `object()` sentinel loses its identity -- so it must be compared by value.
    """
    from py_ci_shared.identity_comparisons import assert_no_identity_comparisons

    assert_no_identity_comparisons(_src_files(), root=REPO_ROOT, min_files=500)


def test_no_naive_utcnow():
    """`datetime.utcnow()` is deprecated for removal and returns a naive value."""
    from py_ci_shared.naive_utcnow import assert_no_naive_utcnow

    assert_no_naive_utcnow(REPO_ROOT / "src")


def test_optional_numbers_are_tested_for_none():
    """`if seed:` on an `int | None` reads 0 as absent.

    Two of the 32 first reported were real: `best_desired_score=0.0` never stopped the search and
    `min_relevance_gain=0.0` switched verbose candidate logging off. The rest now say `is not None`, or state
    explicitly that 0 means disabled, so the accepted list is empty.
    """
    from py_ci_shared.optional_truthiness import assert_optionals_test_for_none

    assert_optionals_test_for_none(files=_src_files(), repo_root=REPO_ROOT, baseline=(), min_subjects=100)


_VACUOUS_LOOP_BASELINE = Path(__file__).resolve().parent / "_vacuous_loop_baseline.json"


def _test_files() -> list[Path]:
    """Return every test_*.py file under tests/, sorted."""
    return sorted((REPO_ROOT / "tests").rglob("test_*.py"))


def test_no_new_floorless_assert_loop():
    """A test whose only assertions sit inside a loop passes when the loop runs zero times.

    The baseline is EMPTY: all 451 such loops have been given a floor, and five of them turned out to iterate zero
    times, so the tests named after a behaviour were checking nothing. A new one fails; record it only when the
    loop's own emptiness IS the contract, and say so.
    """
    from py_ci_shared.vacuous_loop_assertions import assert_no_new_floorless_loop

    assert_no_new_floorless_loop(files=_test_files(), repo_root=REPO_ROOT, baseline_path=_VACUOUS_LOOP_BASELINE)


def regenerate_vacuous_loop_baseline() -> None:
    """Rewrite the floorless-loop baseline from the current tree. Called by `regen_baselines.py`."""
    import orjson

    from py_ci_shared.vacuous_loop_assertions import find_floorless_loops

    found = {loop.key: "pre-existing, recorded when the check was wired; not yet individually triaged" for loop in find_floorless_loops(_test_files(), REPO_ROOT)}
    payload = orjson.dumps(dict(sorted(found.items())), option=orjson.OPT_INDENT_2).decode("utf-8")
    _VACUOUS_LOOP_BASELINE.write_bytes((payload + chr(10)).encode("utf-8"))  # bytes: text mode on Windows writes CRLF


_FAIL_OPEN_BASELINE = Path(__file__).resolve().parent / "_fail_open_handlers_baseline.json"
# Gate and transform packages: where a handler that keeps a candidate on error ships the candidate the gate existed to stop.
_FAIL_OPEN_SCOPE = ("src/mlframe/training/composite", "src/mlframe/feature_selection")


def _fail_open_files() -> list[Path]:
    """Every module under the gate/transform packages, benchmark folders excluded."""
    return sorted(p for d in _FAIL_OPEN_SCOPE for p in (REPO_ROOT / d).rglob("*.py") if "_benchmarks" not in p.parts)


def test_no_new_fail_open_handlers():
    """A failure inside a gate must not disable the gate: no new admit-on-error, error-returns-True, quiet fallback or NaN-skipped reject.

    The composite gates kept a spec whenever evaluating it raised, and the tiny rerank's threshold skipped a NaN score, so the
    specs that failed were the ones that shipped. The composite backlog is fixed or carries a reason in the baseline; the
    feature-selection entries were recorded when the check was wired and are not yet triaged.
    """
    from py_ci_shared.fail_open_handlers import assert_no_new_fail_open_handlers

    files = _fail_open_files()
    assert len(files) > 300, f"scanned only {len(files)} modules; the scope paths no longer match the tree"
    assert_no_new_fail_open_handlers(files=files, repo_root=REPO_ROOT, baseline_path=_FAIL_OPEN_BASELINE)


def regenerate_fail_open_baseline() -> None:
    """Rewrite the fail-open baseline from the current tree, keeping every existing note. Called by `regen_baselines.py`."""
    import orjson

    from py_ci_shared.fail_open_handlers import find_fail_open_handlers

    old: dict = orjson.loads(_FAIL_OPEN_BASELINE.read_bytes()) if _FAIL_OPEN_BASELINE.exists() else {}
    counts: dict[str, int] = {}
    found: dict[str, str] = {}
    for h in find_fail_open_handlers(_fail_open_files(), REPO_ROOT):
        counts[h.scope] = counts.get(h.scope, 0) + 1
        key = h.scope if counts[h.scope] == 1 else f"{h.scope}#{counts[h.scope]}"
        found[key] = old.get(key, "pre-existing, recorded when the check was wired; not yet triaged")
    payload = orjson.dumps(dict(sorted(found.items())), option=orjson.OPT_INDENT_2).decode("utf-8")
    _FAIL_OPEN_BASELINE.write_bytes((payload + chr(10)).encode("utf-8"))  # bytes: text mode on Windows writes CRLF


# Synthetic timestamps for generated benchmark data: naive on purpose, like the user frames they stand in for.
_TIMEZONE_ALLOWED: dict = {
    ("profiling/profile_metrics_blocks.py", "DTZ001"): "synthetic naive timestamps for a generated benchmark frame",
    ("profiling/profile_training.py", "DTZ001"): "synthetic naive timestamps for a generated benchmark frame",
}


def test_timezone_honest(monkeypatch, tmp_path):
    """Non-test code states its time frame: no naive `datetime.now()` / `datetime(...)`, elapsed time is monotonic.

    Runs ruff's DTZ rules over every directory that holds Python, including ones `[tool.ruff] exclude` drops.
    pyproject's ruff config `extend`s `$PY_CI_SHARED_DIR/configs/ruff-base.toml`; the DTZ rules are selected
    explicitly, so when no checkout is configured an empty base stands in for it.
    """
    import os

    from py_ci_shared.timezone_honest import assert_timezone_honest

    if not os.environ.get("PY_CI_SHARED_DIR"):
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "ruff-base.toml").write_text("", encoding="utf-8")
        monkeypatch.setenv("PY_CI_SHARED_DIR", str(tmp_path))
    assert_timezone_honest(REPO_ROOT, scan_paths=("src", "profiling", "scripts", "benchmarks"), allowed=_TIMEZONE_ALLOWED)


_FUNCTION_LENGTH_BASELINE = Path(__file__).resolve().parent / "_function_length_baseline.json"


def test_long_functions_do_not_grow():
    """No new function over 150 lines, and the long ones already there may not get longer.

    Ratcheted per `path::qualname`, so moving a function does not reset it. A function that shrinks must have its
    ceiling lowered: refresh via `python tests/test_meta/regen_baselines.py`.
    """
    from py_ci_shared.function_length import assert_functions_do_not_grow

    assert_functions_do_not_grow(_src_files(), REPO_ROOT, _FUNCTION_LENGTH_BASELINE, limit=150, min_functions=5000)


def regenerate_function_length_baseline() -> None:
    """Rewrite the function-length ceilings from the current tree. Called by `regen_baselines.py`."""
    from py_ci_shared.function_length import function_lengths, write_length_baseline

    write_length_baseline(_FUNCTION_LENGTH_BASELINE, function_lengths(_src_files(), REPO_ROOT), limit=150)


# Documents that legitimately name things the code does not contain.
_DOC_PARITY_EXCLUDED = {
    "CHANGELOG.md",  # names symbols as they were when each entry was written
    "audits/",  # historical findings and dispositions
    "research/",  # design notes for things not built yet
    "docs/MRMR_RESEARCH.md",
    "docs/pysr_fe_upgrade_research.md",
    "docs/date_features_kaggle_research.md",
    "docs/BENCHMARK_PREREGISTRATION.md",  # binding pre-registration: frozen by design, never edited after the fact
    "src/mlframe/feature_selection/_benchmarks/fs_hybrid/AGENT_IDEAS_ROUND4.md",  # idea backlog
    "src/mlframe/feature_engineering/transformer/RESULTS.md",  # experiment log
    "tests/perf/results/",  # measurement logs
    "tests/feature_selection/MRMR_AUDIT_2026_06_22.md",  # audit record
}
# Names that exist only once formatted at runtime, or that belong to another tool.
_DOC_PARITY_IGNORED = {
    "rolling_mean_w30",  # f"rolling_mean_w{W} (ts)" for W in (7, 30)
    "test_log_loss_micro",  # f"{split_name}_log_loss_micro"
    "--python-backtrace",  # nsys CLI flags, documented as absent/present in nsys itself
    "--python-functions-trace",
    # CLAUDE.md: flags of external tools, a deliberately bad example name, and a rename it records
    "--write",  # py_ci_shared.black_filtered_apply
    "--metrics",  # nvprof
    "--events",  # nvprof
    "test_thing_works",  # the name CLAUDE.md tells you NOT to use
    "test_fused_bundle_returns_none_on_tied_scores",  # "was X, now Y" history
}


def test_docs_name_real_identifiers():
    """A backticked flag or snake_case identifier in a document must occur somewhere in the code.

    Its first run found a classification baseline table listing two time-series baselines that were never built,
    a scenario name that had since gained a word, and a guide citing an internal note as if it were code.
    """
    from py_ci_shared.doc_identifier_parity import assert_doc_identifiers_exist

    assert_doc_identifiers_exist(REPO_ROOT, exclude_docs=_DOC_PARITY_EXCLUDED, ignore=_DOC_PARITY_IGNORED)


_EXTRAS_BULLET = r'(?m)mlframe\[([\w-]+)\]"\s+#\s*(.+)$'
_ALL_EXTRAS_LINE = r'mlframe\[(all)\]"\s+#\s*all runtime extras: ([\w, ]+?) \('


def test_readme_install_block_matches_the_extras():
    """Every extras group has a README install line naming exactly its packages, and `[all]` names its groups.

    Its first run found `[transformer_ann]` advertised as hnswlib while the group installs pynndescent, `[signal]`
    advertised with antropy it does not contain, six groups missing packages, and nine groups with no line at all.
    Aggregates (`all`, `transformer_full`, `gpu-cuda12`) are checked by member group instead; `dev` names its
    headline tools only, the full list being pyproject's.
    """
    import re

    from py_ci_shared.docs_inventory_parity import assert_no_inventory_drift, find_aggregate_group_drift, find_extras_documentation_drift

    readme = REPO_ROOT / "README.md"
    assert re.search(_ALL_EXTRAS_LINE, readme.read_text(encoding="utf-8")), "the [all] line no longer states its member groups"
    problems = find_extras_documentation_drift(PYPROJECT, readme, _EXTRAS_BULLET, undocumented_groups=("all", "dev", "transformer_full", "gpu-cuda12"))
    problems += find_aggregate_group_drift(PYPROJECT, readme, _ALL_EXTRAS_LINE)
    assert_no_inventory_drift(problems, "README install block vs pyproject extras")


def _user_docs() -> list[Path]:
    """The maintained, user-facing prose: root docs plus docs/ guides and recipes, minus research and roadmap notes."""
    docs = [REPO_ROOT / name for name in PROSE_FILES if (REPO_ROOT / name).exists()]
    docs += sorted((REPO_ROOT / "docs").glob("*.md")) + sorted((REPO_ROOT / "docs" / "examples").glob("*.md"))
    return [p for p in docs if not any(k in p.name for k in ("RESEARCH", "research", "PREREGISTRATION", "ROADMAP", "BACKLOG"))]


def test_docs_name_real_paths():
    """A backticked repo path in the docs must exist; paths may be written relative to the subpackage they sit in."""
    from py_ci_shared.docs_inventory_parity import assert_no_inventory_drift, find_phantom_doc_paths

    src = REPO_ROOT / "src" / "mlframe"
    roots = [REPO_ROOT / "src", src, src / "training", src / "training" / "composite", src / "feature_selection", src / "feature_selection" / "filters"]
    problems = find_phantom_doc_paths(
        _user_docs(), REPO_ROOT, search_roots=roots,
        ignore=("infer/my_featureset/lgb.dump", "infer/my_featureset/lgb.dump.sha256"),  # an example layout, not a repo path
    )
    assert_no_inventory_drift(problems, "backticked paths in the docs")


def test_docs_use_only_declared_markers():
    """A `@pytest.mark.<name>` shown in the docs is a collection error under --strict-markers if undeclared."""
    from py_ci_shared.docs_inventory_parity import assert_no_inventory_drift, find_undeclared_markers

    assert_no_inventory_drift(find_undeclared_markers([*_user_docs(), REPO_ROOT / "CLAUDE.md"], PYPROJECT), "pytest markers named in the docs")


def test_package_doctests_pass():
    """The examples in docstrings run and print what they say.

    Nothing ran them before: the first run failed 26 of 80, every one a documentation error (undefined names,
    stale API, missing expected output, a `>>>` where `...` belonged). Modules under `_benchmarks` are skipped:
    they are scripts that execute on import. The floor keeps deleting examples from turning this green.
    """
    import doctest

    from py_ci_shared.package_doctests import assert_package_doctests_pass

    assert_package_doctests_pass("mlframe", skip_parts=("_benchmarks",), min_examples=100, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)


# Rounds whose findings carry a `**Disposition:** VERDICT. text` line. The older rounds record status only in their
# tracker table, which this check does not read; listing the rounds keeps a format change from passing vacuously.
_DISPOSITION_ROUNDS = ("audits/ci_review_2026-09-08", "audits/full_audit_2026-09-01")


@pytest.mark.parametrize("round_dir", _DISPOSITION_ROUNDS)
def test_resolved_dispositions_name_real_files(round_dir):
    """A finding marked RESOLVED/PARTIAL that names a file must name one that exists (repo- or package-relative)."""
    from py_ci_shared.audit_disposition_parity import assert_dispositions_name_real_artefacts

    src = REPO_ROOT / "src" / "mlframe"
    roots = [REPO_ROOT / "src", src, *sorted(p for p in src.iterdir() if p.is_dir())]
    assert_dispositions_name_real_artefacts(REPO_ROOT / round_dir, REPO_ROOT, search_roots=roots)


# Names the reference detector reads as tests but that are not test functions.
_DISPOSITION_TEST_NAMES_KNOWN = (
    "xcut_nondiscriminating_asserts.md: `test_preds`: no test of that name in tests/",  # a dict key, not a test
    "xcut_test_quality.md: `test_no_single_shot_timing_assertion`: no test of that name in tests/",  # a meta-test module name
)


def test_tests_named_by_audits_exist():
    """A disposition saying "covered by test_x" must name a test that exists."""
    from py_ci_shared.disposition_test_references import assert_disposition_tests_exist

    files = sorted((REPO_ROOT / "audits").rglob("*.md"))
    assert_disposition_tests_exist(files, REPO_ROOT, known=_DISPOSITION_TEST_NAMES_KNOWN, min_files=100)


_NON_PRODUCT = "tests/scripts/benchmarks/profiling are not the shipped package; tests have their own blocking twin hook"
# Every place a blocking gate narrows its scope or lowers its bar, with why. A new narrowing fails until it is
# written down here; a removed one fails until its entry goes.
_DECLARED_NARROWINGS: dict[str, str] = {
    r"pre-commit::check-added-large-files::exclude=^uv\.lock$": "the lockfile is large by nature and generated",
    r"pre-commit::mixed-line-ending::exclude=\.(bat|cmd|ps1)$": "Windows scripts keep CRLF",
    r"pre-commit::check-json::exclude=\.vscode/": "VS Code settings are JSON with comments",
    r"pre-commit::end-of-file-fixer::exclude=\.secrets\.baseline$": "generated by detect-secrets, rewritten on every scan",
    r"pre-commit::trailing-whitespace::exclude=\.secrets\.baseline$": "generated by detect-secrets, rewritten on every scan",
    r"pre-commit::detect-secrets::exclude=\.secrets\.baseline$": "the baseline lists the accepted hashes itself",
    r"pre-commit::ruff::--ignore=C901": "complexity is ratcheted by test_c901_debt_ratchet and the function-length ratchet instead",
    r"pre-commit::ruff::exclude=(^|/)(tests|scripts|legacy|benchmarks|_benchmarks|profiling)/": _NON_PRODUCT,
    r"pre-commit::codespell-blocking::exclude=(^|/)(tests|scripts|legacy|benchmarks|_benchmarks|profiling)/": _NON_PRODUCT,
    r"pre-commit::black-filtered-blocking::exclude=(^|/)(tests|scripts|legacy|benchmarks|_benchmarks|profiling)/": _NON_PRODUCT,
    r"pre-commit::bandit-blocking::exclude=(^|/)(tests|scripts|legacy|benchmarks|_benchmarks|profiling)/": _NON_PRODUCT,
    r"pre-commit::interrogate-blocking::--fail-under=100": "100 is the strictest bar, not a lowered one",
    r"pre-commit::interrogate-blocking::exclude=(^|/)(tests|scripts|legacy|benchmarks|_benchmarks|profiling)/": _NON_PRODUCT,
    r"pre-commit::ruff::files=^tests/": "the tests twin of the src ruff hook, with the tests ruff config",
    r"pre-commit::black-filtered-tests-blocking::files=^tests/": "the tests twin of the src hook",
    r"pre-commit::bandit-tests-blocking::files=^tests/": "the tests twin of the src hook",
    r"pre-commit::interrogate-tests-blocking::files=^tests/": "the tests twin of the src hook",
    r"pre-commit::codespell-tests-blocking::files=^tests/": "the tests twin of the src hook",
    r"pre-commit::yamllint-blocking::files=^(\.github/workflows/.*\.ya?ml|\.pre-commit-config\.yaml)$": "yamllint targets the CI and hook configs",
    r"pre-commit::zizmor-blocking::files=^\.github/workflows/.*\.ya?ml$": "zizmor audits GitHub workflows only",
    r"pre-commit::mypy-full-manual::exclude=(^|/)(legacy|_?benchmarks|profiling)/": "frozen bench/profiling scripts, not the package",
    r"ci.yml::run::--ignore=tests/training/test_core.py": "run by the dedicated serial test-heavy-serial job on every Python version",
    r"ci.yml::with::ignore=C901": "complexity is ratcheted by test_c901_debt_ratchet and the function-length ratchet instead",
    r"ci.yml::with::interrogate-fail-under=100": "100 is the strictest bar, not a lowered one",
    r"ci.yml::run::--ignore=": "a parse of mypy's --ignore-missing-imports in the consumer-position type check, not a path ignore",
    r"deep-nightly.yml::run::--ignore=tests/training/test_core.py": "the RuntimeWarning census mirrors the per-push selection; test_core.py has its own job",
    r"numba-coverage.yml::run::--ignore=tests/feature_selection/biz_val": "business-value fits are slow and gate outcomes, not line coverage of numba bodies",
    r"numba-coverage.yml::run::--ignore=tests/training/test_core.py": "run by the dedicated serial test-heavy-serial job",
    r"pre-commit::mypy::files=^(src/mlframe/calibration/|src/mlframe/utils/safe_pickle\.py$|src/mlframe/system/_gpu_guard\.py$|src/mlframe/metrics/(_numba_params|rank_correlation|_core_precision_mape)\.py$)": "the strict-typed beachhead modules; pinned to pyproject's override list by test_precommit_mypy_beachhead_coverage",
    r"pyproject::[tool.ruff]::exclude": "tests use the tests ruff config via their own hook; the rest is not the shipped package",
    r"pyproject::[tool.mypy]::exclude": "frozen bench/profiling scripts, not the package",
}


def test_blocking_gate_narrowings_are_declared():
    """A blocking gate that skips files or lowers its bar says why, here; an undeclared one is a gate quietly shrinking."""
    from py_ci_shared.gate_integrity import assert_coverage_gate_parity, assert_narrowings_declared

    assert_narrowings_declared(REPO_ROOT / ".pre-commit-config.yaml", WORKFLOWS_DIR, _DECLARED_NARROWINGS, PYPROJECT,
                               ("tool.ruff", "tool.mypy", "tool.pytest.ini_options", "tool.coverage.report"))
    assert_coverage_gate_parity(PYPROJECT, WORKFLOWS_DIR)


def test_gates_run_their_tools_with_the_project_config():
    """A hook or CI step running a configured tool must pass its config, and a gate called blocking must be able to fail.

    Its first run found both blocking bandit hooks running without `-c pyproject.toml`, so `[tool.bandit]` never
    applied there.
    """
    from py_ci_shared.gate_config_honesty import assert_gates_honest

    assert_gates_honest(REPO_ROOT / ".pre-commit-config.yaml", sorted(WORKFLOWS_DIR.glob("*.yml")), PYPROJECT)


def _pytest_runner_commands() -> list:
    """Every CI step that runs pytest, as (label, command) pairs."""
    from py_ci_shared.gate_config_honesty import gate_commands

    return [(label, command) for label, (command, _name) in gate_commands(None, sorted(WORKFLOWS_DIR.glob("*.yml"))).items() if "pytest" in command]


def _pytest_addopts() -> str:
    """Return the pytest addopts from pyproject.toml as a single string."""
    from py_ci_shared._toml_compat import tomllib

    addopts = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"].get("addopts", "")
    return " ".join(addopts) if isinstance(addopts, list) else str(addopts)


@pytest.mark.parametrize("marker,min_marked", [("slow", 100), ("gpu", 50), ("fuzz", 1)])
def test_marked_tests_are_selected_by_some_ci_run(marker, min_marked):
    """The per-push run deselects `slow`/`gpu`; some other CI run must select them, or they never run anywhere."""
    from py_ci_shared.marker_runner_coverage import assert_every_marked_test_is_selected

    assert_every_marked_test_is_selected(REPO_ROOT / "tests", REPO_ROOT, marker=marker, commands=_pytest_runner_commands(),
                                         addopts=_pytest_addopts(), min_marked=min_marked)


# Every status word a tracker row in audits/ uses. Wider than the checker's default four because the rounds were
# written weeks apart with their own words (DONE in the mrmr rounds, CLOSED per report in 2026-07-21, FIXED /
# CONSOLIDATED in 2026-09-05, COMPLETE per cluster in 2026-08-28); mapping them onto four would change what the
# rows say. A new word has to be added here, which is the point: a typo'd status is otherwise uncounted.
_TRACKER_STATUSES = (
    "RESOLVED", "WON'T FIX", "DEFERRED", "NOT A DEFECT", "TODO", "DONE", "FIXED", "CLOSED", "COMPLETE", "REJECTED",
    "DOC", "FUTURE", "PARTIAL", "PARTIALLY RESOLVED", "CONSOLIDATED", "NOT CONSOLIDATED", "SUPERSEDED", "CHECKED",
    "UNRESOLVED",
)
_TRACKERS = sorted((REPO_ROOT / "audits").glob("*/_TRACKER.md"))
# The eleven rounds that exist today; the floor stops a moved or renamed audits/ tree from parametrising to nothing.
_MIN_TRACKERS = 11


def test_audit_trackers_exist():
    """The tracker glob still finds the rounds, so the parametrised check below is not silently empty."""
    assert len(_TRACKERS) >= _MIN_TRACKERS, f"only {len(_TRACKERS)} audits/*/_TRACKER.md found; expected at least {_MIN_TRACKERS}"


@pytest.mark.parametrize("tracker", _TRACKERS, ids=lambda p: p.parent.name)
def test_audit_tracker_statuses_are_countable(tracker):
    """Every tracker row that names a status names it as the first cell in the one `**WORD**` spelling.

    Its first run found no tracker in that form: every status was free text in the LAST column
    (``RESOLVED (base_seed forwarded; ...)``), so nothing could count a round. Converting them surfaced ten
    2026-08-05 rows in a table with no header that no count had ever included, one round (2026-09-01) whose
    tracker carried no status at all, and one (reporting 2026-09-06) with no tracker.
    """
    from py_ci_shared.audit_round_format import assert_tracker_statuses_countable

    assert_tracker_statuses_countable(tracker, statuses=_TRACKER_STATUSES, min_rows=5)


# Trackers carrying a `| File | Findings | <STATUS> ... |` summary over `### `<file>`` sections of rows. The older
# rounds keep their counts in prose and per-severity headings, which this check does not parse; they were
# recounted by hand when their rows were converted.
_SUMMARISED_TRACKERS = ("audits/reporting_audit_2026-09-06/_TRACKER.md",)


@pytest.mark.parametrize("tracker", _SUMMARISED_TRACKERS)
def test_audit_tracker_summaries_agree_with_rows(tracker):
    """A tracker's summary counts are recomputed from its rows and must match.

    Its first run was on a summary written for it. The hand recount that preceded it found three stale summaries:
    2026-08-05 said 67 P1 over 68 rows, 2026-08-28's per-cluster dispositions predated the reconciliation its own
    prose describes (13 FUTURE against 3), and 2026-09-05's 124 findings sit on 125 rows.
    """
    from py_ci_shared.tracker_summary_parity import assert_tracker_summaries_agree

    assert_tracker_summaries_agree(REPO_ROOT / "audits", REPO_ROOT / tracker, statuses=_TRACKER_STATUSES)
