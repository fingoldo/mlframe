"""Wire the cross-project checks py-ci-shared ships that mlframe never ran.

Every rule below is implemented and unit-tested in ``py_ci_shared``; this module is purely the consumption
point plus this repo's own allowlists. mlframe consumed three of roughly forty-six shared checks, so
maintained and directly-applicable rules sat unused while the findings they cover kept landing here -- the
same gap pyutilz closed with its own `test_shared_checks_wired.py`, which this mirrors.

Keeping them in ONE file makes "which shared checks does this repo actually run?" answerable by reading a
single import block, rather than by grepping for `py_ci_shared` across a hundred-odd meta modules.

Checks already covered by a first-party mlframe meta test are deliberately NOT re-wired here: the LOC budget
(`test_no_file_over_1k_loc.py`), import cycles (`test_no_import_cycles.py`), mutable defaults
(`test_no_mutable_defaults.py`), the code-audit ratchet (`test_code_audit_baseline.py`), README env-var
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
_REVIEWED_ADVISORY_STEPS: set[str] = set()

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
    import json

    found = _stale_comment_keys()
    accepted = json.loads(_STALE_COMMENT_BASELINE.read_text(encoding="utf-8")) if _STALE_COMMENT_BASELINE.exists() else {}

    new = {k: v for k, v in found.items() if k not in accepted}
    assert not new, "new stale comment(s) -- do it, delete it, or reference an issue:\n  " + "\n  ".join(f"{k}: {v}" for k, v in sorted(new.items()))

    drained = [k for k in accepted if k not in found]
    if drained:
        print(f"\n[stale-comments] {len(drained)} baseline entr(y/ies) DRAINED; refresh to lock the smaller set in.")


def regenerate_baseline() -> None:
    """Rewrite the stale-comment baseline from the current tree. Called by `regen_baselines.py`."""
    import json

    payload = json.dumps(dict(sorted(_stale_comment_keys().items())), indent=2, ensure_ascii=False)
    _STALE_COMMENT_BASELINE.write_text(payload + chr(10), encoding="utf-8")
