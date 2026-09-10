"""Every baseline file must be reachable by the machinery that refreshes and documents it.

There are three hand-maintained lists of the same thing and nothing compared them: the ``_*baseline*.json``
files on disk, ``regen_baselines.py``'s ``_BASELINES`` mapping, and the ``--refresh-*`` flags available to
this directory. The 2026-09-08 review measured 27 files against 7 regen entries and a differently-sized
flag list, so a baseline could be refreshable by no documented route at all.

This is the same parity class the repo already gates for pre-commit-vs-CI scope and for the mypy beachhead;
it had just never been pointed at the baselines themselves.

Flags are resolved by asking the live pytest config rather than by grepping ``conftest.py``: several are
registered indirectly through ``py_ci_shared``'s ``register_refresh_option``, which a text scan cannot see
and which would make every such baseline look unreachable.
"""

from __future__ import annotations

import re
from pathlib import Path

META_DIR = Path(__file__).resolve().parent

# Baselines whose refresh route is deliberately not a `--refresh-<stem>-baseline` flag. Each entry names
# the route that does exist, so "no flag" stays a decision on the record rather than an omission.
FLAG_EXEMPT: dict[str, str] = {
    # Flag predates the file name and renaming it would break the documented invocation.
    "_logger_lazy_baseline.json": "--refresh-logger-baseline",
    "_module_level_logging_disable_baseline.json": "--refresh-logging-disable-baseline",
    "_numba_config_env_mutation_baseline.json": "--refresh-numba-config-env-baseline",
    # Refreshed by the batch entry point instead of a per-gate flag.
    "_stale_comment_baseline.json": "python tests/test_meta/regen_baselines.py",
    # Shares the src-level scanner's flag rather than a file-derived one: `assert_no_new_code_audit_findings`
    # gates both `_code_audit_baseline.json` (src/) and this file (tests/) through the SAME
    # `--refresh-code-audit-baseline` option, so the file-name-derived `--refresh-code-audit-tests-baseline`
    # this gate would otherwise look for was never going to exist. See audits/ci_review_2026-09-08/_TRACKER.md
    # (L1.13b).
    "_code_audit_tests_baseline.json": "--refresh-code-audit-baseline",
}


def _baseline_files() -> set[str]:
    """Every ``_*baseline*.json`` file in tests/test_meta."""
    names = {p.name for p in META_DIR.glob("_*baseline*.json")}
    assert names, f"no baseline files found in {META_DIR}; this gate cannot run and must not report green"
    return names


def _expected_flag(name: str) -> str:
    """`_nondiscriminating_assert_baseline.json` -> `--refresh-nondiscriminating-assert-baseline`."""
    return "--refresh-" + name[: -len(".json")].lstrip("_").replace("_", "-")


def _is_registered(pytestconfig, flag: str) -> bool:
    """Whether pytest actually knows this option, however it was registered."""
    try:
        pytestconfig.getoption(flag)
    except ValueError:
        return False
    return True


def test_every_baseline_file_has_a_refresh_flag(pytestconfig):
    """A baseline nothing can regenerate is a file that drifts until someone edits it by hand."""
    unreachable = sorted(name for name in _baseline_files() if name not in FLAG_EXEMPT and not _is_registered(pytestconfig, _expected_flag(name)))
    assert not unreachable, (
        f"{len(unreachable)} baseline file(s) have no matching --refresh-* option. Register one (directly "
        f"or via py_ci_shared's register_refresh_option), rename the file to match its existing flag, or "
        f"record the real route in FLAG_EXEMPT:\n" + "\n".join(f"  {n}  (expected {_expected_flag(n)})" for n in unreachable)
    )


def test_regen_baselines_entries_all_exist_on_disk():
    """regen_baselines.py naming a file that is not there means its batch refresh silently does nothing."""
    text = (META_DIR / "regen_baselines.py").read_text(encoding="utf-8")
    listed = set(re.findall(r'"(_[a-z0-9_]*baseline[a-z0-9_]*\.json)"', text))
    assert listed, "expected baseline filenames in regen_baselines.py's _BASELINES mapping"
    missing = sorted(listed - _baseline_files())
    assert not missing, "regen_baselines.py lists baseline file(s) that do not exist: " + ", ".join(missing)


def test_baselines_readme_documents_every_baseline_file():
    """BASELINES_README.md is the documented route; a file absent from it is one nobody knows to refresh."""
    readme = META_DIR / "BASELINES_README.md"
    assert readme.is_file(), f"{readme} is missing"
    text = readme.read_text(encoding="utf-8")
    undocumented = sorted(name for name in _baseline_files() if name not in text)
    assert not undocumented, f"{len(undocumented)} baseline file(s) are not mentioned in BASELINES_README.md:\n" + "\n".join(f"  {n}" for n in undocumented)
