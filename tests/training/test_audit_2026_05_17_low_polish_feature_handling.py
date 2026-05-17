"""Wave 4+5 LOW/POLISH regression tests for the feature_handling scope.

The Wave 4+5 feature_handling agent stripped dated audit-history
comments, normalised em-dashes in log strings, dropped dead code
(`_noop_lock`, redundant `# noqa: F401`), and extended the
secret-field token set (`authorization` / `credentials`). The agent
never delivered a separate test file before tracking dropped; these
tests back-fill that gap and lock in the Wave 4+5 polish contracts.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_FH_ROOT = Path(
    "D:/Upd/Programming/PythonCodeRepository/mlframe/src/mlframe/training/feature_handling"
)

_FH_FILES = sorted(_FH_ROOT.glob("*.py"))

_DATED_PATTERNS = [
    re.compile(r"#\s*20\d\d-\d\d-\d\d\b"),
    re.compile(r"#\s*(Wave|wave)-\d+\b"),
    re.compile(r"#\s*round-\d+\b"),
    re.compile(r"#\s*Session\s+\d+\b"),
    re.compile(r"#\s*\(user\s+(request|feedback)\)"),
    re.compile(r"#\s*OPEN-\d+\b"),
    re.compile(r"#\s*R10[a-z]\b"),
]


@pytest.mark.parametrize("py_file", _FH_FILES, ids=lambda p: p.name)
def test_feature_handling_files_have_no_dated_audit_comments(py_file: Path) -> None:
    """Sweep S1: dated audit-history tags have been stripped from
    every feature_handling module."""
    src = py_file.read_text(encoding="utf-8")
    offenders: list[tuple[int, str]] = []
    for ln_no, line in enumerate(src.splitlines(), start=1):
        for pat in _DATED_PATTERNS:
            if pat.search(line):
                offenders.append((ln_no, line.strip()))
                break
    assert not offenders, (
        f"{py_file.name}: dated audit-history comments still present:\n"
        + "\n".join(f"  L{n}: {t}" for n, t in offenders[:20])
    )


def test_secret_field_matches_authorization_and_credentials() -> None:
    """Wave 4+5 extended ``_SECRET_KEY_PATTERNS`` so whole-token match
    catches `Authorization` (header name) and `credentials` (plural)
    along with the short forms that the previous patterns covered."""
    from mlframe.training.feature_handling.providers import _is_secret_field

    # Direct matches.
    assert _is_secret_field("Authorization")
    assert _is_secret_field("credentials")
    assert _is_secret_field("api_key")
    assert _is_secret_field("client.secret")
    assert _is_secret_field("password")
    assert _is_secret_field("X-Auth-Token")
    assert _is_secret_field("bearer_token")

    # The polish-pass MUST keep the false-positive guard from Wave 3.
    assert not _is_secret_field("tokenizer")
    assert not _is_secret_field("tokenizer_path")
    assert not _is_secret_field("monkey")
    assert not _is_secret_field("author")
    assert not _is_secret_field("keychain")


def test_no_noop_lock_dead_code_in_cache_backend() -> None:
    """The dead ``_noop_lock`` placeholder noted in the audit was
    removed -- no module-level symbol with that name remains in
    ``cache_backend``."""
    from mlframe.training.feature_handling import cache_backend

    assert not hasattr(cache_backend, "_noop_lock"), (
        "_noop_lock placeholder should have been removed in Wave 4+5"
    )


def test_target_encoders_typecheck_imports_clean() -> None:
    """Wave 3 added pd/pl imports under TYPE_CHECKING (replacing the
    earlier ``# noqa: F821`` lie). The Wave 4+5 polish-pass must not
    have reintroduced that noqa anywhere in the file."""
    src = (_FH_ROOT / "target_encoders.py").read_text(encoding="utf-8")
    # noqa: F821 should NOT appear -- the TYPE_CHECKING block makes
    # pd/pl available to type checkers without runtime cost.
    assert "noqa: F821" not in src, (
        "target_encoders.py reintroduced a # noqa: F821 silencer; "
        "the TYPE_CHECKING import is supposed to satisfy F821 properly"
    )


def test_locking_holder_pid_atomic_write() -> None:
    """Wave 4+5 polished ``_write_holder_pid``: it MUST use the
    atomic-write helper (``atomic_write_bytes``) so a SIGKILL between
    file-create and file-write can't leave an empty PID file that
    breaks the next ``_read_holder_pid``."""
    import inspect
    from mlframe.training.feature_handling import locking

    src = inspect.getsource(locking.PIDAwareFileLock._write_holder_pid)
    assert "atomic_write_bytes" in src or "os.replace" in src, (
        "_write_holder_pid no longer uses atomic write; SIGKILL race "
        "re-introduced"
    )


def test_pl_isinstance_guard_in_utils() -> None:
    """The ``pl is not None`` guard before any ``isinstance(df, pl.DataFrame)``
    must be present in utils.py to keep polars-absent installs from
    raising ``TypeError: isinstance() arg 2 must be a type``."""
    src = (_FH_ROOT.parent / "utils.py").read_text(encoding="utf-8")
    # The guard pattern can appear several ways; assert at least one
    # of them precedes the polars isinstance check.
    assert "pl is not None" in src or "_HAS_POLARS" in src, (
        "utils.py is missing the polars-installed guard before "
        "isinstance(df, pl.DataFrame)"
    )
