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


import mlframe as _mlframe

_FH_ROOT = Path(_mlframe.__file__).resolve().parent / "training" / "feature_handling"

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
    assert not offenders, f"{py_file.name}: dated audit-history comments still present:\n" + "\n".join(f"  L{n}: {t}" for n, t in offenders[:20])


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

    assert not hasattr(cache_backend, "_noop_lock"), "_noop_lock placeholder should have been removed in Wave 4+5"


def test_target_encoders_typecheck_imports_clean() -> None:
    """target_encoders.py has no undefined name even with every ``noqa`` ignored: pd/pl are imported under TYPE_CHECKING, not silenced."""
    import subprocess
    import sys

    result = subprocess.run(  # nosec B603 - fixed argv (sys.executable + literal ruff args), no shell
        [sys.executable, "-m", "ruff", "check", "--select", "F821", "--ignore-noqa", "--no-cache", "--isolated", str(_FH_ROOT / "target_encoders.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_locking_holder_pid_atomic_write(tmp_path, monkeypatch) -> None:
    """``_write_holder_pid`` writes the PID through a temp file and an atomic replace: a failing replace leaves the previous complete PID file intact."""
    import os
    from mlframe.training.feature_handling import locking

    target = tmp_path / "holder"
    instance = locking.PIDAwareFileLock(str(target))
    meta = tmp_path / "holder.pid"

    instance._write_holder_pid()

    assert meta.read_text(encoding="utf-8") == str(os.getpid())
    assert not (tmp_path / "holder.pid.tmp").exists()

    meta.write_text("424242", encoding="utf-8")

    def _failing_replace(src, dst, *a, **kw):
        """Simulates the process dying between writing the temp file and swapping it in."""
        raise OSError("simulated crash before the atomic swap")

    monkeypatch.setattr(os, "replace", _failing_replace)
    instance._write_holder_pid()

    assert meta.read_text(encoding="utf-8") == "424242"
    assert not (tmp_path / "holder.pid.tmp").exists()


def test_pl_isinstance_guard_in_utils(monkeypatch) -> None:
    """With polars absent (``pl`` is None) the column-drop and numpy-coercion helpers still handle pandas input instead of raising from ``isinstance(x, None)``."""
    import numpy as np
    import pandas as pd

    from mlframe.training import utils

    monkeypatch.setattr(utils, "pl", None)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

    dropped = utils.drop_columns_from_dataframe(df, ["b", "missing"], verbose=0)

    assert list(dropped.columns) == ["a", "c"]
    assert utils.coerce_to_numpy(pd.Series([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert isinstance(utils.coerce_to_numpy([1, 2, 3]), np.ndarray)
