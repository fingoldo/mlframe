"""The pyutilz revision is read from ``uv.lock``, and a lock without a git-pinned pyutilz is refused.

``uv.lock`` holds the pyutilz commit; CI reads it through ``scripts/read_pyutilz_pin.py`` instead of a hand-copied SHA. The test pins the
repository's own lock value and checks the refusal on a lock that lacks the entry.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import read_pyutilz_pin as rp


def test_repo_lock_yields_a_full_sha():
    """The committed uv.lock resolves pyutilz to a 40-hex commit SHA."""
    sha = rp.read_pyutilz_sha((REPO / "uv.lock").read_text(encoding="utf-8"))

    assert re.fullmatch(r"[0-9a-f]{40}", sha)


def test_lock_without_a_git_pinned_pyutilz_is_refused():
    """A lock whose pyutilz entry comes from the registry, not git, raises ValueError."""
    lock = 'name = "pyutilz"\nversion = "1.1.0"\nsource = { registry = "https://pypi.org/simple" }\n'

    with pytest.raises(ValueError, match="no git-pinned pyutilz"):
        rp.read_pyutilz_sha(lock)
