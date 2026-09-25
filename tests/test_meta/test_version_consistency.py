"""``mlframe.__version__`` is reported consistently across every source of truth.

Sources: ``mlframe.__version__`` (the package re-export) and ``mlframe.version.__version__`` (the constant),
both read by import because the re-export only exists at runtime. ``pyproject.toml`` declares the version
``dynamic`` (read from ``mlframe.version``), so it states nothing to compare and is not asked for.
`py_ci_shared.version_consistency` compares the rest and refuses a comparison of fewer than two, which would
always agree with itself.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.version_consistency import assert_versions_agree

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_consistent_across_sources() -> None:
    """The package re-export and the version module agree."""
    assert_versions_agree(REPO_ROOT, modules=["mlframe", "mlframe.version"], pyproject=False)
