"""``mlframe.__version__`` is reported consistently across every source of truth.

Sources: ``mlframe.__version__`` (the package re-export) and ``mlframe.version.__version__`` (the constant),
both read by import because the re-export only exists at runtime, plus ``[project].version`` in the repository
``pyproject.toml`` whenever it declares one. `py_ci_shared.version_consistency` compares them and refuses a
comparison of fewer than two, which would always agree with itself.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.version_consistency import assert_versions_agree

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_consistent_across_sources() -> None:
    """The package re-export, the version module and any pyproject version all agree."""
    assert_versions_agree(REPO_ROOT, modules=["mlframe", "mlframe.version"])
