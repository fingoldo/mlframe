"""Meta-test: ``-p no:xdist`` alone must not crash the whole pytest run.

pytest-progress defines a ``pytest_xdist_node_collection_finished`` hookimpl whenever the xdist
PACKAGE is importable, regardless of whether the xdist PYTEST PLUGIN is currently registered.
Passing ``-p no:xdist`` unregisters the plugin but leaves the package installed (still importable),
orphaning that hookimpl -- pluggy's next ``check_pending()`` then raises ``PluginValidationError``
and pytest aborts with an INTERNALERROR before a single test runs. ``tests/conftest.py``'s
``pytest_configure`` works around this by unregistering pytest-progress's hookimpls whenever the
xdist plugin isn't registered.

The crash happens while plugins are configured, before collection finishes, so collecting this file
is enough to reach it. The probe used to be a throwaway test file written into this directory and
deleted afterwards; under ``-n 4`` a directory scan running beside it met the file half-way.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pytest_progress", reason="this regression only manifests when pytest-progress is installed")
pytest.importorskip("xdist", reason="this regression only manifests when pytest-xdist is installed")

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_xdist_flag_does_not_crash_with_plugin_validation_error():
    """Collecting under ``-p no:xdist`` must succeed, not crash at plugin validation."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(Path(__file__).resolve()), "-p", "no:xdist", "--no-cov"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        # Collecting even one file imports the whole conftest: 93 s measured on a loaded machine. The regression
        # itself crashes at plugin validation and exits at once, so the timeout only has to catch a hang.
        timeout=600,
    )
    combined = result.stdout + result.stderr
    assert "PluginValidationError" not in combined, combined
    assert "INTERNALERROR" not in combined, combined
    assert result.returncode == 0, combined
    assert "test_no_xdist_flag_does_not_crash_with_plugin_validation_error" in result.stdout, combined
