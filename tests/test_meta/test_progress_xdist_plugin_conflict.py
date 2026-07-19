"""Meta-test: ``-p no:xdist`` alone must not crash the whole pytest run.

pytest-progress defines a ``pytest_xdist_node_collection_finished`` hookimpl whenever the xdist
PACKAGE is importable, regardless of whether the xdist PYTEST PLUGIN is currently registered.
Passing ``-p no:xdist`` unregisters the plugin but leaves the package installed (still importable),
orphaning that hookimpl -- pluggy's next ``check_pending()`` then raises ``PluginValidationError``
and pytest aborts with an INTERNALERROR before a single test runs. ``tests/conftest.py``'s
``pytest_configure`` works around this by unregistering pytest-progress's hookimpls whenever the
xdist plugin isn't registered.
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
    """A trivial run under ``-p no:xdist`` must collect and pass, not crash at collection."""
    probe = Path(__file__).resolve().parent / "_probe_no_xdist_flag.py"
    probe.write_text("def test_trivial():\n    assert True\n")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(probe), "-p", "no:xdist", "--no-cov", "-q"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        probe.unlink(missing_ok=True)
    combined = result.stdout + result.stderr
    assert "PluginValidationError" not in combined, combined
    assert "INTERNALERROR" not in combined, combined
    assert result.returncode == 0, combined
