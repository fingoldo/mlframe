"""Every run log names the interpreter and the library versions it ran with.

A production crash dump pointed at LightGBM and the log could not say which LightGBM, nor which interpreter: the run
used one from another drive than the checkout that was read afterwards.
"""

from __future__ import annotations

import logging
import sys

from mlframe.training._environment_report import (
    REPORTED_PACKAGES,
    environment_summary,
    log_environment_versions,
    package_versions,
)


def test_versions_cover_the_libraries_a_crash_is_blamed_on():
    """numpy and the installed boosters must be named; an absent package is simply left out."""
    versions = package_versions()
    assert versions["numpy"]
    assert "lightgbm" in versions, "lightgbm is installed in this environment and must be reported"
    assert set(versions) <= set(REPORTED_PACKAGES)


def test_summary_names_the_interpreter_and_platform():
    """The interpreter path is the part that distinguishes two installs of the same Python version."""
    summary = environment_summary()
    assert sys.executable in summary
    assert sys.version.split()[0] in summary
    assert "numpy=" in summary


def test_banner_is_logged_once_at_info(caplog):
    """The banner goes to the run log, not only to a return value."""
    with caplog.at_level(logging.INFO):
        returned = log_environment_versions()
    messages = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Environment: ")]
    assert len(messages) == 1
    assert returned and returned in messages[0]


def test_a_broken_distribution_does_not_cost_the_banner(monkeypatch):
    """Reporting is diagnostics: a metadata lookup that raises must not take the run down with it."""
    import mlframe.training._environment_report as er

    def _raise(_name):
        raise RuntimeError("metadata is unreadable")

    monkeypatch.setattr("importlib.metadata.version", _raise)
    monkeypatch.setitem(sys.modules, "numpy", sys.modules["numpy"])
    versions = er.package_versions()
    assert versions.get("numpy"), "the already-imported module's __version__ is the fallback"


def test_lookup_does_not_import_a_reported_package():
    """Reporting must not pay torch's or cupy's import cost, nor change what the run has loaded."""
    before = set(sys.modules)
    package_versions()
    assert not {m for m in set(sys.modules) - before if m.split(".")[0] in ("torch", "cupy", "shap", "catboost")}
