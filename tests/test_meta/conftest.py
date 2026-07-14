"""Conftest for tests/test_meta/ - register --refresh-* baseline flags.

Several meta-tests in this directory implement a "snapshot" pattern: they
compare current state against a frozen baseline file and fail when net
new offenders are introduced. Each such test exposes a CLI flag like
``--refresh-foo-baseline`` which (a) overwrites the baseline file from
current state and (b) makes the test pass that same run.

Each individual test probes ``sys.argv`` to detect its own flag, but
pytest itself (in --strict-config mode) refuses unknown CLI args before
they reach test code. Register all the refresh flags here as no-op
options so pytest accepts them on the command line; the tests then read
them from sys.argv as before.
"""

from __future__ import annotations

_REFRESH_FLAGS = [
    "--refresh-api-snapshot",
    "--refresh-annotation-baseline",
    "--refresh-bare-except-baseline",
    "--refresh-console-unicode-baseline",
    "--refresh-debt-baseline",
    "--refresh-docstring-baseline",
    "--refresh-logger-baseline",
    "--refresh-mutable-defaults-baseline",
    "--refresh-resource-handle-baseline",
]


def pytest_addoption(parser):
    """Register every ``--refresh-*-baseline`` flag as a no-op boolean toggle."""
    for flag in _REFRESH_FLAGS:
        # action=store_true so each flag is a boolean toggle; default off.
        # The tests themselves check sys.argv for the literal flag string,
        # so we don't need to expose the parsed value via a fixture.
        try:
            parser.addoption(flag, action="store_true", default=False, help=f"meta-test snapshot refresh: {flag}")
        except ValueError:
            # already registered elsewhere; ignore
            pass
    from py_ci_shared.code_audit_meta import register_refresh_option

    register_refresh_option(parser)  # --refresh-code-audit-baseline, shared with every other consumer
