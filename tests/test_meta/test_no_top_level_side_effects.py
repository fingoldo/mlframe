"""Importing mlframe performs no network I/O, no environment write and no database open of its own.

A module that reaches the network or a database at import time blocks every import of it, fails offline and in
sandboxes, and repeats the side effect in every process that imports it. The probe is
py_ci_shared.import_side_effects: a fresh interpreter with sockets and urlopen blocked, os.environ writes
recorded and open() of a database-style path refused. A side effect a module swallows still counts. Only this
project's own code is judged: numpy, joblib and urllib3 write the environment or open a socket while they import,
and those are theirs.
"""

from __future__ import annotations

from py_ci_shared.import_side_effects import assert_imports_have_no_side_effects

_TARGETS = [
    "mlframe",
]


def test_imports_have_no_side_effects_of_their_own():
    """No probed module opens a socket, writes os.environ or opens a database file while it is imported."""
    assert_imports_have_no_side_effects(
        _TARGETS,
        first_party=("mlframe",),
        block_environ=True,
        forbid_open_tokens=(".db", ".sqlite", "postgresql://", "postgres://"),
        isolate=False,
        timeout=900,
    )
