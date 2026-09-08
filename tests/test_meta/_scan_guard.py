"""Fail-closed guard for the AST/text meta-gates that scan a derived source root.

Roughly fifty meta-tests in this directory locate the tree they check by walking up from ``__file__``
(``Path(__file__).resolve().parents[N] / "src" / "mlframe"``) and then ``rglob``-ing it. That derivation is
silent when it breaks: a directory rename, an off-by-one in ``parents[N]``, or running from an installed
wheel yields a path that does not exist or holds nothing, the glob returns an empty sequence, the gate finds
no violations, and it reports green forever while checking nothing. A ``pytest.skip`` in that situation is
no better -- a skipped test is green in CI too.

The fix is one assertion: a gate that scanned far fewer files than the tree actually holds did not pass, it
failed to run. Use :func:`assert_scanned_enough` right after the scan, before evaluating any findings.
"""

from __future__ import annotations

# The src tree held ~1550 .py files when this guard was added. The floor is deliberately far below that:
# it exists to separate "scanned nothing" from "scanned the tree", not to ratchet the file count.
DEFAULT_MIN_FILES = 500


def assert_scanned_enough(scanned: int, what: str, minimum: int = DEFAULT_MIN_FILES) -> None:
    """Assert a meta-gate's scan actually reached the tree it means to check.

    Args:
        scanned: how many files the gate examined.
        what: what was being scanned, for the failure message (e.g. ``"src/mlframe"``).
        minimum: the floor below which the scan is treated as broken rather than clean.

    Raises:
        AssertionError: if fewer than ``minimum`` files were scanned.
    """
    assert scanned >= minimum, (
        f"meta-gate scanned only {scanned} file(s) of {what}, expected at least {minimum}. "
        f"This is a broken scan reporting as a clean one: check the root-path derivation "
        f"(a renamed directory, a wrong parents[N], or running outside a source checkout)."
    )
