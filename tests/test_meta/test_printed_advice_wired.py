"""Every advising message in composite/ is registered with the test that follows it (py_ci_shared.printed_advice)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.training.composite import test_printed_advice as advice_tests

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_every_composite_advice_has_a_test_that_follows_it():
    """A new message that tells the reader to act needs an entry in PRINTED_ADVICE_TESTS and the test it names; a removed one its entry gone."""
    printed_advice = pytest.importorskip("py_ci_shared.printed_advice")
    root = REPO_ROOT / "src" / "mlframe" / "training" / "composite"
    files = sorted(p for p in root.rglob("*.py") if "_benchmarks" not in p.parts)
    found = {a.key: a for a in printed_advice.find_printed_advice(files, REPO_ROOT)}
    table = advice_tests.PRINTED_ADVICE_TESTS
    assert len(found) >= 15, "the scan lost its subject"
    unregistered = sorted(repr(found[k]) for k in set(found) - set(table))
    assert not unregistered, "add a test that follows each advice to PRINTED_ADVICE_TESTS: " + "; ".join(unregistered)
    assert not sorted(set(table) - set(found)), f"stale PRINTED_ADVICE_TESTS keys: {sorted(set(table) - set(found))}"
    missing = sorted({t for t in table.values() if not callable(getattr(advice_tests, t, None))})
    assert not missing, f"PRINTED_ADVICE_TESTS names tests that do not exist: {missing}"
