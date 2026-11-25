"""Cluster 28 regression: a failed grid variant is a typed GridVariantError (audit 2026-11-20, finding 28-F70).

With stop_on_error=False a raising variant is stored as GridVariantError with its label, exception class and repr; the other variants
keep their results.
"""

from mlframe.training.grid import GridVariantError, run_grid


def _suite(**kwargs):
    """Stub suite: raises for the variant marked ``boom``, returns a marker otherwise."""
    if kwargs.get("boom"):
        raise ValueError("bad variant")
    return {"ok": kwargs.get("tag")}


def test_failed_variant_is_typed_error_with_attributes():
    """The failing variant is a GridVariantError carrying label, exc_type and message (28-F70)."""
    results = run_grid({}, [("good", {"tag": "g"}), ("bad", {"boom": True})], suite_fn=_suite, stop_on_error=False)
    err = results["bad"]
    assert isinstance(err, GridVariantError)
    assert err.label == "bad"
    assert err.exc_type == "ValueError"
    assert "bad variant" in err.message


def test_successful_variant_is_unchanged():
    """Control: a successful variant keeps the suite's return value (28-F70)."""
    results = run_grid({}, [("good", {"tag": "g"})], suite_fn=_suite, stop_on_error=False)
    assert results["good"] == {"ok": "g"}
    assert not isinstance(results["good"], GridVariantError)
