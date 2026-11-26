"""Cluster 32 regression: the profile harness's skipped-target result has the same arity as a normal result (audit 2026-11-20, finding 32-F28).

The normal path returns 16 fields that the caller unpacks; the skip paths returned 6. The skip paths now use one helper that builds the same 16-field
tuple, and the source no longer contains the 6-field returns.
"""

from mlframe.training._benchmarks import _profile_fuzz_1m_run_suite as suite


def test_skipped_result_has_sixteen_fields():
    """The helper returns the full 16-field tuple the caller unpacks (32-F28)."""
    out = suite._skipped_profile_result("UNSUPPORTED_TARGET_TYPE")
    assert len(out) == 16
    assert out[0] == 0.0 and out[1] is False and out[2] == "UNSUPPORTED_TARGET_TYPE"
