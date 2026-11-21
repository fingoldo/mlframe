"""Bucketing an unmeasurable (NaN) fingerprint dimension keeps the marker and does not raise (regression for the round-of-a-string crash)."""

from mlframe.utils._param_oracle import _NAN_BUCKET, bucketize_fingerprint


def test_nan_unknown_dimension_keeps_the_marker_instead_of_raising():
    """An unknown numeric dimension that is NaN buckets to the marker; before the fix round() of the string raised TypeError."""
    out = bucketize_fingerprint({"some_extra_dim": float("nan")})
    assert out["some_extra_dim"] == _NAN_BUCKET


def test_finite_unknown_dimension_is_still_rounded():
    """A finite value buckets to the 0.1 linear grid (1.23 -> 1.2) and is returned as a float."""
    out = bucketize_fingerprint({"some_extra_dim": 1.23456789})
    assert out["some_extra_dim"] == 1.2
