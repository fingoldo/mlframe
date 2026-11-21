"""Positions past one million are encoded as themselves, not wrapped onto smaller ones (audit 2026-11-20, finding 09-F8)."""
import numpy as np

from mlframe.feature_engineering.transformer.random_features import compute_positional_encoding


def test_position_one_million_plus_500_differs_from_500():
    """The old modulus made 1_000_500 identical to 500; the two positions must now encode differently."""
    out = compute_positional_encoding(np.array([500, 1_000_500], dtype=np.int64), d_model=8, base=10000.0, dtype=np.float64)
    arr = out.to_numpy()
    assert not np.allclose(arr[0], arr[1])
