"""Cluster 39 regression: object-column fingerprints depend on content, not object identity (audit 2026-11-20, finding 39-F21)."""

import numpy as np
import pandas as pd

from mlframe.training.feature_handling import fingerprint as fp


def _payload(values):
    """Pandas row payload for a single object column holding ``values``."""
    df = pd.DataFrame({"s": pd.Series(values, dtype=object)})
    return fp._pandas_rows_payload(df, ["s"], np.arange(len(values)))


def test_equal_string_content_gives_equal_payload_from_distinct_objects():
    """Two frames holding equal strings built independently hash the same (39-F21)."""
    a = ["".join(["ab", "cd"]), "xy"]
    b = ["".join(["a", "bcd"]), "xy"]
    assert _payload(a) == _payload(b)


def test_different_string_content_gives_different_payload():
    """Control: different content still changes the fingerprint (39-F21)."""
    assert _payload(["ab", "cd"]) != _payload(["ab", "ce"])
