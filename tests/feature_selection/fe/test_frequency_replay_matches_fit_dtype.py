"""Frequency-encoding replay equals the fit column exactly (no float32 cast at serve time)."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._count_freq_interaction_fe import apply_frequency_encoding, frequency_encode_fit


def test_frequency_replay_is_bit_identical_to_the_fit_column():
    """Replaying the fit frame through the recipe returns the float64 fit column, not a float32 rounding of it."""
    cats = np.array(["a"] * 7 + ["b"] * 3 + ["c"] * 1 + ["d"] * 1)
    X = pd.DataFrame({"k": cats})
    enc, rec = frequency_encode_fit(X, ["k"])
    out = apply_frequency_encoding(X, "k", rec["k"])
    assert out.dtype == np.float64
    assert np.array_equal(out, enc["k__freq"].to_numpy())
