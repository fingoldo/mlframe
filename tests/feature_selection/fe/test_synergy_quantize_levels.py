"""The low-cardinality branch of the synergy quantiser equals the dictionary lookup it replaced."""

import numpy as np

from mlframe.feature_selection.filters._synergy_detector import _quantize


def _lut_reference(col, nbins):
    """The per-row dict lookup of the original implementation."""
    col = np.asarray(col, dtype=np.float64).ravel()
    uniq = np.unique(col[np.isfinite(col)])
    lut = {v: i for i, v in enumerate(uniq.tolist())}
    return np.array([lut.get(v, 0) for v in col], dtype=np.int64)


def test_low_cardinality_codes_match_the_dict_lookup():
    """Binary, ternary and 8-level columns with NaN and inf rows, in float32 and float64."""
    rng = np.random.default_rng(3)
    for levels in (2, 3, 8):
        for dtype in (np.float64, np.float32):
            col = rng.integers(0, levels, 4000).astype(dtype)
            col[::9] = np.nan
            col[4::53] = np.inf
            got = _quantize(col, 10, rng)
            want = _lut_reference(col.astype(np.float64), 10)
            np.testing.assert_array_equal(got, want)
