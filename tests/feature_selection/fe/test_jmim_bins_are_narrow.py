"""The JMIM column bins are held at the narrowest width that holds their codes (PLAN-10 memory): the same values, an eighth of the bytes.

``_bin_columns`` returned one int64 array per column (590 MB for 100000 rows x 572 columns at the peak of an every-family fit); every consumer casts a column to int64 or uint8 where it is used, so the codes
only need to hold the values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _orthogonal_jmim_fe as J


def _frame(n: int = 3000, p: int = 6, seed: int = 0):
    """Gaussian columns plus a constant one."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"c{i}" for i in range(p)])
    df["const"] = 1.0
    return df


def test_bins_are_uint8_with_the_same_codes_as_the_int64_reference():
    """Every column is uint8; its values and its level count equal those of the dense int64 batched codes."""
    df = _frame()
    bins, ks, names = J._bin_columns(df, nbins=10)
    assert names == list(df.columns)
    ref = J._quantile_bin_batched(np.ascontiguousarray(df.to_numpy(), dtype=J._crit_np_dtype()), nbins=10)
    assert len(bins) == len(names) > 0, "no column was binned, so the loop below would check nothing"
    for j, b in enumerate(bins):
        assert b.dtype == np.uint8
        assert np.array_equal(b.astype(np.int64), ref[:, j])
        assert ks[j] == max(1, int(ref[:, j].max()) + 1)
    assert ks[-1] == 1, "a constant column is one level"


def test_the_jmim_scores_do_not_change():
    """The score of an engineered column is the same whether its bins come narrow or as int64 (the scorer widens at the point of use)."""
    rng = np.random.default_rng(1)
    n = 4000
    raw = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    eng = pd.DataFrame({"e1": raw["a"] * raw["b"], "e2": raw["a"] + 0.1 * rng.standard_normal(n), "e3": rng.standard_normal(n)})
    y = ((raw["a"] * raw["b"] + 0.3 * rng.standard_normal(n)) > 0).astype(np.int64).to_numpy()
    narrow = J.score_features_by_jmim(raw, eng, y, n_bins=10)
    real = J._bin_columns

    def wide(df, nbins):
        """The previous contract: int64 columns."""
        bins, ks, names = real(df, nbins)
        return [b.astype(np.int64) for b in bins], ks, names

    J._bin_columns = wide
    try:
        reference = J.score_features_by_jmim(raw, eng, y, n_bins=10)
    finally:
        J._bin_columns = real
    pd.testing.assert_frame_equal(narrow.reset_index(drop=True), reference.reset_index(drop=True))
