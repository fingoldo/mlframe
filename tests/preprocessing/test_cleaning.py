"""Smoke test for mlframe.preprocessing.cleaning (W5-4).

Note: cleaning module operates on pandas DataFrames (see module docstring). The audit
brief mentioned pl.DataFrame, but the implementation predates the polars-first migration
of other subpackages. Smoke test uses pandas to exercise the real happy path.
"""

from __future__ import annotations

import pytest


@pytest.mark.fast
def test_analyse_and_clean_features_smoke():
    """analyse_and_clean_features returns a summary dict with the documented buckets."""
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    pytest.importorskip("psutil")
    pytest.importorskip("pyutilz")
    from mlframe.preprocessing.cleaning import analyse_and_clean_features

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "cont_feat": rng.normal(size=300).astype(np.float32),
            "discrete_feat": rng.integers(0, 5, size=300).astype(np.int32),
            "constant_feat": np.zeros(300, dtype=np.float32),
        }
    )
    result = analyse_and_clean_features(df, update_data=False, verbose=False)
    assert isinstance(result, dict)
    for key in (
        "dtypes",
        "features_ranges",
        "constant_features",
        "discrete_features",
        "continuous_features",
        "manyvalued_features",
        "features_transforms",
        "fewlyvalued_features",
    ):
        assert key in result, f"missing key {key} in result"
    assert "constant_feat" in result["constant_features"]


@pytest.mark.fast
def test_get_nunique_float_fastpath_bit_identical_to_npunique():
    """The float fast-path in ``_get_nunique`` (sort + njit count-distinct, no unique-array
    materialization) MUST return the same count as the reference ``np.unique`` filter route for
    every float input shape: plain, NaN-laced, falsy-scalar skip_vals, None skip, all-equal,
    all-NaN. A regression that changes the count silently moves continuous/discrete classification.
    """
    np = pytest.importorskip("numpy")
    pytest.importorskip("numba")
    from mlframe.preprocessing.cleaning import _get_nunique

    def reference(vals, skip_vals=None):
        u = np.unique(vals)
        if u.dtype.kind in ("f", "c"):
            u = u[~np.isnan(u)]
        if skip_vals:
            for v in skip_vals:
                u = u[u != v]
        return len(u)

    rng = np.random.default_rng(1)
    laced = rng.uniform(0, 5, 1000)
    laced[::7] = np.nan
    cases = [
        (rng.uniform(0, 5, 1000), (0.0, 1.0)),
        (laced, (0.0, 1.0)),
        (np.modf(rng.uniform(-10, 10, 1000))[1], (0.0)),  # falsy scalar -> no skip
        (rng.uniform(0, 1, 500), None),
        (np.zeros(100), (0.0, 1.0)),
        (np.full(50, np.nan), (0.0, 1.0)),
    ]
    for arr, sk in cases:
        arr = arr.astype(np.float64)
        assert _get_nunique(arr, skip_vals=sk) == reference(arr, skip_vals=sk)
