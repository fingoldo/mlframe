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
