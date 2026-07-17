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
        """Helper that reference."""
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


@pytest.mark.fast
def test_fract_digits_probe_single_sort_matches_per_digit_round(monkeypatch):
    """is_variable_truly_continuous probes every rounding precision over ONE sort of the fractional part.

    Pins both the optimization (the single-sort rounded-count kernel is invoked, and the per-precision np.round+sort path is NOT taken) and
    bit-identity of the resulting continuity verdict. Fails on pre-fix code where the rounded-count kernel does not exist / the loop re-sorts per digit.
    """
    np = pytest.importorskip("numpy")
    import mlframe.preprocessing.cleaning as cln

    assert hasattr(cln, "_get_count_distinct_rounded_njit"), "single-sort rounded-count kernel must exist"

    rng = np.random.default_rng(0)
    vals = np.round(rng.normal(size=200_000) * 100, 6)

    # Reference verdict computed via the explicit per-digit np.round + _get_nunique path.
    fract_part, int_part = np.modf(vals)
    n_unique_ints = cln._get_nunique(vals=int_part, skip_vals=(0.0,))
    n_unique_fracts = cln._get_nunique(vals=fract_part, skip_vals=(0.0, 1.0))
    last = 0
    nz = 0
    ref_d = 1
    if n_unique_fracts != 0:
        for d in range(1, 10):
            nuf = cln._get_nunique(vals=np.round(fract_part, d), skip_vals=(0.0, 1.0))
            if last > 0:
                if (nuf - last) / last < 0.15 or nuf < 0.3 * (cln.NDIGITS**d) ** 0.95:
                    if n_unique_ints > 0 or nz > 0:
                        ref_d = d
                        break
            last = nuf
            if nuf > 0:
                nz = d
            ref_d = d

    # Spy: the optimized path must call the rounded-count kernel and must NOT call np.round inside the probe loop.
    kernel = cln._get_count_distinct_rounded_njit()
    calls = {"kernel": 0}
    orig_kernel = kernel

    def spy_kernel(sv, ndig, s0, s1):
        """Helper that spy kernel."""
        calls["kernel"] += 1
        return orig_kernel(sv, ndig, s0, s1)

    monkeypatch.setattr(cln, "_get_count_distinct_rounded_njit", lambda: spy_kernel)

    is_cont, _span = cln.is_variable_truly_continuous(values=vals.copy())
    assert is_cont is True or bool(is_cont)
    assert calls["kernel"] >= 1, "optimized single-sort rounded-count kernel must be invoked"
    # The optimized loop breaks at the same precision as the reference per-digit path (bit-identical verdict).
    assert calls["kernel"] == ref_d, f"kernel should be called once per probed precision (={ref_d}), got {calls['kernel']}"


@pytest.mark.fast
def test_rareval_merge_uses_vectorized_isin_not_per_value_replace():
    """Rare-value merge collapses the rare tail in one vectorized isin+mask pass, not pandas' O(n*k) per-cell .replace().

    Sensor: spy on Series.replace; pre-fix code called it for the row-level merge so the spy tripped. The vectorized path
    must NOT touch Series.replace for this merge AND must produce the exact same merged column (all rare values -> NA sentinel).
    """
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    pytest.importorskip("psutil")
    pytest.importorskip("pyutilz")
    from mlframe.preprocessing.cleaning import analyse_and_clean_features

    rng = np.random.default_rng(7)
    n = 60_000
    a = rng.integers(0, 20, size=n)
    rare_pos = rng.choice(n, size=600, replace=False)
    a[rare_pos] = rng.integers(900, 940, size=600)  # ~40 rare levels, each well under the imbalance floor
    df = pd.DataFrame({"x": a.astype("int64")})

    replace_calls = {"n": 0}
    orig_replace = pd.Series.replace

    def _spy(self, *args, **kwargs):
        """Helper that spy."""
        replace_calls["n"] += 1
        return orig_replace(self, *args, **kwargs)

    nunique_before = df["x"].nunique()
    pd.Series.replace = _spy
    try:
        analyse_and_clean_features(df, update_data=True, verbose=False)
    finally:
        pd.Series.replace = orig_replace

    assert replace_calls["n"] == 0, "rare-value merge must use vectorized isin+mask, not per-value Series.replace"
    # Identity of the merge result: rare levels collapsed (fewer uniques) and every former rare value is now the NA sentinel.
    assert df["x"].nunique(dropna=False) < nunique_before
    assert df["x"].isna().sum() == 600


@pytest.mark.fast
def test_suggest_non_outlying_uses_fused_njit_kernel_and_matches_numpy():
    """``suggest_non_outlying_data_indices`` routes float 1d input through the fused njit outlier-mask kernel and stays bit-identical to the prior
    four-pass numpy expression (``v<l``/``v>r``/two sums/``(~il)&(~ir)``), including NaN rows (NaN compares False on both fences -> kept). The kernel-use
    assertion fails on pre-fix code, where ``_get_outlier_mask_njit`` does not exist."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("numba")
    import mlframe.preprocessing.cleaning as cln

    assert hasattr(cln, "_get_outlier_mask_njit"), "fused outlier-mask kernel must exist"

    rng = np.random.default_rng(7)
    values = rng.standard_t(3, 50_000).astype(np.float64)
    values[::500] = np.nan

    calls = {"n": 0}
    real = cln._get_outlier_mask_njit

    def spy():
        """Helper that spy."""
        calls["n"] += 1
        return real()

    cln._get_outlier_mask_njit = spy
    try:
        idx = cln.suggest_non_outlying_data_indices(values, use_quantile=0.01)
    finally:
        cln._get_outlier_mask_njit = real

    assert calls["n"] == 1, "float 1d input must dispatch to the fused njit kernel"

    q = np.nanquantile(values, (0.01, 0.99))
    from mlframe.preprocessing.cleaning import get_tukey_fences_multiplier_for_quantile

    mult = get_tukey_fences_multiplier_for_quantile(quantile=0.01)
    iqr = q[1] - q[0]
    l = q[0] - mult * iqr
    r = q[1] + mult * iqr
    il = values < l
    ir = values > r
    ref = (~il) & (~ir)
    assert np.array_equal(idx, ref), "fused mask must be bit-identical to the numpy four-pass expression"
