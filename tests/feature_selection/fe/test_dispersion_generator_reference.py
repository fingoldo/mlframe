"""The conditional-dispersion generator matches a plain numpy reference, with and without non-finite values.

The generator shares per-column statistics across pairs and uses compiled single-pass reductions; this pins its output to the straightforward definition
``z = (x_i - mean_bin) / std_bin`` (population std, global std for small or constant bins).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _extra_fe_families_dispersion as disp
from mlframe.feature_selection.filters._extra_fe_families import _digitize_with_edges, _quantile_edges


def _reference(xi, xj, n_bins, kind):
    """Slow definition: per-bin mean/std of the finite x_i, floors as documented, NaN rows emit 0."""
    edges = _quantile_edges(xj, n_bins)
    codes = _digitize_with_edges(xj, edges)
    nb = edges.size - 1
    finite = np.isfinite(xi)
    gm = xi[finite].mean()
    gs = xi[finite].std()
    if not np.isfinite(gs) or gs < disp._DISPERSION_SIGMA_FLOOR:
        gs = 1.0
    z = np.zeros(xi.size)
    for b in range(nb):
        rows = finite & (codes == b)
        cnt = rows.sum()
        mean = xi[rows].mean() if cnt else gm
        std = xi[rows].std() if cnt else 0.0
        if cnt < disp._DISPERSION_MIN_BIN_ROWS or std < disp._DISPERSION_SIGMA_FLOOR:
            std = gs
        z[rows] = (xi[rows] - mean) / std
    return np.abs(z) if kind == "absz" else z * z


@pytest.mark.parametrize("nan_frac", [0.0, 0.15])
def test_generator_matches_the_plain_reference(nan_frac):
    """Every emitted column equals the reference for its (x_i, x_j, kind)."""
    rng = np.random.default_rng(5)
    n = 3_000
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.exponential(size=n), "c": rng.normal(size=n) * 4 + 2})
    df["a"] = df["a"] * (1 + np.abs(df["b"]))
    if nan_frac:
        for col in df.columns:
            df.loc[rng.random(n) < nan_frac, col] = np.nan
    out, recipes = disp.generate_conditional_dispersion_features(df, ["a", "b", "c"], n_bins=8, kinds=("absz", "z2"))
    assert len(out.columns) == 12
    for name, rec in recipes.items():
        want = _reference(df[rec["x_i"]].to_numpy(), df[rec["x_j"]].to_numpy(), 8, rec["kind"])
        np.testing.assert_allclose(out[name].to_numpy(), want, rtol=1e-9, atol=1e-12)


def test_a_constant_emission_is_skipped():
    """A column whose dispersion fold is constant carries no information and is not emitted."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": np.ones(500), "b": rng.normal(size=500)})
    out, _ = disp.generate_conditional_dispersion_features(df, ["a", "b"], n_bins=5, kinds=("absz",))
    assert not any(c.startswith("a__") for c in out.columns)


def test_std_njit_matches_numpy():
    """The temporary-free std agrees with np.std."""
    a = np.random.default_rng(2).normal(7.0, 3.0, 10_000)
    assert abs(disp._std_njit(a) - a.std()) < 1e-12
