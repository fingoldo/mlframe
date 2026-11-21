"""Audit 2026-11-20 finding 04-F25: a non-default SAPP distribution takes its shape parameters first, with loc and scale applied after."""

from mlframe.data.datasets._sapp import get_sapp_dataset


def test_gamma_columns_have_the_loc_scale_mean():
    """Gamma(shape=3) with loc=5, scale=2 has mean 5 + 2*3 = 11 on every column; a swapped argument order would miss by far more."""
    df, _ = get_sapp_dataset(loc=5.0, scale=2.0, distr_name="gamma", distr_params=(3.0,), N=20000, binarize=False)
    assert abs(float(df["X1"].mean()) - 11.0) < 0.3
