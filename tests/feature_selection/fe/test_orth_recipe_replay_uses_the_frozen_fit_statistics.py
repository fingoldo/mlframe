"""Orth-basis recipes replay a row identically alone and inside a batch once the fit-time NaN fill and tanh std are frozen into them."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.engineered_recipes import apply_recipe, build_orth_univariate_recipe, freeze_orth_fit_stats


def _recipe(pre_transform):
    """A Chebyshev degree-2 recipe on column x with the given pre-transform (no frozen basis params, so the basis axis refits from the replayed batch in both cases)."""
    return build_orth_univariate_recipe(name="r", src_name="x", basis="legendre", degree=2, pre_transform=pre_transform)


def _fit_frame():
    """A fit column with a known mean and std."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": 10.0 + 10.0 * rng.standard_normal(500)})


def test_nan_fill_is_the_fit_mean_not_the_batch_mean():
    """A NaN row alone and the same row inside [NaN, 10, 20] must get the same basis value once the fit-time fill is frozen."""
    rec = _recipe("raw")
    freeze_orth_fit_stats([rec], _fit_frame())
    alone = np.asarray(apply_recipe(rec, pd.DataFrame({"x": [np.nan]})))
    batch = np.asarray(apply_recipe(rec, pd.DataFrame({"x": [np.nan, 10.0, 20.0]})))
    assert "nan_fill" in rec.extra
    assert abs(float(alone[0]) - float(batch[0])) < 1e-9, (alone, batch)


def test_tanh_std_is_the_fit_std_not_the_batch_std():
    """tanh(x / sd): a one-row batch has sd 0 and fell back to tanh(x); the frozen fit std gives the fit-time value."""
    rec = _recipe("tanh")
    fit = _fit_frame()
    freeze_orth_fit_stats([rec], fit)
    assert abs(rec.extra["pre_sd"] - float(np.std(fit["x"]))) < 1e-9
    alone = np.asarray(apply_recipe(rec, pd.DataFrame({"x": [5.0]})))
    full = np.asarray(apply_recipe(rec, pd.DataFrame({"x": [5.0, 15.0, 25.0]})))
    assert abs(float(alone[0]) - float(full[0])) < 1e-9, (alone, full)
