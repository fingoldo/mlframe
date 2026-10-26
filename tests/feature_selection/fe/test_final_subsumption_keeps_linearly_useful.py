"""A column the compounds subsume by information is still kept when a linear model gains from it; a noise column is not."""
from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl._final_subsumption_stage import _without_linearly_useful


def _category_problem(n=4000, seed=0):
    """An eight-level category with a non-monotone per-category effect on y, a compound that recodes it injectively (exp of the code) and the per-category mean of y."""
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 8, n)
    effect = np.array([6.0, -5.0, 3.0, 0.0, -3.0, 5.0, -1.0, 2.0])
    y = effect[cat] + rng.standard_normal(n)
    anchor = np.exp(cat.astype(float))
    target_mean = np.array([y[cat == k].mean() for k in range(8)])[cat]
    return y, anchor, target_mean, rng.standard_normal(n)


def test_target_encoding_beyond_an_injective_recoding_is_kept():
    """The per-category mean of y carries no information beyond exp(code), yet a linear model needs it: it leaves the dropped list."""
    y, anchor, target_mean, _noise = _category_problem()
    model = SimpleNamespace(fe_residual_rescue_operand_min_linear_gain=0.01)
    kept = _without_linearly_useful(model, ["te"], {"anchor": anchor, "te": target_mean}, ["anchor"], y)
    assert kept == []


def test_a_noise_column_stays_dropped():
    """A column unrelated to y gives a linear model nothing, so the information verdict stands."""
    y, anchor, _target_mean, noise = _category_problem()
    model = SimpleNamespace(fe_residual_rescue_operand_min_linear_gain=0.01)
    kept = _without_linearly_useful(model, ["noise"], {"anchor": anchor, "noise": noise}, ["anchor"], y)
    assert kept == ["noise"]
