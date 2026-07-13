"""Wave 13 finding #5 (DORMANT path -- score_pair_mi/_score_plug_in have no caller in the per-pair MRMR loop,
verified by grep): _score_plug_in re-derived np.unique(y) every call for a low-cardinality float y. Light fix:
cache keyed on id(y_arr) via _get_unique_y, mirroring _mah._get_y_binning. Equivalence-only test since the path
is unreached in production -- no biz_value/perf investment warranted for a dead path."""
import numpy as np

from mlframe.feature_selection.filters._mi_dispatch import _get_unique_y, score_pair_mi


def test_get_unique_y_matches_np_unique():
    rng = np.random.default_rng(0)
    y = rng.choice([0.0, 1.0], size=200)
    assert np.array_equal(_get_unique_y(y), np.unique(y))
    # repeat call with the same object -- still correct (cache hit)
    assert np.array_equal(_get_unique_y(y), np.unique(y))


def test_score_pair_mi_plug_in_unaffected_by_unique_y_cache():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(size=n)
    y = rng.choice([0.0, 1.0], size=n)
    mi1 = score_pair_mi(x, y, estimator="plug_in")
    mi2 = score_pair_mi(x, y, estimator="plug_in")
    assert mi1 == mi2
    assert np.isfinite(mi1) and mi1 >= 0.0
