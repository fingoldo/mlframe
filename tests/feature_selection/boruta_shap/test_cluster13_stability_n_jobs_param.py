"""Cluster 13 regression: stability_n_jobs is a real constructor parameter (audit 2026-11-20, finding 13-F20).

The stability sub-fits read ``stability_n_jobs`` with getattr although it was not a constructor argument, so ``get_params`` did not list it and ``clone``
(GridSearchCV, Pipeline) dropped any value a caller had set.
"""

from sklearn.base import clone

from mlframe.feature_selection.boruta_shap import BorutaShap


def test_stability_n_jobs_round_trips_through_get_params_and_clone():
    """The value is listed by get_params, defaults to -1, and survives clone (13-F20)."""
    assert BorutaShap().get_params()["stability_n_jobs"] == -1
    assert clone(BorutaShap(stability_n_jobs=1)).stability_n_jobs == 1
