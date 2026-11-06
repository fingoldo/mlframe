"""``ShapProxiedFS`` keeps a ``None`` width as ``None`` through ``get_params`` and ``clone``; the per-host default is resolved at fit.

The constructor used to store the per-host default in place of ``None``. That value then appeared in ``get_params``, was frozen into clones and
pickles, and ``set_params`` could not bring the default back (audit API3-1, wave 3).
"""

from __future__ import annotations

from sklearn.base import clone

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_init_params import (
    effective_brute_force_max_features,
    effective_cluster_su_auto_max_features,
)


def test_none_widths_round_trip_through_get_params_and_clone():
    """``None`` stays ``None`` in get_params and in a clone."""
    est = ShapProxiedFS()
    assert est.get_params()["brute_force_max_features"] is None
    assert est.get_params()["cluster_su_auto_max_features"] is None
    cloned = clone(est)
    assert cloned.get_params()["brute_force_max_features"] is None
    assert cloned.get_params()["cluster_su_auto_max_features"] is None


def test_none_widths_resolve_to_the_per_host_default_at_fit():
    """The effective helpers return an integer default when the parameter is ``None``, and the given value otherwise."""
    est = ShapProxiedFS()
    assert isinstance(effective_brute_force_max_features(est), int)
    assert isinstance(effective_cluster_su_auto_max_features(est), int)
    est.brute_force_max_features = 7
    assert effective_brute_force_max_features(est) == 7
