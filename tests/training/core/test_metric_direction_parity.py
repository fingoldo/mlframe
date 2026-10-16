"""The metric-direction helper agrees with the metrics registry on every known name; the default-metric check matches the AUC default."""

from __future__ import annotations

from mlframe.training import metrics_registry as R
from mlframe.training.core._phase_finalize import _is_minimise_metric
from mlframe.models.ensembling.selection import _default_metric_is_auc


def test_direction_matches_registry_for_every_known_higher_metric() -> None:
    """A metric the registry says is higher-is-better is not treated as minimise."""
    for name in sorted(R._KNOWN_METRIC_DIRECTIONS_HIGHER):
        assert _is_minimise_metric(name) is False, name


def test_direction_matches_registry_for_every_known_lower_metric() -> None:
    """A metric the registry says is lower-is-better is treated as minimise."""
    for name in sorted(R._KNOWN_METRIC_DIRECTIONS_LOWER):
        assert _is_minimise_metric(name) is True, name


def test_unknown_metric_defaults_to_minimise() -> None:
    """A name the registry does not know falls back to the RMSE-like lower-is-better default."""
    assert _is_minimise_metric("not_a_registered_metric_xyz") is True


def test_default_metric_check_is_the_unset_metric() -> None:
    """Only an unset metric selects the ROC-AUC default."""
    assert _default_metric_is_auc(None) is True
    assert _default_metric_is_auc(lambda y, p: 0.0) is False
