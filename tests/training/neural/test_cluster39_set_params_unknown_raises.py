"""Cluster 39 regression: set_params refuses unknown and method names instead of ignoring them (audit 2026-11-20, finding 39-F55)."""

import pytest

from mlframe.training.neural._base_sklearn_params import set_params


class _Est:
    """Minimal estimator carrying one settable attribute and one method."""

    def __init__(self):
        """Set the one real parameter."""
        self.learning_rate = 0.1

    def fit(self):
        """A method name that set_params must not overwrite."""
        return self


def test_unknown_key_raises_value_error():
    """A misspelled parameter raises so a grid search cannot silently run identical configs (39-F55)."""
    with pytest.raises(ValueError, match="lerning_rate"):
        set_params(_Est(), lerning_rate=0.5)


def test_method_name_cannot_be_overwritten():
    """A key that names a method is refused, not assigned over the method (39-F55)."""
    with pytest.raises(ValueError, match="fit"):
        set_params(_Est(), fit=1)


def test_known_key_is_still_set():
    """Control: a real attribute is updated (39-F55)."""
    est = set_params(_Est(), learning_rate=0.01)
    assert est.learning_rate == 0.01
