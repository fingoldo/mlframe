"""Audit 2026-11-20 finding 36-F9: a class factory's sample_weight support is read from its fit signature, without building it."""

from mlframe.training.composite import stacking_multi_stage as sms


class _NoConstruct:
    """A class whose constructor fails: the signature check must not construct it."""

    def __init__(self):
        raise AssertionError("constructed")

    def fit(self, X, y, sample_weight=None):
        """Accepts sample weights."""
        return self


class _NoWeights:
    """A class whose fit takes no sample_weight."""

    def fit(self, X, y):
        """No weights."""
        return self


def test_class_factory_is_inspected_without_construction():
    """A class with sample_weight in fit reports True, and the constructor never runs."""
    assert sms._factory_accepts_sample_weight(_NoConstruct) is True


def test_class_factory_without_weights_reports_false():
    """A class whose fit has no sample_weight reports False."""
    assert sms._factory_accepts_sample_weight(_NoWeights) is False


def test_callable_factory_still_asks_the_built_estimator():
    """A plain lambda is called to inspect the estimator it builds."""
    assert sms._factory_accepts_sample_weight(lambda: _NoConstruct.__new__(_NoConstruct)) is True
