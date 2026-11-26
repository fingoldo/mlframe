"""A predict callable that rejects the keyword params is reported at WARNING, not silently declined."""

import logging

from mlframe.training.targets._ttr_eval_set_scaling import _predict_accepts


def test_rejected_keyword_params_return_false_and_warn(caplog):
    """A callable that takes no such keyword is refused, and the refusal is logged with the params named."""

    def predict(x):
        """Take one positional argument and no keywords."""
        return x

    with caplog.at_level(logging.WARNING):
        assert _predict_accepts(predict, {"bogus": 1}) is False
    assert any(r.levelno == logging.WARNING and "bogus" in r.getMessage() for r in caplog.records)


def test_accepted_keyword_params_return_true_silently(caplog):
    """A callable that takes the keyword is accepted and nothing is logged."""

    def predict(x, scale=1.0):
        """Take a positional argument and an optional scale keyword."""
        return x * scale

    with caplog.at_level(logging.WARNING):
        assert _predict_accepts(predict, {"scale": 2.0}) is True
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
