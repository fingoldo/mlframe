"""Cluster 41 regression: the envelope-clip docstring matches the code (audit 2026-11-20, finding 41-F28).

The docstring used to promise an eval-derived clip (k_sigma=10) when train statistics are absent. The code does no clip in that case, on purpose,
because the scored split's own labels would bound its predictions. The docstring now says so.
"""

import inspect

from mlframe.training.reporting._reporting_regression import _sensors as sensors


def _fn():
    """The envelope-clip function whose docstring is checked."""
    for name, obj in vars(sensors).items():
        if name.startswith("apply_prediction_envelope_clip") and callable(obj):
            return obj
    raise AssertionError("envelope clip function not found")


def test_docstring_no_longer_promises_an_eval_derived_clip():
    """The stale k_sigma=10 eval-fallback promise is gone (41-F28)."""
    doc = inspect.getdoc(_fn())
    assert "k_sigma=10" not in doc
    assert "eval-derived bound" not in doc


def test_docstring_states_no_clip_without_train_stats():
    """The docstring says predictions are not clipped when train stats are absent (41-F28)."""
    doc = inspect.getdoc(_fn())
    assert "NOT clipped" in doc
