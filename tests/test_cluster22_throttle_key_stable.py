"""Cluster 22 regression: the throttle key of a failed calibration-target coercion is a stable name, not a line number (audit 2026-11-20, finding 43).

A key built from the source line changes whenever the file is edited above it, which resets the once-per-key throttle and repeats the warning.
"""

import logging

from mlframe.calibration import _post_train_calibrators as mod


class _BrokenTarget:
    """A test target whose ``values`` attribute raises, as a malformed frame would."""

    @property
    def values(self):
        """Fail the coercion."""
        raise ValueError("cannot coerce")


class _Model:
    """A model carrying only the broken test target."""

    test_target = _BrokenTarget()


def test_a_failed_target_coercion_is_throttled_under_a_fixed_name(monkeypatch):
    """The warning goes through log_throttle under the fixed key ``post_train_calib_target_coerce_failed`` at WARNING (22-F43)."""
    seen = []

    def spy(logger, key, level, *args, **kwargs):
        """Record how the warning is throttled."""
        seen.append((key, level))

    monkeypatch.setattr(mod, "log_throttle", spy)
    mod._train_one_model_postcalibrators({"m": _Model()}, [0, 1], None, 0.9, 0.5, {})
    assert seen == [("post_train_calib_target_coerce_failed", logging.WARNING)]
