"""Gate fallbacks of the tail-dependence and synergy-combo device paths warn (throttled) and a mismatched target length is refused before any launch; all driven on the CPU."""

import logging

import numpy as np

from mlframe.feature_selection.filters import _synergy_combo_gpu as sg
from mlframe.feature_selection.filters import _tail_dependence_gpu as tg
from mlframe.utils.log_throttle import reset_throttle_counts


class _BadSpec:
    """A tuner spec whose choice always raises."""

    def choose(self, **kwargs):
        """Always fail."""
        raise RuntimeError("cache broken")


def test_tail_use_device_warns_when_the_tuner_choice_fails(monkeypatch, caplog):
    """A failing KTC choose logs a WARNING and the rows floor decides."""
    reset_throttle_counts()
    monkeypatch.setattr(tg, "_SPEC", _BadSpec())
    monkeypatch.setattr("mlframe.feature_selection.filters._gpu_policy.gpu_globally_disabled", lambda: False)
    with caplog.at_level(logging.WARNING):
        got = tg.use_device(tg.DEVICE_FALLBACK_ROWS + 1)
    assert got is True
    assert any("KTC choose() failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_synergy_device_chosen_warns_when_the_tuner_choice_fails(monkeypatch, caplog):
    """A failing KTC choose logs a WARNING and the rows floor decides."""
    reset_throttle_counts()
    monkeypatch.setattr(sg, "_SPEC", _BadSpec())
    with caplog.at_level(logging.WARNING):
        got = sg.synergy_device_chosen(sg.DEVICE_FALLBACK_ROWS, 1)
    assert got is True
    assert any("KTC choose() failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_tail_device_refuses_a_target_of_another_length(caplog):
    """A target shorter than the columns returns None with a WARNING instead of reading past the flag array on the device."""
    reset_throttle_counts()
    cols = np.zeros((2, 100), dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        assert tg.tail_dependence_columns_device(cols, np.zeros(50), 0.95) is None
    assert any("target has 50 rows" in r.getMessage() for r in caplog.records)
