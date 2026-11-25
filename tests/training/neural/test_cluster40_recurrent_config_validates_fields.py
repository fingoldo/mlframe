"""Cluster 40 regression: RecurrentConfig refuses bad field values at construction (audit 2026-11-20, finding 40-F29)."""

import pytest

from mlframe.training.neural._recurrent_config import RecurrentConfig


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"sequence_preprocessing": "bogus"}, "sequence_preprocessing"),
        ({"ema_decay": 0.0}, "ema_decay"),
        ({"ema_decay": 1.5}, "ema_decay"),
        ({"num_classes": 0}, "num_classes"),
        ({"mixup_alpha": -0.1}, "mixup_alpha"),
        ({"mixup_alpha": float("nan")}, "mixup_alpha"),
    ],
)
def test_bad_value_is_refused_at_construction(kwargs, match):
    """Each field fails at construction, not later inside fit (40-F29)."""
    with pytest.raises(ValueError, match=match):
        RecurrentConfig(**kwargs)


def test_default_config_still_constructs():
    """Control: the defaults remain valid."""
    cfg = RecurrentConfig()
    assert cfg.sequence_preprocessing == "none"
