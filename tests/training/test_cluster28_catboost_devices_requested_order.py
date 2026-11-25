"""Cluster 28 regression: parse_catboost_devices returns GPUs in the requested order, each once (audit 2026-11-20, finding 28-F67)."""

from mlframe.training.helpers import parse_catboost_devices

GPUS = [{"index": 0, "name": "g0"}, {"index": 1, "name": "g1"}, {"index": 2, "name": "g2"}]


def test_requested_order_is_preserved():
    """'1:0' returns [gpu1, gpu0], not device order (28-F67)."""
    assert [g["index"] for g in parse_catboost_devices("1:0", GPUS)] == [1, 0]


def test_duplicate_requests_return_each_gpu_once():
    """'0:0' returns a single gpu0 (28-F67)."""
    assert [g["index"] for g in parse_catboost_devices("0:0", GPUS)] == [0]


def test_range_still_returns_ascending_devices():
    """Control: a range keeps ascending order (28-F67)."""
    assert [g["index"] for g in parse_catboost_devices("0-2", GPUS)] == [0, 1, 2]
