"""Cluster 04 regression: tail_gate_term builds only the quantile masks its direction uses (audit 2026-11-20, finding 04-F29)."""

import numpy as np
import pytest

from mlframe.data.datasets import _links


def _count_quantile_calls(monkeypatch, direction):
    """Run tail_gate_term under a counting np.quantile and return the call count."""
    real = np.quantile
    calls = {"n": 0}

    def _counting(*a, **k):
        """Count and delegate to the real quantile."""
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(_links.np, "quantile", _counting)
    rng = np.random.default_rng(0)
    ops = [rng.normal(size=200), rng.normal(size=200)]
    _links.tail_gate_term(ops, 0.1, direction=direction)
    return calls["n"]


@pytest.mark.parametrize("direction", ["upper", "lower"])
def test_single_direction_computes_one_quantile_per_operand(monkeypatch, direction):
    """A one-sided gate needs one quantile per operand, not two (04-F29)."""
    assert _count_quantile_calls(monkeypatch, direction) == 2


def test_both_directions_compute_two_quantiles_per_operand(monkeypatch):
    """The two-sided gate still needs both quantiles per operand (04-F29)."""
    assert _count_quantile_calls(monkeypatch, "both") == 4


def test_unknown_direction_is_refused():
    """An unsupported direction raises before any work (04-F29)."""
    with pytest.raises(ValueError, match="unsupported tail gate direction"):
        _links.tail_gate_term([np.arange(10.0)], 0.1, direction="sideways")


def test_upper_gate_values_unchanged():
    """Control: the upper gate fires exactly where every operand is in its upper tail (04-F29)."""
    a = np.arange(100.0)
    fires = _links.tail_gate_term([a, a], 0.1, direction="upper")
    assert fires.sum() == 90  # rows >= the 0.1 quantile (9.9) of 0..99
    assert fires[10] == 1.0 and fires[9] == 0.0
