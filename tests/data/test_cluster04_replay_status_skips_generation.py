"""Audit 2026-11-20 finding 04-F24: a status-only replay compares the spec hash and does not generate the bed.

replay_report prints one line per scenario, so generating every scenario's dataset to read its status was wasted work
that grew with the results file. The check is that generate() is not called when with_data is False, and that the
status is the same as the full replay's.
"""

from mlframe.data.datasets import generator, scenarios
from mlframe.data.datasets.replay import replay, replay_report


def _any_scenario():
    """The first registered scenario name, so the test does not depend on a particular bed."""
    return sorted(scenarios.names())[0]


def test_status_only_replay_does_not_generate(monkeypatch):
    """With with_data=False, generate is never called and dataset is None, while the status still reports the hash match."""
    calls = []
    real = generator.generate

    def counting(spec):
        """Record one generate() call, then delegate to the real generator."""
        calls.append(1)
        return real(spec)

    monkeypatch.setattr(generator, "generate", counting)
    name = _any_scenario()
    full = replay(name, 0, expected_hash=None)
    assert calls, "the full replay generates the bed"
    calls.clear()
    light = replay(name, 0, expected_hash=full.actual_hash, with_data=False)
    assert calls == []
    assert light.dataset is None
    assert light.status == "exact"


def test_replay_report_does_not_generate_any_bed(monkeypatch):
    """replay_report over a results file generates nothing and still names every scenario it saw."""
    calls = []
    monkeypatch.setattr(generator, "generate", lambda spec: calls.append(1))
    name = _any_scenario()
    lines = replay_report([{"scenario": name, "dataset_seed": 0, "spec_hash": None}])
    assert calls == []
    assert any(line.startswith(name) for line in lines)
