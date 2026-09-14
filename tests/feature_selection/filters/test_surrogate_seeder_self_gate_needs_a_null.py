"""The surrogate seeder's self-gate must not pass without a usable null (mrmr_audit_2026-09-14 NUM-4).

When every permuted-y run failed, the seeder set ``perm_std = 0.0`` and computed
``z = (oof_real - perm_mean) / (perm_std + 1e-9)``. Any positive gap then became z ~ 1e9, so the PAIR
self-gate passed unconditionally -- the opposite of the "nominal spread so the z-gate still applies" its
comment promised. A degenerate null (every permuted run scoring identically) hit the same division.
Absent a usable null the z-statistic is undefined, and undefined must not emit pair seeds.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _surrogate_interaction_seeder as S


class _FakeBooster:
    """Stands in for a fitted LightGBM booster; the post-gate co-occurrence walk only calls ``dump_model``."""

    def dump_model(self):
        """No trees, so no split co-occurrence is tallied -- the test isolates the gate, not the ranking."""
        return {"tree_info": []}


def _run_with_perm_scores(monkeypatch, perm_score_fn, real_score=0.95):
    """Drive ``surrogate_gbm_interaction_seeds`` with controlled real and permuted OOF scores."""
    calls = {"perm": 0}

    def _fake_fit(disc_X, y, *, shuffle_y, **kwargs):
        """Real runs succeed with a high OOF score; permuted runs return whatever the case dictates."""
        if not shuffle_y:
            return _FakeBooster(), real_score
        calls["perm"] += 1
        return None, perm_score_fn(calls["perm"])

    monkeypatch.setattr(S, "_fit_surrogate_and_oof", _fake_fit)
    rng = np.random.default_rng(0)
    disc_X = rng.integers(0, 5, size=(200, 3))
    y = rng.integers(0, 2, size=200)
    return S.surrogate_gbm_interaction_seeds(disc_X, y, [0, 1, 2], is_classification=True, self_gate_reps=5)


def test_all_permuted_runs_failing_does_not_pass_the_pair_gate(monkeypatch):
    """No null at all: z is undefined, so the gate must fail and emit no pair seeds."""
    pairs, _triples, info = _run_with_perm_scores(monkeypatch, lambda i: float("nan"))
    assert info["self_gate_null_available"] is False
    assert info["self_gate_z"] == float("-inf"), f"no null must not yield a finite z, got {info['self_gate_z']!r}"
    assert info["gated"] is False
    assert pairs == []


def test_a_degenerate_null_does_not_pass_the_pair_gate(monkeypatch):
    """Every permuted run scoring identically gives std == 0; z is undefined there too, not huge."""
    pairs, _triples, info = _run_with_perm_scores(monkeypatch, lambda i: 0.5)
    assert info["self_gate_null_available"] is True
    assert info["self_gate_z"] == float("-inf")
    assert info["gated"] is False
    assert pairs == []


def test_a_healthy_null_still_passes_the_gate_on_real_signal(monkeypatch):
    """Control: a genuine spread well below the real score must keep the gate working normally."""
    spread = [0.50, 0.52, 0.48, 0.51, 0.49]
    _pairs, _triples, info = _run_with_perm_scores(monkeypatch, lambda i: spread[(i - 1) % len(spread)])
    assert info["self_gate_null_available"] is True
    assert np.isfinite(info["self_gate_z"]) and info["self_gate_z"] > 2.0
    assert info["gated"] is True


def test_a_healthy_null_rejects_a_real_score_inside_the_noise(monkeypatch):
    """Control: with a usable null, a real score that does not clear it must still be rejected."""
    spread = [0.50, 0.52, 0.48, 0.51, 0.49]
    _pairs, _triples, info = _run_with_perm_scores(monkeypatch, lambda i: spread[(i - 1) % len(spread)], real_score=0.505)
    assert np.isfinite(info["self_gate_z"])
    assert info["gated"] is False


@pytest.mark.parametrize("perm_fn", [lambda i: float("nan"), lambda i: 0.5], ids=["no-null", "degenerate-null"])
def test_the_failure_is_warned(monkeypatch, caplog, perm_fn):
    """An undefined z silently disabling pair seeding must be visible in the log."""
    import logging

    with caplog.at_level(logging.WARNING):
        _run_with_perm_scores(monkeypatch, perm_fn)
    assert any("self-gate z is undefined" in r.message for r in caplog.records if r.levelno >= logging.WARNING)
