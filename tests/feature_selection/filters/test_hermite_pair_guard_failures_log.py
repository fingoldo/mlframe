"""A failing noise-floor guard or trivial-baseline check in ``optimise_hermite_pair`` is logged at WARNING, not swallowed at debug."""

from __future__ import annotations

import logging

import numpy as np

from mlframe.feature_selection.filters import _hermite_fe_optimise_pair as P


def _xor_data(n: int = 1500):
    """Two operands and a binary target that depends on their product only."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    return a, b, (a * b > 0).astype(np.int64)


def _run(**kw):
    """Run the optimiser on the XOR fixture with the cheap random_batch search."""
    a, b, y = _xor_data()
    return P.optimise_hermite_pair(a, b, y, optimizer="random_batch", n_trials=80, max_degree=3, use_trivial_baseline=False, warm_start_als=True, **kw)


def test_noise_floor_guard_failure_is_logged(monkeypatch, caplog):
    """If the permutation guard raises, a WARNING says the pair was admitted without the null check."""
    from mlframe.feature_selection.filters.hermite_fe import shared

    def boom(*a, **k):
        """Stand-in plug-in MI that faults."""
        raise RuntimeError("njit fault")

    monkeypatch.setattr(shared, "plugin_mi_from_binned_njit", boom)
    with caplog.at_level(logging.WARNING, logger=P.logger.name):
        res = _run()
    assert res is not None
    assert any("noise-floor permutation guard failed" in r.getMessage() for r in caplog.records)


def test_trivial_baseline_failure_is_logged(monkeypatch, caplog):
    """If the trivial-baseline probe raises, a WARNING says the uplift gate used the identity baseline only."""
    from mlframe.feature_selection.filters import fe_baselines

    def boom(*a, **k):
        """Stand-in trivial-baseline probe that faults."""
        raise RuntimeError("probe fault")

    monkeypatch.setattr(fe_baselines, "best_trivial_pair", boom)
    a, b, y = _xor_data()
    with caplog.at_level(logging.WARNING, logger=P.logger.name):
        P.optimise_hermite_pair(a, b, y, optimizer="random_batch", n_trials=20, max_degree=2, use_trivial_baseline=True)
    assert any("trivial baseline check failed" in r.getMessage() for r in caplog.records)
