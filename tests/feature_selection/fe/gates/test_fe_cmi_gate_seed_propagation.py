"""Regression: the FE conditional-MI redundancy gate must seed its permutation null from
the user's ``random_seed`` (the canonical param), not from ``random_state`` (which defaults
to ``None`` -> ``0``). Pre-fix, ``materialise_and_finalise_fe_candidates`` passed
``seed=int(getattr(self, "random_state", 0) or 0)``: a user setting only ``random_seed`` got
the gate seeded 0 on every fit, and two MRMRs with different ``random_seed`` shared an
identical gate null. The existing CMI-gate suite calls ``apply_cmi_redundancy_gate`` directly
with an explicit ``seed=`` so the wiring bug was invisible; this test drives the seed THROUGH
``MRMR.fit``.
"""

from __future__ import annotations

import numpy as np

import mlframe.feature_selection.filters._fe_cmi_redundancy_gate as _gate_mod
from mlframe.feature_selection.filters.mrmr import MRMR


def _fe_world(seed: int, n: int = 4000):
    """Three genuine engineered drivers (a**2/b, log(c)*sin(d), g*h) + noise so the FE pair
    search surfaces >=2 engineered survivors and the conditional-MI gate actually fires."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 3.0, n)
    b = rng.uniform(0.5, 3.0, n)
    c = rng.uniform(0.5, 5.0, n)
    d = rng.uniform(0.0, 2 * np.pi, n)
    g = rng.uniform(0.5, 3.0, n)
    h = rng.uniform(0.5, 3.0, n)
    f = rng.normal(0.0, 1.0, n)
    y = a**2 / b + 3.0 * np.log(c) * np.sin(d) + g * h + f / 5.0
    import pandas as pd

    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "g": g, "h": h, "f": f})
    return X, y


def _capture_gate_seeds(monkeypatch):
    """Patch the gate (imported in-body by the FE step) to record every ``seed`` it receives,
    delegating to the real implementation so the fit proceeds normally."""
    seen: list[int] = []
    real = _gate_mod.apply_cmi_redundancy_gate

    def _spy(*args, **kwargs):
        """Helper that spy."""
        seen.append(int(kwargs.get("seed", -1)))
        return real(*args, **kwargs)

    monkeypatch.setattr(_gate_mod, "apply_cmi_redundancy_gate", _spy)
    return seen


def test_fe_cmi_gate_seeded_from_random_seed_not_zero(monkeypatch):
    """``MRMR(random_seed=K)`` (no ``random_state``) must seed the FE CMI gate with K, not 0."""
    seen = _capture_gate_seeds(monkeypatch)
    X, y = _fe_world(seed=0)
    MRMR(random_seed=12345, max_runtime_mins=2.0).fit(X, y)
    assert seen, "FE conditional-MI gate was never reached -- fixture failed to engineer >=2 candidates"
    assert all(s == 12345 for s in seen), f"gate seeded {seen}, expected all 12345 (random_seed dropped to 0?)"


def test_fe_cmi_gate_seed_falls_back_to_random_state(monkeypatch):
    """When only the sklearn-style ``random_state`` is set, the gate uses it (alias fallback)."""
    seen = _capture_gate_seeds(monkeypatch)
    X, y = _fe_world(seed=0)
    MRMR(random_state=777, max_runtime_mins=2.0).fit(X, y)
    assert seen, "FE conditional-MI gate was never reached"
    assert all(s == 777 for s in seen), f"gate seeded {seen}, expected all 777"
