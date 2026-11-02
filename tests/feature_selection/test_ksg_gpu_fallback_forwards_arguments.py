"""mixed_ksg_mi_gpu without cupy forwards every argument to the CPU estimator (tracker 2026-11-02 wave 13, PAR13B-3)."""

import sys

import numpy as np

from mlframe.feature_selection.filters import _ksg


def test_fallback_passes_intens_max_input_n_and_seed(monkeypatch):
    """seed=7 used to become seed=0 when cupy was missing."""
    seen = {}
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(_ksg, "mixed_ksg_mi", lambda x, y, **kw: seen.update(kw) or 0.0)
    _ksg.mixed_ksg_mi_gpu(np.zeros(10), np.zeros(10), k=3, intens=1e-8, max_input_n=123, seed=7)
    assert seen == {"k": 3, "intens": 1e-8, "max_input_n": 123, "seed": 7}
