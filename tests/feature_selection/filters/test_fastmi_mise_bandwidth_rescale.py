"""fastmi: the MISE bandwidth searched on a 1000-row sub-sample is rescaled by (1000/n)^(1/6) for the full n."""
from __future__ import annotations

import math

import numpy as np

from mlframe.feature_selection.filters._fastmi import fastmi


def _rel_err(rescale: bool) -> float:
    """Mean relative error vs the closed-form Gaussian MI at n=20000 over rho in (0.6, 0.9) and 2 seeds."""
    errs = []
    for rho in (0.6, 0.9):
        true = -0.5 * math.log(1.0 - rho * rho)
        for s in range(2):
            r = np.random.default_rng(100 * s + 7)
            x = r.standard_normal(20000)
            y = rho * x + math.sqrt(1 - rho * rho) * r.standard_normal(20000)
            errs.append(abs(fastmi(x, y, random_seed=s, mise_rescale=rescale) - true) / true)
    return float(np.mean(errs))


def test_rescaled_bandwidth_cuts_bias_at_large_n():
    """Measured relative error: rescaled 0.062, legacy 0.158; the rescaled error must stay under 0.09 and below 0.7x the legacy."""
    new = _rel_err(True)
    old = _rel_err(False)
    assert new < 0.09, f"rescaled rel err {new:.4f} (measured 0.062)"
    assert new < 0.7 * old, f"rescaled {new:.4f} not clearly below legacy {old:.4f}"
