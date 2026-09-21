"""Bootstrap MI-gain replicates of ``MI(y, X)`` must be computed once per base and mask, not once per transform.

The bootstrap generator is re-seeded per candidate from a fixed seed, so every transform on a base with the same valid-row
mask draws the same replicates and recomputed the same ``MI(y, X)`` values (plus a row-major copy of the prebinned block).
They are now shared through the base context; each candidate's LCB and bootstrap p-value are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.discovery import _eval
from mlframe.training.configs import CompositeTargetDiscoveryConfig

_FEATS = ["base", "x1", "x2", "x3"]


def _frame(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """A frame where several transforms on one base all keep every row (same mask)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(10.0, 50.0, n)
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"base": base, "x1": x1, "x2": x2, "x3": x3, "y": 1.5 * base + 2.0 * x1 + rng.normal(size=n)})


def _fit(df, monkeypatch) -> dict:
    """A discovery fit with the opt-in bootstrap on; returns each candidate's (LCB, bootstrap p-value)."""
    import mlframe.training.composite.discovery._fit as fit_mod

    captured: dict = {}
    real_eval = fit_mod.eval_one_transform

    def capturing(*args, **kwargs):
        """Record every candidate's bootstrap outputs as the evaluator returns them."""
        out = real_eval(*args, **kwargs)
        for cand in out or []:
            captured[cand["spec"].name] = (cand.get("mi_gain_lcb"), cand.get("bootstrap_p_value"))
        return out

    monkeypatch.setattr(fit_mod, "eval_one_transform", capturing)
    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, base_candidates=["base"], screening="mi",
        transforms=["diff", "linear_residual", "additive_residual", "monotonic_residual"], mi_gain_bootstrap_n=20,
    ))
    disc.fit(df, "y", _FEATS, np.arange(len(df)))
    return captured


def test_lcb_and_p_values_match_a_run_without_sharing(monkeypatch):
    """Sharing is a speed change only: every candidate's LCB and p-value equal those of an unshared run."""
    df = _frame()
    shared = _fit(df, monkeypatch)
    assert shared, "the bootstrap must have produced LCBs on this fixture"
    real = _eval._bootstrap_gain_replicates

    def unshared(*args, mi_y_reps=None, **kwargs):
        """Ignore any memoised replicates, recomputing MI(y, X) every time."""
        return real(*args, mi_y_reps=None, **kwargs)

    monkeypatch.setattr(_eval, "_bootstrap_gain_replicates", unshared)
    assert _fit(df, monkeypatch) == shared


def test_mi_y_replicates_are_computed_once_per_base(monkeypatch):
    """With several same-mask transforms on one base, the MI(y, X) replicate pass runs once, even though the
    candidates are evaluated concurrently."""
    passes = {"n": 0}
    real = _eval._bootstrap_mi_y_replicates

    def counting(*args, **kwargs):
        """Count every MI(y, X) replicate pass."""
        passes["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(_eval, "_bootstrap_mi_y_replicates", counting)
    stats = _fit(_frame(), monkeypatch)
    assert len(stats) >= 2, "the check needs at least two bootstrapped candidates on the base"
    assert passes["n"] == 1, f"the MI(y, X) replicates were computed {passes['n']} times for {len(stats)} candidates"


def test_a_replayed_failure_is_recorded_like_the_original():
    """A replicate whose MI(y, X) failed for the first candidate fails, with the same message, for the next."""
    rng = np.random.default_rng(0)
    n = 200
    t = rng.normal(size=n)
    y = rng.normal(size=n)
    x_pb = rng.integers(0, 8, size=(n, 3)).astype(np.int16)
    reps = (np.zeros(5), {2: "ValueError: boom"})
    boot = np.empty(5)
    failures: list = []
    count, _ = _eval._bootstrap_gain_replicates(
        boot, 5, np.random.default_rng(1), n, t, y, x_pb, None, dict(nbins=8, aggregation="mean"), None, failures, mi_y_reps=reps,
    )
    assert count == 1 and failures == ["replicate 2: ValueError: boom"]
    assert np.isnan(boot[2]) and np.isfinite(np.delete(boot, 2)).all()
