"""An auto-discovered chain has to beat raw y, and not duplicate a chain the registry already has (DSC-20).

``discover_chains`` surfaced a chain whenever it beat both of its single stages, even when all three lost to the raw-y
baseline it had already computed; the proposal is appended after the rerank, so it bypassed the raw-baseline and
honest-OOF floors and cost a full model-zoo fit. With six chains picked on the same folds and a zero margin, the winner
was as often the luckiest as the best. And ``linear_residual`` + ``cbrt`` is the composition the registry ships as
``chain_linres_cbrt``: proposed under the auto-generated name, nothing downstream could tell the two apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._auto_chain import _registry_equivalent, discover_chains

pytest.importorskip("lightgbm")


def _raw_is_best(n: int = 600, seed: int = 0):
    """A DGP with no base structure to remove: y is pure feature signal, so any residual or tail transform only hurts."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    base = rng.normal(100.0, 1.0, n)  # unrelated to y
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + rng.normal(scale=0.1, size=n)
    return y, base, x


def _chain_helps(n: int = 600, seed: int = 0):
    """A DGP the chain is for: y is base plus a heavy-tailed residual that the cube root compresses."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    base = rng.uniform(10.0, 100.0, n)
    resid = (2.0 * x[:, 0]) ** 3 + rng.standard_t(df=3, size=n)
    return base + resid, base, x


def _chains(y, base, x, **kwargs):
    """Run the chain search with the small settings the discovery phase uses."""
    return discover_chains(y=y, base=base, x_matrix=x, cv_folds=3, n_estimators=25, compute_mi_gain=False,
                           random_state=0, **kwargs)


def _rmse_table(monkeypatch, raw: float, single: float, chain: float):
    """Make the chain search see fixed numbers: ``raw`` for the baseline, ``single`` per stage, ``chain`` per composition."""
    from mlframe.training.composite.discovery import _auto_chain as mod

    def fake(transform, **_kw):
        """The y-scale CV RMSE of a stage or chain, by what it is rather than by fitting anything."""
        if transform is None:
            return raw, 1.0
        return (chain if getattr(transform, "chain_stages", None) else single), 1.0

    monkeypatch.setattr(mod, "_y_scale_cv_rmse", fake)


def test_a_chain_that_beats_both_singles_but_loses_to_raw_is_not_proposed(monkeypatch):
    """The gate that was missing: the chain wins its comparison with the two stages and still loses to the raw target."""
    y, base, x = _raw_is_best()
    _rmse_table(monkeypatch, raw=1.0, single=3.0, chain=2.0)
    assert _chains(y, base, x) == []


def test_a_chain_that_beats_raw_and_both_singles_is_proposed(monkeypatch):
    """The control: with the same machinery, a chain that beats everything by a clear margin still surfaces."""
    y, base, x = _raw_is_best()
    _rmse_table(monkeypatch, raw=3.0, single=3.0, chain=1.0)
    assert _chains(y, base, x)


def test_a_chain_that_only_ties_the_raw_baseline_is_not_proposed(monkeypatch):
    """A hair's difference over several chains picked on the same folds is the winner's curse, not a result."""
    y, base, x = _raw_is_best()
    _rmse_table(monkeypatch, raw=2.0, single=3.0, chain=2.0 * (1.0 - 1e-6))
    assert _chains(y, base, x) == []


def test_nothing_surfaces_when_the_raw_target_is_the_best_target():
    """The same verdict without the stand-in numbers: a DGP whose base carries no signal proposes no chain."""
    y, base, x = _raw_is_best()
    assert _chains(y, base, x) == []


def test_the_registry_equivalent_is_found_by_composition_not_by_name():
    """``linear_residual`` + ``cbrt`` is ``chain_linres_cbrt``; an unrelated pair has no equivalent."""
    assert _registry_equivalent("linear_residual", "cbrt") == "chain_linres_cbrt"
    assert _registry_equivalent("monotonic_residual", "yj") == "chain_monres_yj"
    assert _registry_equivalent("linear_residual", "asinh") is None


def test_a_composition_the_screen_already_carries_is_not_proposed_again():
    """With ``chain_linres_cbrt`` in the transform pool, the same composition is not proposed under its auto name."""
    y, base, x = _chain_helps()
    proposed = {c.chain_name for c in _chains(y, base, x, already_screened=["chain_linres_cbrt"])}
    assert "chain_linear_residual_cbrt" not in proposed, proposed


def test_the_same_composition_is_still_available_when_the_pool_does_not_carry_it():
    """The dedup is about the screen's own pool: without that entry, the search may still propose the composition."""
    y, base, x = _chain_helps()
    names = {c.chain_name for c in _chains(y, base, x, already_screened=[])}
    for name in names:
        assert name.startswith("chain_")
