"""Regression: engineered-feature producers must not silently re-admit a group-aware-demoted leak.

``group_aware_mi=True`` already demotes a between-group-level "leak" raw feature from the RAW-feature
relevance screen (see ``test_group_aware_mi_mrmr.py``). But every engineered-feature producer (the FE
pair-search, the hybrid-orth Hermite basis, ...) scores its candidates with the PLAIN global plug-in MI
-- none of them consult ``get_group_mi()`` -- so a composite built FROM the demoted leak raw feature and
an unrelated noise operand still carries the leak's between-group signal and clears every naive-MI
acceptance gate, re-entering the final selection through the back door.

The fix is a universal group-aware demotion pass in ``_fit_impl_core.py`` (right before ``support_``/
``get_feature_names_out`` are finalised): it re-checks every surviving engineered column's OWN
group-blocked relevance and drops any whose within-group MI comes back exactly zero -- covering every
producer at one choke point instead of threading group-awareness through each one's own hot
per-candidate scoring loop. (``_mrmr_fe_step/_step_score.py`` also runs an earlier, narrower instance of
the same check right after the plain pair-search, as defense-in-depth.)

A real between-group-leak-via-interaction fixture is hard to construct deterministically (an engineered
composite naturally inherits real within-group signal from whichever genuine predictor FE happened to
pair the leak with, so it is rarely EXACTLY zero in practice) -- this test instead pins the WIRING
directly: monkeypatch ``group_relevance_mi`` to report zero within-group signal and confirm the fit
actually strips every engineered survivor as a result, then confirms the control fit (unpatched) keeps
its engineered features normally.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _nonlinear_panel(seed: int, G: int = 60, per: int = 50):
    """Panel with a genuine product-interaction target (guarantees FE finds and keeps at least one
    engineered composite) plus a group structure so ``group_aware_mi=True`` publishes a real
    ``get_group_mi()`` payload for the demotion pass to consult."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(G), per)
    n = groups.size
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = a * b + 0.05 * rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": b, "c": rng.normal(size=n)})
    return X, y, groups


def _fit(X, y, groups, **kw):
    """Fit an MRMR with warnings suppressed, passing groups through and FE enabled.

    ``fit_cache_max=0``: this test fits the SAME (X, y, groups, params) twice with only a global
    monkeypatch differing between calls -- the process-wide ``_FIT_CACHE`` keys on content+params (it
    cannot see an unrelated monkeypatch), so without disabling it the second fit would silently replay
    the first call's cached result instead of actually re-running with the patch active."""
    kw.setdefault("fe_max_steps", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(max_runtime_mins=2, verbose=0, fit_cache_max=0, **kw)
        m.fit(X, y, groups=groups)
    return m


def test_group_aware_demotion_strips_engineered_features_with_zero_within_group_mi(monkeypatch):
    """Pins the demotion mechanism directly: with ``group_relevance_mi`` forced to always report zero
    within-group signal, every surviving engineered feature must be stripped from the final selection."""
    X, y, groups = _nonlinear_panel(1)

    control = _fit(X, y, groups, group_aware_mi=True)
    control_recipes = list(control._engineered_recipes_ or [])
    assert control_recipes, "fixture must produce at least one engineered survivor for this test to be meaningful"

    monkeypatch.setattr(
        "mlframe.feature_selection.filters.info_theory._group_mi.group_relevance_mi",
        lambda *a, **kw: 0.0,
    )
    demoted = _fit(X, y, groups, group_aware_mi=True)
    assert not (demoted._engineered_recipes_ or []), (
        f"engineered recipes should all be stripped once every group-blocked relevance check reports "
        f"zero within-group signal, got {[r.name for r in demoted._engineered_recipes_ or []]}"
    )
    sel = list(demoted.get_feature_names_out())
    control_engineered_names = {r.name for r in control_recipes}
    assert not (set(sel) & control_engineered_names), f"a demoted engineered feature survived into the final selection: {sel}"


def test_group_aware_off_demotion_pass_is_a_noop():
    """No-op control: without ``group_aware_mi`` (or without ``groups``), ``get_group_mi()`` returns
    ``None`` and the demotion pass never runs -- the naive-MI FE selection is untouched."""
    X, y, groups = _nonlinear_panel(2)
    m = _fit(X, y, groups, group_aware_mi=False, strict_groups=False)
    assert list(m._engineered_recipes_ or []), "the naive (group-unaware) fit should keep its engineered features"


def _leak_with_real_signal_panel(seed: int, G: int = 150, per: int = 40):
    """``x_leak`` is a pure between-group LEVEL (high global MI, ~0 within-group); ``x_within`` is a
    genuine within-group predictor. The FE hybrid-orth Hermite basis naturally builds
    ``x_within*x_leak__He*`` composites from these two, and MRMR's raw-redundancy sweep drops BOTH raws
    as subsumed by them -- then the linear no-harm guard (held-out Ridge R2 on the composites alone is
    far below the raw-only baseline) REVERT-restores both raws, including ``x_leak``, purely because it
    looks good on a naive linear fit. This is the exact scenario the leak-exemption fix targets."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(G), per)
    n = groups.size
    gmean = rng.normal(size=G)[groups]
    x_within = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    y = gmean + 0.9 * x_within + 0.05 * rng.normal(size=n)
    X = pd.DataFrame({"x_leak": gmean.copy(), "x_within": x_within.copy(), "x_noise": x_noise.copy()})
    return X, y, groups


def test_group_aware_raw_redundancy_revert_does_not_restore_the_leak():
    """Regression: the raw-redundancy no-harm Ridge REVERT (``_fe_raw_redundancy_drop.py``) must not
    restore a group-aware-demoted leak raw just because dropping it hurts a naive linear fit -- that is
    exactly the leakage the group-aware gate exists to catch. The genuine within-group signal
    (``x_within``) and its real engineered composites must still be retained."""
    X, y, groups = _leak_with_real_signal_panel(1)
    m = _fit(X, y, groups, group_aware_mi=True)
    sel = list(m.get_feature_names_out())
    assert "x_leak" not in sel, f"the raw-redundancy REVERT must not resurrect the group-aware-demoted leak; got {sel}"
    assert "x_within" in sel or any("x_within" in str(nm) for nm in sel), f"genuine within-group signal must be retained; got {sel}"


def test_group_aware_off_raw_redundancy_revert_restores_the_leak_baseline():
    """Baseline confirming the fixture reproduces a real revert-restored leak at all: without
    ``group_aware_mi`` the no-harm guard has no leak-exemption to apply and restores ``x_leak`` exactly
    as ``test_group_naive_selects_the_leak_feature_baseline`` (test_group_aware_mi_mrmr.py) shows for the
    plain raw-only case."""
    X, y, groups = _leak_with_real_signal_panel(1)
    m = _fit(X, y, groups, group_aware_mi=False, strict_groups=False)
    sel = list(m.get_feature_names_out())
    assert "x_leak" in sel, f"group-naive baseline should still resurrect the leak via the no-harm revert; got {sel}"


# Broad stress sweep (item 1 of the group_aware_mi follow-up): the fix lands as a UNIVERSAL final
# choke-point over ``self._engineered_recipes_``/``self._engineered_features_`` in ``_fit_impl_core.py``,
# not per-FE-family wiring -- so it should hold regardless of WHICH producer(s) built the surviving
# composite. Exercise a wide set of independent-opt-in FE families together (each individually gated
# behind its own ``fe_hybrid_orth_*_enable`` flag, none requiring the others) across several seeds.
_STRESS_FE_KW = dict(
    fe_max_steps=2,
    fe_hybrid_orth_enable=True,
    fe_hybrid_orth_adaptive_arity_enable=True,
    fe_hybrid_orth_lasso_enable=True,
    fe_hybrid_orth_elasticnet_enable=True,
    fe_hybrid_orth_bootstrap_enable=True,
    fe_hybrid_orth_three_gate_enable=True,
    fe_hybrid_orth_ksg_enable=True,
    fe_hybrid_orth_copula_enable=True,
    fe_hybrid_orth_dcor_enable=True,
    fe_hybrid_orth_hsic_enable=True,
    fe_hybrid_orth_jmim_enable=True,
    fe_hybrid_orth_tc_enable=True,
    fe_hybrid_orth_cmim_enable=True,
)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_group_aware_leak_excluded_across_a_broad_fe_family_sweep(seed):
    """Stress sweep: with a wide set of independent-opt-in hybrid-orth FE families all enabled together
    (adaptive-arity, lasso, elasticnet, bootstrap, three-gate, KSG, copula, dCor, HSIC, JMIM, TC, CMIM --
    each its own producer/scorer, none requiring the others), the between-group-level leak must never
    reach the final selection AS A RAW COLUMN, regardless of which family's composite would otherwise
    have carried it through. A composite MIXING the leak with the genuine ``x_within`` signal (e.g.
    ``x_within*x_leak__He*``) legitimately retains real within-group information from ``x_within`` and
    is correctly NOT demoted -- only ``x_leak`` alone (no real signal contributed) is asserted excluded."""
    X, y, groups = _leak_with_real_signal_panel(seed)
    m = _fit(X, y, groups, group_aware_mi=True, **_STRESS_FE_KW)
    sel = list(m.get_feature_names_out())
    assert "x_leak" not in sel, f"[seed={seed}] the raw leak resurfaced under the broad FE-family sweep; got {sel}"
