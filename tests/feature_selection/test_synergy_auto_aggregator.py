"""Regression tests for the data-dependent ``redundancy_aggregator='auto'`` synergy gate.

The default Fleuret/CMIM redundancy gate rejects synergistic operands (a feature useless alone but informative jointly with an
already-selected partner). The JMIM aggregator recovers them but OVER-SELECTS correlated decoys on additive data, which is why it
stays opt-in. ``'auto'`` runs a cheap pre-fit synergy probe (interaction information of feature pairs vs a label-permuted null)
and routes to JMIM only when the data is synergistic, else stays plain Fleuret.

Pinned contracts:
  * detect_synergy fires True on a planted XOR/sign-product DGP and False on a planted additive/main-effect DGP (the HARD GATE).
  * MRMR(redundancy_aggregator='auto').fit records the routing decision and matches plain Fleuret on additive data (no
    over-selection regression) -- this is the no-regression guarantee that justified shipping 'auto'.
  * 'auto' is an accepted constructor value; a typo still raises.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._synergy_detector import detect_synergy
from mlframe.feature_selection.filters import MRMR


def _synergistic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    cols, rel, logit = [], [], np.zeros(n)
    for k in range(3):
        if k % 2 == 0:
            a = rng.integers(0, 2, n).astype(float); b = rng.integers(0, 2, n).astype(float)
            contrib = (a.astype(int) ^ b.astype(int)) * 2.0 - 1.0
        else:
            a = rng.standard_normal(n); b = rng.standard_normal(n)
            contrib = np.sign(a) * np.sign(b)
        rel += [len(cols), len(cols) + 1]
        cols += [a + 0.05 * rng.standard_normal(n), b + 0.05 * rng.standard_normal(n)]
        logit += 2.5 * contrib
    for j in range(8):
        cols.append(cols[0] + (0.3 + 0.05 * j) * rng.standard_normal(n))
    for _ in range(8):
        cols.append(rng.standard_normal(n))
    X = np.column_stack(cols)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, y, sorted(rel)


def _additive(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    cols, rel, logit = [], [], np.zeros(n)
    for _ in range(3):
        f = rng.standard_normal(n); cols.append(f); rel.append(len(cols) - 1); logit += 1.5 * f
    for j in range(8):
        cols.append(cols[0] + (0.3 + 0.05 * j) * rng.standard_normal(n))
    for _ in range(8):
        cols.append(rng.standard_normal(n))
    X = np.column_stack(cols)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, y, sorted(rel)


class TestDetector:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_fires_on_synergy(self, seed):
        X, y, _ = _synergistic(seed=seed)
        is_syn, info = detect_synergy(X, y, random_seed=seed)
        assert is_syn, f"synergy not detected: {info}"
        assert info["real_excess"] > info["threshold"]

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_silent_on_additive(self, seed):
        X, y, _ = _additive(seed=seed)
        is_syn, info = detect_synergy(X, y, random_seed=seed)
        assert not is_syn, f"false-positive synergy on additive data: {info}"

    def test_degenerate_inputs(self):
        assert detect_synergy(np.zeros((10, 2)), np.zeros(10))[0] is False
        X = np.random.default_rng(0).standard_normal((200, 3))
        assert detect_synergy(X, np.zeros(200))[0] is False  # constant target


def _fit(X, y, agg, seed=0):
    sel = MRMR(redundancy_aggregator=agg, fe_max_steps=0, interactions_max_order=1,
               full_npermutations=3, baseline_npermutations=2, random_seed=seed,
               use_gpu=False, n_jobs=1, verbose=0, cv=2)
    sel.fit(X, y)
    return sel


class TestAutoGate:
    def test_auto_matches_fleuret_on_additive(self):
        """HARD GATE: on additive data 'auto' must reproduce the plain-Fleuret selection (no JMIM over-selection)."""
        X, y, _ = _additive(seed=0)
        a = _fit(X, y, "auto")
        d = _fit(X, y, None)
        assert a._synergy_auto_decision_["jmim_engaged"] is False
        assert sorted(a.get_support(indices=True).tolist()) == sorted(d.get_support(indices=True).tolist())

    def test_auto_engages_jmim_on_synergy(self):
        X, y, _ = _synergistic(seed=0)
        a = _fit(X, y, "auto")
        assert a._synergy_auto_decision_["jmim_engaged"] is True

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            _fit(*_additive(n=400)[:2], "bogus")
