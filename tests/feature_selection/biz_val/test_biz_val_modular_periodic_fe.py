"""biz_value: PAIRWISE / n-way MODULAR relationship FE (cheap-first + escalate).

Quantitative contracts for ``_pairwise_modular_fe.detect_pairwise_modular`` -- the prototype that
extends single-column ``_periodic_fe`` (``x mod calendar_period``) to the uncovered case: a target
that is a function of an integer modulus of a COMBINATION of columns (``(a+b) mod m``, ``(a*b) mod m``,
n-way parity ``(x0+x1+x2) mod 2``) or a single column with a hidden PRIME period off the calendar ladder.

The detector is a PROTOTYPE (not wired into the public MRMR FE path yet) so these tests call it
directly. They are written to convert cleanly to public-API tests once shipped: the win they pin is the
MI LIFT of the materialised residue over the raw-combiner baseline (the best a smooth poly/Fourier basis
could recover) -- exactly the quantity the public path must preserve.

Measured (bench_modular_period_detection, 5 seeds): per-family MI lift 0.59..0.65, detection 1.0,
control FP 0.0, modulus accuracy 1.0. Floors set ~15-20% below measured to absorb seed noise.
Each test < 5s (n=2000, coarse modulus grid).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._pairwise_modular_fe import (
    cheap_modular_scan,
    detect_pairwise_modular,
)

SEEDS = (1, 7, 13, 42, 101)

# Measured per-family lifts are 0.59..0.65; floor 0.50 keeps regression sensitivity with margin.
_LIFT_FLOOR = 0.50


def _noise(rng, n, k, lo=0, hi=50):
    return {f"n{i}": rng.integers(lo, hi, n) for i in range(k)}


def _pair_add_mod(seed, n=2000, m=7):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 100, n), rng.integers(0, 100, n)
    y = ((a + b) % m >= (m // 2)).astype(int)
    return pd.DataFrame({"a": a, "b": b, **_noise(rng, n, 2)}), y


def _pair_mul_mod(seed, n=2000, m=5):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 40, n), rng.integers(0, 40, n)
    y = ((a * b) % m == 0).astype(int)
    return pd.DataFrame({"a": a, "b": b, **_noise(rng, n, 2)}), y


def _nway_parity(seed, n=2000, k=3):
    rng = np.random.default_rng(seed)
    cols = {f"a{i}": rng.integers(0, 1000, n) for i in range(k)}
    s = sum(cols.values())
    y = (s % 2).astype(int)
    return pd.DataFrame({**cols, **_noise(rng, n, 1)}), y


def _single_hidden_period(seed, n=2000, m=11):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 5000, n)
    y = (a % m >= (m // 2)).astype(int)
    return pd.DataFrame({"a": a, **_noise(rng, n, 2)}), y


class TestModularMILift:
    """The materialised residue must carry a large MI lift over the raw-combiner baseline."""

    @pytest.mark.parametrize(
        "gen,true_m",
        [
            (_pair_add_mod, 7),
            (_pair_mul_mod, 5),
            (_nway_parity, 2),
            (_single_hidden_period, 11),
        ],
    )
    def test_modular_residue_beats_smooth_baseline(self, gen, true_m):
        lifts, mods_ok = [], []
        for s in SEEDS:
            X, y = gen(s)
            hits = detect_pairwise_modular(X, y, top_k=4, seed=s)
            assert hits, f"seed={s}: detector fired nothing on a true modular target (m={true_m})."
            top = hits[0]
            lifts.append(top["margin"])
            mods_ok.append(top["modulus"] == true_m or top["modulus"] % true_m == 0)
        mean_lift = float(np.mean(lifts))
        assert mean_lift >= _LIFT_FLOOR, (
            f"modular residue MI lift {mean_lift:.4f} < {_LIFT_FLOOR} over the smooth-basis baseline "
            f"(per-seed {[round(x, 3) for x in lifts]}); the residue is not recovering structure a "
            f"poly/Fourier leg can't."
        )
        assert float(np.mean(mods_ok)) >= 0.8, (
            f"detected modulus matched the true period (or a multiple) in only {float(np.mean(mods_ok)):.0%} of seeds for m={true_m}."
        )


class TestSpecificity:
    """The detector must NOT fire on non-modular data (smooth / monotone / noise / ordinary interaction)."""

    def _smooth(self, s, n=2000):
        rng = np.random.default_rng(s)
        a, b = rng.integers(0, 100, n), rng.integers(0, 100, n)
        return pd.DataFrame({"a": a, "b": b, **_noise(rng, n, 2)}), ((a + 0.7 * b) > 85).astype(int)

    def _monotone(self, s, n=2000):
        rng = np.random.default_rng(s)
        a = rng.integers(0, 1000, n)
        return pd.DataFrame({"a": a, **_noise(rng, n, 3)}), (a > 500).astype(int)

    def _noise(self, s, n=2000):
        rng = np.random.default_rng(s)
        return pd.DataFrame(_noise(rng, n, 4)), rng.integers(0, 2, n)

    def _ordinary_mul(self, s, n=2000):
        rng = np.random.default_rng(s)
        a, b = rng.integers(0, 50, n), rng.integers(0, 50, n)
        return pd.DataFrame({"a": a, "b": b, **_noise(rng, n, 2)}), ((a * b) > 600).astype(int)

    @pytest.mark.parametrize("name", ["smooth", "monotone", "noise", "ordinary_mul"])
    def test_detector_silent_on_non_modular(self, name):
        gen = getattr(self, f"_{name}")
        fires = 0
        for s in SEEDS:
            X, y = gen(s)
            if detect_pairwise_modular(X, y, top_k=4, seed=s):
                fires += 1
        assert fires == 0, (
            f"detector spuriously injected a modular feature on the '{name}' control in {fires}/"
            f"{len(SEEDS)} seeds; the permutation-null + margin gate is not specific enough."
        )


class TestCheapGate:
    """The cheap scan's responded-gate is the escalation trigger: it must separate TP from controls."""

    def test_responded_flag_separates_tp_from_control(self):
        X_tp, y_tp = _pair_add_mod(7)
        X_ctrl, y_ctrl = TestSpecificity()._ordinary_mul(7)
        tp_hits = cheap_modular_scan(X_tp, y_tp, seed=7)
        ctrl_hits = cheap_modular_scan(X_ctrl, y_ctrl, seed=7)
        assert any(h.responded for h in tp_hits), "no cheap-scan hit responded on a true modular target."
        assert not any(h.responded for h in ctrl_hits), "a cheap-scan hit responded on the ordinary multiplicative-interaction control."


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s", "--no-cov"]))
