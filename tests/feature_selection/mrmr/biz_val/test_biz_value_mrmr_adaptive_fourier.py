"""biz_value: ADAPTIVE-FREQUENCY Fourier univariate FE (2026-06-03).

The fixed Fourier univariate grid only covers z-space frequencies {1, 2}; an
ARBITRARY-period oscillation (``y = sin(3.7*x)``, ``sin(5.3*x)``, ``sin(6.8*x)``)
lands at a NON-integer z-space frequency and is MISSED by the fixed grid
(recovered at |corr| 0.02-0.23). The adaptive detector sweeps a coarse z-space
frequency grid, local-refines around the peak, validates the dominant frequency
on a HELD-OUT stride slice, and -- when it clears the floor -- adds it to that
column's Fourier set, tagged ``adaptive=True`` so MRMR protects the held-out-
validated sin/cos pair past the screen (a single leg has low marginal MI --
phase -- so the screen would otherwise keep a lower-MI fixed-freq twin).

Contracts pinned (verification gates A, C, D, plus the detector unit + the
default-on protection path):

* A  RECOVERY:    default MRMR (adaptive ON) on a 3-tone arbitrary-period
                  signal -> Ridge on the FULL selected support clears OOS
                  R^2 >= 0.9. Judged on the SUPPORT MODEL, not single-feature
                  |corr| (the phase-split legs each have low marginal |corr|).
* C  NOISE:       pure-noise frame (random y) -> adaptive adds NO column.
* D  REPLAY:      transform() reproduces the fit-time adaptive column byte-for-
                  byte (recipe replay is a pure function of X, y-independent).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")

SEEDS = (0, 5, 11)


def _make_mrmr(**overrides):
    from mlframe.feature_selection.filters.mrmr import MRMR

    kwargs = dict(verbose=0, random_seed=0)
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _build_multitone(seed: int, n: int = 4000):
    """``y = sin(3.7*x) + sin(5.3*x) + sin(6.8*x) + small noise`` over
    x ~ uniform(-3, 3), plus 4 pure-noise columns. None of the three tones
    lands on the fixed z-space grid {1, 2}; only adaptive detection recovers
    them.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=n)
    X = pd.DataFrame(
        {
            "a": x,
            "n1": rng.standard_normal(n),
            "n2": rng.standard_normal(n),
            "n3": rng.standard_normal(n),
            "n4": rng.standard_normal(n),
        }
    )
    y = np.sin(3.7 * x) + np.sin(5.3 * x) + np.sin(6.8 * x) + 0.05 * rng.standard_normal(n)
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Detector unit
# ---------------------------------------------------------------------------


class TestDetectorUnit:
    @pytest.mark.parametrize("true_w", [3.7, 5.3, 6.8])
    def test_detects_arbitrary_period(self, true_w):
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _detect_fourier_freq_for_col,
        )

        rng = np.random.default_rng(0)
        n = 4000
        x = rng.uniform(-3.0, 3.0, size=n)
        span = float(x.max() - x.min())
        z = (x - x.min()) / span
        y = np.sin(true_w * x)
        f = _detect_fourier_freq_for_col(
            z,
            y,
            f_grid=tuple(0.5 * k for k in range(1, 17)),
            min_val_corr=0.15,
            min_rows=800,
        )
        assert f is not None, f"detector returned None for sin({true_w}*x)"
        expected = true_w * span / (2.0 * np.pi)
        # Coarse-sweep + 0.05 refine should land within ~0.3 of the true freq.
        assert abs(f - expected) < 0.35, f"detected z-freq {f:.3f} far from expected {expected:.3f} for sin({true_w}*x)"

    def test_n_gate_below_min_rows_returns_none(self):
        """N-gated: below min_rows the detector must not fire (small-n false
        positive guard)."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _detect_fourier_freq_for_col,
        )

        rng = np.random.default_rng(1)
        n = 400  # below 800
        x = rng.uniform(-3.0, 3.0, size=n)
        z = (x - x.min()) / float(x.max() - x.min())
        y = np.sin(5.3 * x)
        f = _detect_fourier_freq_for_col(
            z,
            y,
            f_grid=tuple(0.5 * k for k in range(1, 17)),
            min_val_corr=0.15,
            min_rows=800,
        )
        assert f is None, "detector fired below min_rows (small-n FP)"

    def test_noise_returns_none(self):
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _detect_fourier_freq_for_col,
        )

        rng = np.random.default_rng(2)
        n = 2000
        x = rng.standard_normal(n)
        z = (x - x.min()) / float(x.max() - x.min())
        y = rng.standard_normal(n)  # independent of x
        f = _detect_fourier_freq_for_col(
            z,
            y,
            f_grid=tuple(0.5 * k for k in range(1, 17)),
            min_val_corr=0.15,
            min_rows=800,
        )
        assert f is None, "detector fired on pure noise"


# ---------------------------------------------------------------------------
# Gate A: end-to-end recovery via the FULL selected support
# ---------------------------------------------------------------------------


class TestGateARecovery:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.timeout(300)
    def test_support_model_recovers_multitone(self, seed):
        X, y = _build_multitone(seed)
        yv = y.to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(
            X,
            yv,
            test_size=0.3,
            random_state=seed,
        )
        sel = _make_mrmr()  # adaptive ON by default
        sel.fit(Xtr, pd.Series(ytr, name="y"))
        Ztr = sel.transform(Xtr)
        Zte = sel.transform(Xte)
        Ztr_a = np.asarray(Ztr, dtype=np.float64)
        Zte_a = np.asarray(Zte, dtype=np.float64)
        assert Ztr_a.shape[1] >= 1
        model = Ridge(alpha=1.0).fit(Ztr_a, ytr)
        r2 = r2_score(yte, model.predict(Zte_a))
        assert r2 >= 0.9, (
            f"GATE A: Ridge on the full selected support must recover the "
            f"arbitrary-period multitone signal at OOS R^2 >= 0.9; got {r2:.4f}. "
            f"adaptive_features={getattr(sel, '_adaptive_fourier_features_', None)}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.timeout(300)
    def test_adaptive_feature_present_in_support(self, seed):
        X, y = _build_multitone(seed)
        sel = _make_mrmr()
        sel.fit(X, y)
        adaptive = list(getattr(sel, "_adaptive_fourier_features_", []) or [])
        assert adaptive, "no adaptive Fourier feature detected on a 3-tone signal"
        eng = set(getattr(sel, "_engineered_features_", []) or [])
        assert any(a in eng for a in adaptive), f"adaptive feature(s) {adaptive} not protected into the support engineered set {sorted(eng)}"


# ---------------------------------------------------------------------------
# Gate C: noise control
# ---------------------------------------------------------------------------


class TestGateCNoiseControl:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.timeout(300)
    def test_pure_noise_adds_no_adaptive_column(self, seed):
        rng = np.random.default_rng(seed)
        n = 2000
        Xn = pd.DataFrame({f"c{i}": rng.standard_normal(n) for i in range(6)})
        yn = pd.Series(rng.standard_normal(n), name="y")
        sel = _make_mrmr()
        sel.fit(Xn, yn)
        adaptive = list(getattr(sel, "_adaptive_fourier_features_", []) or [])
        assert not adaptive, f"GATE C: adaptive detector fired on a pure-noise frame: {adaptive}"
        # Support stays tiny (no engineered oscillation injected).
        eng = list(getattr(sel, "_engineered_features_", []) or [])
        fourier_eng = [c for c in eng if ("sin" in c or "cos" in c)]
        assert not fourier_eng, f"GATE C: Fourier engineered column(s) injected on noise: {fourier_eng}"


# ---------------------------------------------------------------------------
# Gate D: transform()/recipe replay byte-matches fit for a re-added adaptive
# feature
# ---------------------------------------------------------------------------


class TestGateDReplayByteMatch:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.timeout(300)
    def test_transform_replays_adaptive_column_byte_for_byte(self, seed):
        X, y = _build_multitone(seed)
        sel = _make_mrmr()
        sel.fit(X, y)
        adaptive = list(getattr(sel, "_adaptive_fourier_features_", []) or [])
        assert adaptive, "no adaptive feature to test replay against"

        out1 = sel.transform(X)
        assert isinstance(out1, pd.DataFrame), "expected a named DataFrame output"
        # At least one re-added adaptive column must survive into transform out.
        present = [a for a in adaptive if a in out1.columns]
        assert present, f"no adaptive feature {adaptive} present in transform output columns {list(out1.columns)}"

        # Replay determinism: transform with a SHUFFLED y must be identical
        # (recipes are pure functions of X), and re-running transform matches.
        rng = np.random.default_rng(seed + 1000)
        y_shuf = pd.Series(rng.permutation(y.to_numpy()), name="y")
        out2 = sel.transform(X, y=y_shuf)
        pd.testing.assert_frame_equal(out1, out2)

        # And the recomputed adaptive column equals an independent
        # recipe-replay from the stored recipe (fit-time value reproduction).
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

        recipes = {r.name: r for r in (getattr(sel, "_engineered_recipes_", []) or [])}
        for a in present:
            assert a in recipes, f"adaptive feature {a} has no replayable recipe"
            replayed = np.asarray(apply_recipe(recipes[a], X), dtype=np.float64)
            np.testing.assert_allclose(
                out1[a].to_numpy(dtype=np.float64),
                replayed,
                rtol=0.0,
                atol=0.0,
                err_msg=f"transform output for {a} != standalone recipe replay",
            )
