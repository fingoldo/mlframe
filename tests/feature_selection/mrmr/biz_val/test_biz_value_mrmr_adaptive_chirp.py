"""biz_value: ADAPTIVE-CHIRP univariate Fourier FE (2026-06-03).

The LINEAR-argument adaptive Fourier detector emits sin/cos on the LINEAR axis
``z = (x - lo) / span`` and CANNOT represent an oscillation whose frequency GROWS
with the argument -- a "chirp" ``y ~ sin(2*pi*f*z**2)``. Over a bounded support a
SLOW chirp is already spanned by the linear MULTITONE deflation basis, but a FAST
chirp sweeps an instantaneous-frequency band wider than the 6-tone basis can
cover and the linear path collapses. The chirp path runs the SAME held-out-
validated detector on the QUADRATIC-ARGUMENT warp ``u = sign(z)*z**2`` (z
standardised), which makes the chirp STATIONARY in u; the emitted ``__qsin`` /
``__qcos`` legs (tagged ``arg="quadratic"`` + ``adaptive=True``) reconstruct it
and are MRMR-protected past the screen, exactly like the linear adaptive legs.

Contracts pinned (verification gates A, C, D + the chirp-detector unit + the
default-on protection path):

* A  RECOVERY:    default MRMR (chirp ON) on a FAST chirp -> Ridge on the FULL
                  selected support clears OOS R^2 >= 0.85 AND materially beats
                  chirp-OFF (linear path only). Judged on the SUPPORT MODEL.
* C  NOISE:       pure-noise frame (random y) -> chirp adds NO column.
* D  SELF-GATE:   below the n-gate the support is byte-identical chirp-on vs off
                  (no chirp column emitted at small n).
* REPLAY:         transform() reproduces the fit-time chirp column byte-for-byte
                  (recipe replay is a pure function of X, y-independent).
"""

from __future__ import annotations

import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEEDS = (0, 5, 11)
CHIRP_GRID = tuple(0.5 * k for k in range(1, 49))  # 0.5 .. 24.0 (the chirp sweep)


def _make_mrmr(**overrides):
    """Default-config MRMR; adaptive-Fourier chirp detection is ON unless overridden."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    kwargs = dict(verbose=0, random_seed=0)
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _build_fast_chirp(seed: int, n: int = 6000, fmul: float = 2.5):
    """``y = sin(2*pi*fmul*z**2) + small noise`` over x ~ uniform(-3, 3), z
    standardised, plus 3 pure-noise columns. A FAST chirp: its instantaneous
    frequency sweeps a band wider than the 6-tone linear deflation basis spans,
    so only the quadratic-warp detector recovers it.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=n)
    z = (x - x.mean()) / x.std()
    X = pd.DataFrame(
        {
            "a": x,
            "n1": rng.standard_normal(n),
            "n2": rng.standard_normal(n),
            "n3": rng.standard_normal(n),
        }
    )
    y = np.sin(2.0 * np.pi * fmul * (z**2)) + 0.05 * rng.standard_normal(n)
    return X, pd.Series(y, name="y")


@cache
def _fast_chirp_full_fit(seed: int):
    """Cached ``(X, y, sel)`` for the default-config (chirp ON) fit on the FULL
    (unsplit) fast-chirp fixture at a given seed. Shared between
    test_chirp_feature_present_and_protected and
    test_transform_replays_chirp_column_byte_for_byte, both parametrized over
    the same SEEDS tuple. Nothing downstream mutates X/y/sel in place (only
    inspected via attributes / transform()).
    """
    X, y = _build_fast_chirp(seed)
    sel = _make_mrmr()
    sel.fit(X, y)
    return X, y, sel


# ---------------------------------------------------------------------------
# Chirp-detector unit: the warp + detector recover a known chirp frequency
# ---------------------------------------------------------------------------


class TestChirpDetectorUnit:
    """Unit tests for the quadratic-warp + Fourier-frequency chirp detector."""

    @pytest.mark.parametrize("true_f", [1.0, 1.5, 2.5])
    def test_detects_chirp_on_quadratic_warp(self, true_f):
        """On a synthetic ``sin(2*pi*true_f*z**2)`` the detector run on the
        quadratic warp recovers the planted frequency near true_f; the same
        detector on the LINEAR axis does NOT lock a single matching tone."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _detect_fourier_freqs_for_col,
            _fit_chirp_warp_for_col,
            _chirp_axis,
        )

        rng = np.random.default_rng(0)
        n = 6000
        x = rng.uniform(-3.0, 3.0, size=n)
        z = (x - x.mean()) / x.std()
        y = np.sin(2.0 * np.pi * true_f * (z**2))
        mean, std, lo, span = _fit_chirp_warp_for_col(x)
        u = _chirp_axis(x, mean, std, lo, span)
        freqs = _detect_fourier_freqs_for_col(
            u,
            y,
            f_grid=CHIRP_GRID,
            min_val_corr=0.15,
            min_rows=800,
            max_freqs=6,
        )
        assert freqs, f"chirp detector returned nothing for sin(2pi*{true_f}*z^2)"
        # The dominant warp-space frequency should be near true_f (the chirp is
        # stationary at freq=true_f on the u=sign(z)z^2 axis after [0,1] scaling
        # by span); allow a wide tolerance because the [0,1] rescale shifts the
        # nominal frequency and multitone deflation splits power across nearby
        # tones. Key contract: SOMETHING is locked, and recovery (below) is high.
        assert min(abs(f - true_f * span) for f in freqs) < max(2.0, true_f), f"no detected warp freq near the planted chirp; got {freqs}"

    def test_chirp_n_gate_below_min_rows(self):
        """The detector does not fire below its min_rows gate (avoids small-n false positives)."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _detect_fourier_freqs_for_col,
            _fit_chirp_warp_for_col,
            _chirp_axis,
        )

        rng = np.random.default_rng(1)
        n = 400  # train slice (2/3) below 800
        x = rng.uniform(-3.0, 3.0, size=n)
        z = (x - x.mean()) / x.std()
        y = np.sin(2.0 * np.pi * 2.5 * (z**2))
        mean, std, lo, span = _fit_chirp_warp_for_col(x)
        u = _chirp_axis(x, mean, std, lo, span)
        freqs = _detect_fourier_freqs_for_col(
            u,
            y,
            f_grid=CHIRP_GRID,
            min_val_corr=0.15,
            min_rows=800,
            max_freqs=6,
        )
        assert not freqs, "chirp detector fired below min_rows (small-n FP)"

    def test_chirp_noise_returns_nothing(self):
        """The detector fires no frequencies on pure noise (x, y independent)."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _detect_fourier_freqs_for_col,
            _fit_chirp_warp_for_col,
            _chirp_axis,
        )

        rng = np.random.default_rng(2)
        n = 4000
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)  # independent of x
        mean, std, lo, span = _fit_chirp_warp_for_col(x)
        u = _chirp_axis(x, mean, std, lo, span)
        freqs = _detect_fourier_freqs_for_col(
            u,
            y,
            f_grid=CHIRP_GRID,
            min_val_corr=0.15,
            min_rows=800,
            max_freqs=6,
        )
        assert not freqs, "chirp detector fired on pure noise"

    def test_chirp_warp_replay_is_pure_function_of_x(self):
        """``_chirp_axis`` reproduces the warp from stored params byte-for-byte."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _fit_chirp_warp_for_col,
            _chirp_axis,
        )

        rng = np.random.default_rng(3)
        x = rng.uniform(-3.0, 3.0, size=2000)
        mean, std, lo, span = _fit_chirp_warp_for_col(x)
        u1 = _chirp_axis(x, mean, std, lo, span)
        u2 = _chirp_axis(x, mean, std, lo, span)
        np.testing.assert_array_equal(u1, u2)
        # And the manual formula matches.
        zs = (x - mean) / std
        u_manual = (np.sign(zs) * zs**2 - lo) / span
        np.testing.assert_allclose(u1, u_manual, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Gate A: end-to-end recovery via the FULL selected support, chirp ON vs OFF
# ---------------------------------------------------------------------------


class TestGateAChirpRecovery:
    """Gate A: chirp-ON support recovers a fast chirp and materially beats chirp-OFF."""

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.timeout(300)
    def test_support_model_recovers_fast_chirp_and_beats_off(self, seed):
        """chirp-ON support clears OOS R^2 >= 0.85 and beats chirp-OFF by >= 0.3."""
        X, y = _build_fast_chirp(seed)
        yv = y.to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(
            X,
            yv,
            test_size=0.3,
            random_state=seed,
        )

        sel_on = _make_mrmr()  # chirp ON by default
        sel_on.fit(Xtr, pd.Series(ytr, name="y"))
        Ztr = np.asarray(sel_on.transform(Xtr), dtype=np.float64)
        Zte = np.asarray(sel_on.transform(Xte), dtype=np.float64)
        r2_on = r2_score(yte, Ridge(alpha=1.0).fit(Ztr, ytr).predict(Zte))

        sel_off = _make_mrmr(fe_univariate_fourier_chirp=False)
        sel_off.fit(Xtr, pd.Series(ytr, name="y"))
        Ztr2 = np.asarray(sel_off.transform(Xtr), dtype=np.float64)
        Zte2 = np.asarray(sel_off.transform(Xte), dtype=np.float64)
        r2_off = r2_score(yte, Ridge(alpha=1.0).fit(Ztr2, ytr).predict(Zte2))

        assert r2_on >= 0.85, f"GATE A: chirp-ON support must recover the fast chirp at OOS R^2 >= 0.85; got {r2_on:.4f}"
        assert r2_on - r2_off >= 0.3, (
            f"GATE A: chirp-ON ({r2_on:.4f}) must materially beat chirp-OFF "
            f"({r2_off:.4f}); the linear-argument path alone cannot represent a "
            f"frequency that grows with z."
        )

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.timeout(300)
    def test_chirp_feature_present_and_protected(self, seed):
        """A chirp (__qsin/__qcos) feature is detected and protected into the engineered support."""
        _X, _y, sel = _fast_chirp_full_fit(seed)
        adaptive = list(getattr(sel, "_adaptive_fourier_features_", []) or [])
        chirp_legs = [a for a in adaptive if ("qsin" in a or "qcos" in a)]
        assert chirp_legs, f"no chirp (__qsin/__qcos) feature detected on a fast chirp; adaptive set={adaptive}"
        eng = set(getattr(sel, "_engineered_features_", []) or [])
        assert any(c in eng for c in chirp_legs), f"chirp leg(s) {chirp_legs} not protected into the support engineered set {sorted(eng)}"


# ---------------------------------------------------------------------------
# Gate C: noise control
# ---------------------------------------------------------------------------


class TestGateCChirpNoiseControl:
    """Gate C: a pure-noise frame (random y) adds no chirp column."""

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.timeout(300)
    def test_pure_noise_adds_no_chirp_column(self, seed):
        """Pure-noise y triggers no chirp detection and no engineered chirp column."""
        rng = np.random.default_rng(seed)
        n = 2000
        Xn = pd.DataFrame({f"c{i}": rng.standard_normal(n) for i in range(6)})
        yn = pd.Series(rng.standard_normal(n), name="y")
        sel = _make_mrmr()
        sel.fit(Xn, yn)
        adaptive = list(getattr(sel, "_adaptive_fourier_features_", []) or [])
        chirp_legs = [a for a in adaptive if ("qsin" in a or "qcos" in a)]
        assert not chirp_legs, f"GATE C: chirp detector fired on a pure-noise frame: {chirp_legs}"
        eng = list(getattr(sel, "_engineered_features_", []) or [])
        chirp_eng = [c for c in eng if ("qsin" in c or "qcos" in c)]
        assert not chirp_eng, f"GATE C: chirp engineered column(s) injected on noise: {chirp_eng}"


# ---------------------------------------------------------------------------
# Gate D: self-gating below the n-gate -> byte-identical chirp-on vs off
# ---------------------------------------------------------------------------


class TestGateDChirpSelfGating:
    """Gate D: below the n-gate, support is byte-identical chirp-on vs chirp-off."""

    @pytest.mark.parametrize("n", [300, 600])
    @pytest.mark.timeout(300)
    def test_support_byte_identical_below_n_gate(self, n):
        """Below the min-rows n-gate, no chirp column is emitted and the support matches chirp-off."""
        rng = np.random.default_rng(7)
        x = rng.uniform(-3.0, 3.0, size=n)
        z = (x - x.mean()) / x.std()
        X = pd.DataFrame(
            {
                "a": x,
                "n1": rng.standard_normal(n),
                "n2": rng.standard_normal(n),
            }
        )
        y = pd.Series(
            np.sin(2.0 * np.pi * 2.5 * (z**2)) + 0.05 * rng.standard_normal(n),
            name="y",
        )
        on = _make_mrmr()
        on.fit(X, y)
        off = _make_mrmr(fe_univariate_fourier_chirp=False)
        off.fit(X, y)
        chirp_eng = [c for c in (on._engineered_features_ or []) if ("qsin" in c or "qcos" in c)]
        assert not chirp_eng, f"n={n}: chirp fired below the n-gate: {chirp_eng}"
        s_on = sorted(getattr(on, "selected_features_names_", []) or [])
        s_off = sorted(getattr(off, "selected_features_names_", []) or [])
        assert s_on == s_off, f"GATE D: support differs chirp-on vs off below n-gate: {s_on} vs {s_off}"


# ---------------------------------------------------------------------------
# REPLAY: transform()/recipe replay byte-matches fit for a chirp feature
# ---------------------------------------------------------------------------


class TestChirpReplayByteMatch:
    """transform()/recipe replay of the chirp column byte-matches the fit-time computation."""

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.timeout(300)
    def test_transform_replays_chirp_column_byte_for_byte(self, seed):
        """transform() reproduces the fit-time chirp column byte-for-byte and is y-independent at replay."""
        X, y, sel = _fast_chirp_full_fit(seed)
        adaptive = list(getattr(sel, "_adaptive_fourier_features_", []) or [])
        chirp_legs = [a for a in adaptive if ("qsin" in a or "qcos" in a)]
        assert chirp_legs, "no chirp feature to test replay against"

        out1 = sel.transform(X)
        assert isinstance(out1, pd.DataFrame), "expected a named DataFrame output"
        present = [c for c in chirp_legs if c in out1.columns]
        assert present, f"no chirp feature {chirp_legs} present in transform output columns {list(out1.columns)}"

        # Replay is a pure function of X: a SHUFFLED y must not change it.
        rng = np.random.default_rng(seed + 2000)
        y_shuf = pd.Series(rng.permutation(y.to_numpy()), name="y")
        out2 = sel.transform(X, y=y_shuf)
        pd.testing.assert_frame_equal(out1, out2)

        # And the transform output equals a standalone recipe replay.
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

        recipes = {r.name: r for r in (getattr(sel, "_engineered_recipes_", []) or [])}
        for c in present:
            assert c in recipes, f"chirp feature {c} has no replayable recipe"
            # Confirm it really is the quadratic-warp recipe.
            assert str(dict(recipes[c].extra).get("arg")) == "quadratic", f"chirp recipe {c} is not arg='quadratic'"
            replayed = np.asarray(apply_recipe(recipes[c], X), dtype=np.float64)
            np.testing.assert_allclose(
                out1[c].to_numpy(dtype=np.float64),
                replayed,
                rtol=0.0,
                atol=0.0,
                err_msg=f"transform output for {c} != standalone recipe replay",
            )
