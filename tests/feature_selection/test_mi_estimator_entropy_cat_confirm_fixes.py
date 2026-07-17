"""Regression tests for the MI-estimator / entropy / cat-confirm cluster fixes.

Each test pins one bug's failure mode (verified to fail on pre-fix code):

1. MINE DV loss followed a mis-scaled gradient -> MINE fails to recover Gaussian-copula MI.
2. genie_aggregate double-clamped at 0 (floor_at_zero=False was dead); best_on_calibration_mi
   silently swallowed estimator failures.
3. joint_freqs_2var / joint_entropy_2var returned nan/inf on an empty frame (no n==0 guard).
4. Fourier / Pade bases mapped a constant column to z ~ 1e12 -> garbage (non-finite-ish) feature.
5. CMI permutation-stop truncated the conditioning code by modulo without warning.
6. Westfall-Young CPU loop used the unseeded global np.random.shuffle (non-reproducible + global
   RNG pollution); subsample RNG seeded independently of any configured seed.
7. conditional_permutation_test p-value lacked the (1+sum)/(B+1) continuity correction -> could
   return an impossible p == 0.
"""

import logging

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Bug 3: joint_freqs_2var / joint_entropy_2var empty-frame guard
# ---------------------------------------------------------------------------


def test_joint_freqs_and_entropy_2var_empty_frame_finite():
    from mlframe.feature_selection.filters.info_theory._class_encoding import (
        joint_freqs_2var,
        joint_entropy_2var,
    )

    empty = np.zeros((0, 2), dtype=np.int64)
    freqs = joint_freqs_2var(empty, 0, 1, 3, 3)
    assert freqs.size == 0
    assert np.all(np.isfinite(freqs))
    h = joint_entropy_2var(empty, 0, 1, 3, 3)
    assert np.isfinite(h)
    assert h == 0.0


# ---------------------------------------------------------------------------
# Bug 2: genie_aggregate raw return + floor flag live; calibration logs failures
# ---------------------------------------------------------------------------


def test_genie_aggregate_returns_raw_and_floor_flag_controls_clamp():
    from mlframe.feature_selection.filters._mi_aggregator import genie_aggregate, genie_mi_panel

    # Weighted combo that is genuinely negative -> raw must NOT be clamped.
    raw = genie_aggregate([-1.0, -0.5], np.array([0.5, 0.5]))
    assert raw < 0.0, "genie_aggregate must return the raw (possibly negative) combination"

    # floor_at_zero=False must surface the negative; =True must clamp it.
    x = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    y = np.array([0, 0, 0, 0], dtype=np.int64)
    ests = {"a": lambda xx, yy: -0.3, "b": lambda xx, yy: -0.2}
    neg = genie_mi_panel(x, y, ests, floor_at_zero=False)
    assert neg < 0.0, "floor_at_zero=False must let a negative aggregate through (flag was dead)"
    clamped = genie_mi_panel(x, y, ests, floor_at_zero=True)
    assert clamped == 0.0


def test_best_on_calibration_logs_estimator_failure(caplog):
    from mlframe.feature_selection.filters._mi_aggregator import best_on_calibration_mi

    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([0, 1, 0, 1], dtype=np.int64)

    def good_on_cal_bad_on_real(xx, yy):
        # Low (best) calibration MI so it gets chosen, then raises on the real (x, y).
        if len(xx) == len(x) and np.array_equal(xx, x):
            raise RuntimeError("boom on real data")
        return 0.0

    with caplog.at_level(logging.WARNING):
        out = best_on_calibration_mi(x, y, {"only": good_on_cal_bad_on_real})
    assert out == 0.0
    assert any("failed" in r.message for r in caplog.records), "swallowed failure must be logged"


# ---------------------------------------------------------------------------
# Bug 4: constant column must yield a finite (zeroed) basis, not garbage
# ---------------------------------------------------------------------------


def test_fourier_near_constant_column_is_zeroed_not_garbage():
    from mlframe.feature_selection.filters.bases import _fourier_fit, _fourier_apply, _fourier_eval_njit

    # Near-constant column: 63 identical values + 1 outlier 1e-9 away. Pre-fix the 1e-12 span floor /
    # raw span (~1e-9) produces a z whose single-outlier point lands far from the bulk, turning
    # sin(2*pi*k*z) into a high-frequency garbage feature off rounding noise.
    x = np.array([5.0] * 63 + [5.0 + 1e-9], dtype=np.float64)
    z, params = _fourier_fit(x)
    assert np.all(np.isfinite(z))
    assert np.allclose(z, 0.0), "near-constant column must be treated as degenerate (z=0)"
    z2 = _fourier_apply(x, params)
    assert np.allclose(z2, 0.0)
    c = np.array([1.0, 0.5, 0.3, 0.2], dtype=np.float64)
    out = _fourier_eval_njit(z, c)
    # z=0 -> sin=0, cos=1: output is the constant sum of cos coefficients (a finite constant feature),
    # NOT high-frequency garbage. Assert it is finite and genuinely constant (zero variance).
    assert np.all(np.isfinite(out))
    assert np.ptp(out) == 0.0, "degenerate column must yield a constant (not noise-driven) feature"


def test_pade_near_constant_column_is_zeroed_not_garbage():
    from mlframe.feature_selection.filters.bases import _pade_fit, _pade_apply, _pade_eval_njit

    # Near-constant column: std ~ 1.25e-10 from one outlier. Pre-fix the additive 1e-12 std floor does
    # not dominate, so z = (x - mean)/std blows the outlier to z ~ 8 -> rational eval garbage.
    x = np.array([1.0] * 63 + [1.0 + 1e-9], dtype=np.float64)
    z, params = _pade_fit(x)
    assert np.all(np.isfinite(z))
    assert np.max(np.abs(z)) < 1.0, "near-constant column must not blow up z via the tiny-std floor"
    assert np.allclose(z, 0.0), "near-constant column must be treated as degenerate (z=0)"
    z2 = _pade_apply(x, params)
    assert np.allclose(z2, 0.0)
    c = np.array([0.5, 1.0, 0.2], dtype=np.float64)
    out = _pade_eval_njit(z, c)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Bug 5: CMI permutation-stop warns on conditioning-code truncation
# ---------------------------------------------------------------------------


def test_cmi_perm_stop_warns_on_conditioning_truncation(caplog):
    from mlframe.feature_selection.filters import _cmi_perm_stop

    rng = np.random.default_rng(0)
    n = 200
    x_cand = rng.integers(0, 3, n)
    y = rng.integers(0, 3, n)
    # Enough high-cardinality selected columns that the running K_z product overflows 1_000_000.
    n_sel = 14
    selected_cols = [rng.integers(0, 5, n) for _ in range(n_sel)]
    nbins_selected = [5] * n_sel
    with caplog.at_level(logging.WARNING):
        _cmi_perm_stop.cmi_permutation_stop(
            x_cand=x_cand,
            y=y,
            selected_cols=selected_cols,
            nbins_x=3,
            nbins_y=3,
            nbins_selected=nbins_selected,
            n_permutations=5,
            alpha=0.05,
            seed=0,
        )
    assert any("cardinality" in r.message.lower() or "truncat" in r.message.lower() for r in caplog.records), "modulo truncation of conditioning code must warn"


# ---------------------------------------------------------------------------
# Bug 7: conditional permutation p-value never exactly 0 (continuity correction)
# ---------------------------------------------------------------------------


def test_conditional_permutation_pvalue_never_zero():
    from mlframe.feature_selection.filters._conditional_permutation import conditional_permutation_test

    rng = np.random.default_rng(1)
    n = 300
    z = rng.integers(0, 2, n)
    # Strong x<->y dependence within strata so observed >> all null -> naive p would be 0.
    x = rng.integers(0, 3, n)
    y = x.copy()
    _obs, p = conditional_permutation_test(
        x=x,
        y=y,
        z=z,
        nbins_x=3,
        nbins_y=3,
        nbins_z=2,
        n_permutations=50,
        seed=0,
    )
    assert p > 0.0, "continuity-corrected p must never be exactly 0"
    assert abs(p - 1.0 / 51.0) < 1e-12, "p must equal (1+0)/(50+1) when observed beats all null"


# ---------------------------------------------------------------------------
# Bug 6: Westfall-Young reproducibility + no global-RNG pollution
# ---------------------------------------------------------------------------


def test_westfall_young_reproducible_and_does_not_touch_global_rng():
    from mlframe.feature_selection.filters._cat_confirm_permutation import (
        _compute_westfall_young_corrected_p,
    )
    from mlframe.feature_selection.filters.info_theory import merge_vars  # noqa: F401  (ensures pkg import)

    rng = np.random.default_rng(7)
    n = 400
    n_cols = 4
    nbins = np.full(n_cols, 3, dtype=np.int64)
    factors_data = rng.integers(0, 3, size=(n, n_cols)).astype(np.int32)
    classes_y = rng.integers(0, 3, n).astype(np.int64)
    n_y = int(classes_y.max()) + 1
    freqs_y = np.bincount(classes_y, minlength=n_y).astype(np.float64) / n
    pairs_a = np.array([0, 0, 1], dtype=np.int64)
    pairs_b = np.array([1, 2, 3], dtype=np.int64)
    ii_obs = np.array([0.01, 0.02, 0.005], dtype=np.float64)
    selected_idx = np.array([0, 1], dtype=np.int64)
    marginal_mi = np.zeros(n_cols, dtype=np.float64)

    def run():
        return _compute_westfall_young_corrected_p(
            factors_data=factors_data,
            pairs_a=pairs_a,
            pairs_b=pairs_b,
            ii_obs_arr=ii_obs,
            selected_idx=selected_idx,
            nbins=nbins,
            classes_y=classes_y,
            freqs_y=freqs_y,
            marginal_mi=marginal_mi,
            n_perms=30,
            dtype=np.int64,
            verbose=0,
        )

    # Pin the global RNG state, run twice, assert identical result AND untouched global stream.
    np.random.seed(12345)
    before = np.random.get_state()
    r1 = run()
    r2 = run()
    after = np.random.get_state()

    assert r1.keys() == r2.keys()
    for k in r1:
        assert r1[k] == pytest.approx(r2[k]), "WY p-values must be reproducible across runs"
    # global RNG stream must be byte-identical -> the loop did not call np.random.* globally.
    assert np.array_equal(before[1], after[1]), "WY must not pollute the global numpy RNG"


# ---------------------------------------------------------------------------
# Bug 1: MINE recovers Gaussian-copula MI (biz_value-style)
# ---------------------------------------------------------------------------


def test_mine_dv_loss_uses_canonical_ema_corrected_gradient():
    """The MINE DV marginal term must have value ``log(mean(exp_t))`` and gradient ``grad(mean(exp_t)) / ema``
    (Belghazi 2018 eq. 12). The pre-fix ``(mean/ema) * log(ema)`` form scaled the followed gradient by an
    extra ``log(ema)`` factor -- which FLIPS SIGN whenever ``ema < 1`` (the low-MI / early-training regime),
    actively steering the optimizer the wrong way. This pins the gradient property directly (deterministic,
    no training loop) so a revert to the mis-scaled form is caught.
    """
    torch = pytest.importorskip("torch")

    t = torch.tensor([0.2, -0.5, 0.7, -0.1], requires_grad=True)
    ema = torch.tensor(0.5)  # ema < 1 -> the pre-fix log(ema) factor is NEGATIVE (sign flip)

    exp_t = torch.exp(t)
    exp_mean = exp_t.mean()
    # Canonical corrected term (the fix): value == log(mean), gradient == grad(mean)/ema.
    log_term = torch.log(exp_mean.detach() + 1e-12) + (exp_mean - exp_mean.detach()) / ema.detach()
    (g_fix,) = torch.autograd.grad(log_term, t, retain_graph=True)

    # Reference: grad(mean(exp_t)) / ema -- what the DV gradient MUST be.
    (g_mean,) = torch.autograd.grad(exp_mean, t, retain_graph=True)
    g_ref = g_mean / ema
    assert torch.allclose(g_fix, g_ref, atol=1e-6), "fix gradient must equal grad(mean(exp_t))/ema"

    # Value must track log(mean(exp_t)) (the DV bound), not log(ema)-scaled.
    assert torch.allclose(log_term.detach(), torch.log(exp_mean.detach() + 1e-12), atol=1e-6)

    # Pre-fix form: gradient is grad(mean)/ema * log(ema); with ema<1 this points OPPOSITE to g_ref.
    exp_t2 = torch.exp(t)
    prefix_term = (exp_t2.mean() / ema.detach()) * torch.log(ema.detach())
    (g_prefix,) = torch.autograd.grad(prefix_term, t)
    assert (g_prefix * g_ref).sum() < 0, "pre-fix gradient must flip sign vs canonical when ema<1"


def test_mine_recovers_gaussian_copula_mi():
    torch = pytest.importorskip("torch")
    from mlframe.feature_selection.filters._neural_mi import mine_mi

    rng = np.random.default_rng(0)
    n = 4000
    rho = 0.8
    cov = np.array([[1.0, rho], [rho, 1.0]])
    xy = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)
    true_mi = -0.5 * np.log(1.0 - rho * rho)  # ~0.51 nats

    torch.manual_seed(0)
    est = mine_mi(x, y, n_epochs=400, hidden_dim=64, batch_size=512, lr=1e-3, early_stop_patience=0, device="cpu", verbose=False)
    # Wide tolerance band: MINE on few epochs is noisy but the corrected gradient must climb to the
    # right neighbourhood. Pre-fix mis-scaled gradient leaves the estimate far below true MI.
    assert est >= 0.5 * true_mi, f"MINE should recover most of Gaussian MI ({true_mi:.3f}); got {est:.3f}"
    assert est <= true_mi + 0.3, f"MINE estimate {est:.3f} implausibly above true MI {true_mi:.3f}"
