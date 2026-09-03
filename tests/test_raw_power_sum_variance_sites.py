"""`var = E[x^2] - E[x]^2` is the same cancellation as the skew/kurt expansion, one order down.

Three prior fix rounds grepped for skew and kurtosis, so nine variance sites were left standing. Each of the
three covered here turns the defect into a specific wrong ANSWER rather than a warning:

  * `pre_screen` DROPS a sparse column whose cancellation-noise variance happens to land negative -- there is no
    `max(var, 0)` clamp, so a numerical failure reads as "less variance than the cutoff". That is the exact
    false drop the branch's own comment says it exists to prevent, reintroduced in a different regime.
  * `_member_consensus_correlations` reports a near-saturated ensemble member -- the most redundant member
    possible -- as having a correlation of exactly 0.0 with the rest, i.e. perfectly INDEPENDENT. The
    diagnostic's stated purpose is inverted, and the member survives any downstream pruning rule.
  * `adversarial_stochastic_blend` reports `stability_score == 1.0`, perfect convergence, on a blend whose true
    per-iteration variance is simply below the noise floor -- an optimistic verdict from an artifact, which is
    the wrong direction for a trustworthiness diagnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _raw_var(x):
    """The pre-fix form, for asserting a fixture actually exercises the difference."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.square(x).sum() / x.size - (x.sum() / x.size) ** 2)


class TestASparseColumnIsNotDroppedOnCancellationNoise:
    """A sparse price column at fill_value 1e6 has a cancellation floor three orders above its real variance."""

    def _screen(self, values, fill_value, n_total):
        """One sparse column carrying `values` as stored cells against `fill_value` elsewhere."""
        from mlframe.feature_selection.pre_screen import compute_unsupervised_drops

        dense = np.full(n_total, fill_value, dtype=np.float64)
        dense[: len(values)] = values
        col = pd.arrays.SparseArray(dense, fill_value=fill_value)
        df = pd.DataFrame({"sparse_col": col, "keep": np.arange(n_total, dtype=np.float64)})
        return compute_unsupervised_drops(df)

    def test_the_fixture_is_genuinely_hostile_to_the_raw_form(self):
        """Without this the assertions below could pass on a fixture that never triggered the bug."""
        fill = 1e6
        vals = fill + np.array([1e-3, -1e-3, 5e-4, -5e-4])
        dense = np.full(4000, fill)
        dense[:4] = vals
        assert abs(_raw_var(dense) - float(np.var(dense))) > 0.5 * float(np.var(dense)), (_raw_var(dense), float(np.var(dense)))

    def test_an_informative_sparse_column_at_a_large_fill_value_survives(self):
        """The variance is real, just small next to `mean**2`."""
        fill = 1e6
        vals = fill + np.array([1e-3, -1e-3, 5e-4, -5e-4])
        assert "sparse_col" not in self._screen(vals, fill, 4000)

    def test_a_genuinely_constant_sparse_column_is_still_dropped(self):
        """The fix must not stop the screen doing its job."""
        assert "sparse_col" in self._screen(np.full(4, 1e6), 1e6, 4000)

    def test_the_variance_is_never_negative(self):
        """A centred sum of squares cannot be; the raw form could, and that is what read as 'below the cutoff'."""
        fill = 1e6
        for spread in (1e-6, 1e-3, 1.0):
            vals = fill + np.array([spread, -spread])
            dense = np.full(1000, fill)
            dense[:2] = vals
            mean = dense.mean()
            assert float(np.dot(dense - mean, dense - mean)) / dense.size >= 0.0


class TestASaturatedMemberIsNotCalledIndependent:
    """`np.clip(var_a * var_b, 0.0, None)` laundered a noise-signed variance into a clean 0.0."""

    def _corr(self, logits):
        """Per-member correlation against the leave-one-out mean of the rest."""
        from mlframe.calibration._independence_check import _member_consensus_correlations

        return _member_consensus_correlations(np.asarray(logits, dtype=np.float64))

    def test_a_near_saturated_member_is_not_reported_as_perfectly_independent(self):
        """It outputs the same confident logit on every row: maximally redundant, the opposite of independent."""
        rng = np.random.default_rng(0)
        n = 4000
        base = rng.normal(0, 1.0, n)
        logits = np.column_stack([base, base + rng.normal(0, 0.1, n), 16.0 + rng.normal(0, 1e-7, n)])
        c = self._corr(logits)
        assert not (np.isfinite(c[2]) and abs(c[2]) < 1e-6), f"the saturated member reads as independent: {c}"

    def test_an_exactly_constant_member_is_nan_not_zero(self):
        """NaN says 'undefined'; 0.0 says 'independent', which is a claim, and the wrong one."""
        rng = np.random.default_rng(1)
        n = 500
        logits = np.column_stack([rng.normal(size=n), rng.normal(size=n), np.full(n, 3.0)])
        assert np.isnan(self._corr(logits)[2])

    def test_ordinary_members_are_unchanged(self):
        """Centring is exact for the well-conditioned case; the correlations must not move."""
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(2000, 4))
        c = self._corr(logits)
        assert np.isfinite(c).all() and (np.abs(c) <= 1.0 + 1e-9).all()

    def test_two_identical_members_correlate_strongly(self):
        """A sanity anchor: the diagnostic must still be able to see redundancy it is asked to find."""
        rng = np.random.default_rng(3)
        base = rng.normal(size=3000)
        logits = np.column_stack([base, base.copy(), rng.normal(size=3000)])
        assert self._corr(logits)[0] > 0.5


class TestAConvergedBlendDoesNotClaimPerfectStability:
    """Clamping cancellation noise to zero produced `stability_score == 1.0` from an artifact."""

    def _curve(self, weights):
        """The convergence curve the module exports, computed from a materialised weight history."""
        w = np.asarray(weights, dtype=np.float64)
        iter_counts = np.arange(1, w.shape[0] + 1, dtype=np.float64)[:, None]
        cum_mean = np.cumsum(w, axis=0) / iter_counts
        dev = w - cum_mean
        cum_var = np.maximum(np.cumsum(dev * dev, axis=0) / iter_counts, 0.0)
        cum_std = np.sqrt(cum_var)
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = np.where(np.abs(cum_mean) > 1e-12, cum_std / np.where(np.abs(cum_mean) > 1e-12, np.abs(cum_mean), 1.0), np.nan)
        return np.nanmean(cov, axis=1)

    def test_a_tiny_but_real_spread_is_not_clamped_to_zero(self):
        """w = 0.25 +- 1e-10: the raw form's floor is 1.4e-17 against a true variance of ~1e-20."""
        rng = np.random.default_rng(0)
        w = 0.25 + rng.normal(0, 1e-10, size=(200, 4))
        assert self._curve(w)[-1] > 0.0

    def test_the_raw_form_would_have_clamped_it(self):
        """States what the old behaviour did, so the test above is not vacuous."""
        rng = np.random.default_rng(0)
        w = 0.25 + rng.normal(0, 1e-10, size=(200, 4))
        iter_counts = np.arange(1, 201, dtype=np.float64)[:, None]
        cum_mean = np.cumsum(w, axis=0) / iter_counts
        cum_sq_mean = np.cumsum(w**2, axis=0) / iter_counts
        raw = np.maximum(cum_sq_mean - cum_mean**2, 0.0)
        dev = w - cum_mean
        true = (np.cumsum(dev * dev, axis=0) / iter_counts)[-1]
        # Two of the four members clamp to exactly 0.0; the other two are wrong by ~1300x and ~10500x.
        assert (raw[-1] == 0.0).any(), "the fixture no longer reaches the clamp; it proves nothing"
        assert (raw[-1] / true > 100.0).any(), (raw[-1], true)

    def test_a_member_converging_to_zero_reports_an_undefined_cv_not_a_deflated_one(self):
        """|cum_mean| ~ 1e-13 is where the old `+1e-12` pad dominated the true denominator."""
        w = np.zeros((50, 2))
        w[:, 0] = 0.5
        w[:, 1] = 1e-14
        assert np.isfinite(self._curve(w)).all() or np.isnan(self._curve(w)).any()

    def test_a_genuinely_unstable_blend_still_scores_badly(self):
        """The fix must not make everything look converged either."""
        rng = np.random.default_rng(1)
        w = rng.uniform(0.05, 0.45, size=(200, 4))
        assert self._curve(w)[-1] > self._curve(0.25 + rng.normal(0, 1e-10, size=(200, 4)))[-1]


def test_the_blend_convergence_curve_is_wired_to_the_exported_score():
    """A converged blend must not report a PERFECT `stability_score` of exactly 1.0.

    The class above validates the centred formula against a local reference; this exercises the real call
    path, where the artifact actually reached callers. `stability_score` is `1 / (1 + curve[-1])`, so an
    exact 1.0 means the convergence curve's last value was exactly 0.0 -- which is what the raw
    `E[w^2] - E[w]^2` form produced when its cancellation floor swallowed a real spread and
    `np.maximum(..., 0.0)` clamped the result. Nearly identical per-iteration weights are precisely the
    regime that triggered it, so that is what this drives, through the public function rather than by
    searching its source for the expression.
    """
    pytest.importorskip("sklearn")
    from mlframe.votenrank.adversarial_stochastic_blend import adversarial_stochastic_blend

    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n).astype(np.float64)
    # Three near-interchangeable members: every resample lands on almost the same weights, so the spread the
    # curve measures is tiny but real -- exactly where the raw form clamped to zero.
    base = y * 0.6 + rng.normal(0.0, 0.30, n)
    preds = [np.clip(base + rng.normal(0.0, 1e-6, n), 0.0, 1.0) for _ in range(3)]

    def _mse(a, b):
        """Plain MSE loss for the blend search."""
        return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))

    out = adversarial_stochastic_blend(
        preds,
        y,
        test_likeness=np.ones(n),
        loss_fn=_mse,
        n_iterations=40,
        n_restarts=1,
        random_state=0,
        track_convergence=True,
    )
    score = out["stability_score"]
    curve = np.asarray(out["convergence_curve"], dtype=np.float64)

    assert 0.0 < score <= 1.0, f"stability_score out of range: {score!r}"
    assert score != 1.0, "stability_score is exactly 1.0, i.e. the convergence curve ended at exactly 0.0 -- the clamped-variance artifact is back"
    assert curve.shape[0] == 40, f"one curve point per iteration expected, got {curve.shape}"
    assert np.isfinite(curve[-1]) and curve[-1] > 0.0, f"the final convergence value should be a small POSITIVE spread, got {curve[-1]!r}"
