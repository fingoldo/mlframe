"""Tests for regression_residual_audit module.

Each noise pattern from the user's hypothesis table gets a dedicated
test that generates the right kind of residual structure and asserts
the audit reaches the expected hypothesis + suggested loss.

| pattern                  | generator                                        | expected hypothesis     | suggested loss              |
|--------------------------|--------------------------------------------------|--------------------------|------------------------------|
| Gaussian (well-behaved)  | residuals = N(0, 1)                              | "Gaussian"               | "MSE"                        |
| Heavy tails (Laplace)    | residuals = Laplace(0, 1)                        | "Laplace / Student-t"    | "MAE or Huber"               |
| Heavy tails (Student-t)  | residuals = StudentT(df=3)                       | "Laplace / Student-t"    | (same)                       |
| Outlier-contaminated     | mostly N(0,1) + 5% from N(0, 30)                 | "Contaminated"           | "Huber"                      |
| Gamma / Exponential      | y_true = exp(scale*N(0,1))-1 + Exp(rate=1)       | "Gamma / Exponential"    | "Gamma deviance loss"        |
| LogNormal multiplicative | y_true = y_pred * exp(N(0, 0.5)), y > 0          | "LogNormal"              | "MSE on log(y)"              |

Plus structural tests: format report, plot doesn't crash, sample_size
honored, NaN-filtering, edge cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.targets.regression_residual_audit import (
    DEFAULT_DIAG_SAMPLE_SIZE,
    HETERO_SPEARMAN_THRESHOLD,
    ResidualAudit,
    audit_residuals,
    format_residual_audit_report,
    plot_residual_diagnostics,
)


# -----------------------------------------------------------------------------
# Each noise pattern → expected hypothesis
# -----------------------------------------------------------------------------


def test_gaussian_well_behaved():
    """Symmetric, light-tailed, homoscedastic → MSE-default."""
    rng = np.random.default_rng(0)
    n = 5_000
    y_pred = rng.uniform(-3, 3, size=n)
    y_true = y_pred + rng.normal(0, 1.0, size=n)
    audit = audit_residuals(y_true, y_pred)
    assert audit.hypothesis == "Gaussian (well-behaved)"
    assert "MSE" in audit.suggested_loss
    assert abs(audit.skew) < 0.3
    assert abs(audit.excess_kurt) < 1.0


def test_mild_leptokurtosis_not_called_gaussian():
    """2026-05-11 regression: user-reported case where residuals
    showed ``excess_kurt=+2.40`` and the audit verdict said
    ``Gaussian (well-behaved)``. Visually the histogram was sharply
    peaked at 0 -- nothing close to Normal. With the threshold
    lowered from 3.0 to 1.5 (still allowing logistic-like residuals
    through), mildly leptokurtic distributions are now flagged
    honestly as ``Near-Gaussian (mildly peaky)`` or
    ``Laplace / Student-t`` depending on severity. The user's
    specific kurt=+2.4 sits in the heavy-tails bucket.
    """
    rng = np.random.default_rng(0)
    n = 5_000
    y_pred = rng.uniform(-3, 3, size=n)
    # Mixture: 70% tight (sigma=0.3) + 30% loose (sigma=2.0) -> sharp
    # peak at 0 with thicker shoulders. Mimics the production case
    # where 0.6% of residuals exceeded 3 sigma and kurt was around 2-3.
    component = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    y_true = y_pred + np.where(
        component == 0,
        rng.normal(0, 0.3, size=n),
        rng.normal(0, 2.0, size=n),
    )
    audit = audit_residuals(y_true, y_pred)
    # Sanity: the fixture indeed produces moderate leptokurtosis.
    assert audit.excess_kurt > 1.0, (
        f"fixture too weak: kurt={audit.excess_kurt:+.2f}; "
        "needs > 1 to test the bug"
    )
    # The contract: anything with excess_kurt > 1.5 must NOT be called
    # plain Gaussian.
    assert audit.hypothesis != "Gaussian (well-behaved)", (
        f"regression: residuals with excess_kurt={audit.excess_kurt:+.2f} "
        f"called Gaussian; this was the production profanation user "
        f"reported on 2026-05-11."
    )
    # Verdict label should indicate non-Gaussian (one of the documented
    # heavy-tails / peaky verdicts).
    assert audit.hypothesis in {
        "Laplace / Student-t",
        "Near-Gaussian (mildly peaky)",
        "Contaminated / outliers",
    }, f"unexpected hypothesis: {audit.hypothesis!r}"


def test_laplace_heavy_tails():
    """Symmetric but heavy-tailed → Laplace / Student-t → MAE/Huber."""
    rng = np.random.default_rng(0)
    n = 5_000
    y_pred = rng.uniform(-3, 3, size=n)
    # Laplace with scale b has variance 2*b^2; excess kurt = 3
    y_true = y_pred + rng.laplace(0, 1.0, size=n)
    audit = audit_residuals(y_true, y_pred)
    assert audit.hypothesis == "Laplace / Student-t"
    assert "MAE" in audit.suggested_loss or "Huber" in audit.suggested_loss
    assert audit.excess_kurt > 1.5  # Laplace's excess kurt is 3 in expectation


def test_student_t_heavy_tails():
    """Student-t with low df → very heavy tails → same recommendation."""
    rng = np.random.default_rng(0)
    n = 5_000
    y_pred = rng.uniform(-3, 3, size=n)
    # Student-t df=3: excess kurt = 6, undefined for df<=2
    y_true = y_pred + rng.standard_t(df=3, size=n)
    audit = audit_residuals(y_true, y_pred)
    # Could land in either Heavy-tails (Laplace/Student-t) OR Outlier-
    # contaminated bucket depending on the realised tail mass; both
    # recommend MAE/Huber, which is the right user-facing answer.
    assert audit.hypothesis in {"Laplace / Student-t", "Contaminated / outliers"}
    assert ("MAE" in audit.suggested_loss
            or "Huber" in audit.suggested_loss)


def test_contaminated_outliers():
    """Mixture: 95% N(0,1) + 5% N(0, 30) → extreme outliers → Huber."""
    rng = np.random.default_rng(0)
    n = 5_000
    y_pred = rng.uniform(-3, 3, size=n)
    eps_clean = rng.normal(0, 1.0, size=n)
    eps_outliers = rng.normal(0, 30.0, size=n)
    contaminated = rng.uniform(size=n) < 0.05
    eps = np.where(contaminated, eps_outliers, eps_clean)
    y_true = y_pred + eps
    audit = audit_residuals(y_true, y_pred)
    assert audit.hypothesis == "Contaminated / outliers"
    assert "Huber" in audit.suggested_loss
    assert audit.pct_outliers_3sigma > 0.01  # > 1% beyond 3σ
    assert audit.excess_kurt > 5.0


def test_gamma_right_skewed_nonneg():
    """Right-skewed, non-negative residuals + nonneg y → Gamma family."""
    rng = np.random.default_rng(0)
    n = 5_000
    # y_pred is the mean of a Gamma(shape=2, scale=1) target; residuals
    # inherit the right-skew.
    y_pred = np.full(n, 2.0)  # Gamma(2, 1) has mean=2
    y_true = rng.gamma(shape=2.0, scale=1.0, size=n)
    audit = audit_residuals(y_true, y_pred)
    # Gamma should be recognised as right-skewed nonneg.
    assert audit.hypothesis in {"Gamma / Exponential", "LogNormal"}
    assert audit.skew > 0.5
    assert audit.y_true_all_nonneg


def test_lognormal_multiplicative_heteroscedastic():
    """y_true = y_pred * exp(N(0, σ)): heteroscedastic + right-skewed +
    nonneg → LogNormal. The heteroscedasticity test is what
    distinguishes this from Gamma."""
    rng = np.random.default_rng(0)
    n = 8_000
    y_pred = rng.uniform(0.5, 50.0, size=n)
    # Multiplicative noise: residuals scale with y_pred — variance of
    # residuals grows with prediction magnitude.
    eps_log = rng.normal(0, 0.6, size=n)
    y_true = y_pred * np.exp(eps_log)
    audit = audit_residuals(y_true, y_pred)
    assert audit.hetero_significant is True
    assert audit.hetero_spearman > HETERO_SPEARMAN_THRESHOLD
    assert audit.hypothesis == "LogNormal"
    assert "log" in audit.suggested_loss.lower()
    # Should warn about back-transformation bias
    assert "bias" in audit.suggested_loss.lower() or "smearing" in audit.suggested_loss.lower()


def test_homoscedastic_gaussian_says_mse():
    rng = np.random.default_rng(0)
    n = 5_000
    y_pred = rng.uniform(0, 100, size=n)
    y_true = y_pred + rng.normal(0, 1.0, size=n)  # σ doesn't depend on y_pred
    audit = audit_residuals(y_true, y_pred)
    assert audit.hetero_significant is False
    assert "MSE" in audit.suggested_loss


# -----------------------------------------------------------------------------
# Diagnostics edge cases
# -----------------------------------------------------------------------------


def test_biased_model_warns_about_mean():
    """Mean residual much larger than std → "model is biased" warning."""
    rng = np.random.default_rng(0)
    n = 5_000
    y_pred = rng.uniform(-3, 3, size=n)
    y_true = y_pred + rng.normal(2.0, 0.1, size=n)  # constant offset of +2
    audit = audit_residuals(y_true, y_pred)
    assert any("model is biased" in w for w in audit.warnings_)


def test_negative_y_pred_for_nonneg_target_warns():
    """y_true >= 0 but y_pred has negatives → domain-violation warning."""
    rng = np.random.default_rng(0)
    n = 1_000
    y_true = rng.gamma(2.0, 1.0, size=n)  # all positive
    y_pred = rng.normal(2.0, 2.0, size=n)  # may go negative
    if y_pred.min() >= 0:
        # Make sure at least one prediction is negative for this test
        y_pred[0] = -1.0
    audit = audit_residuals(y_true, y_pred)
    assert any("non-negative but y_pred has negative" in w for w in audit.warnings_)


def test_nan_filtering():
    """NaN / inf rows are dropped from the audit; warning lists the count."""
    rng = np.random.default_rng(0)
    n = 1_000
    y_pred = rng.uniform(-3, 3, size=n)
    y_true = y_pred + rng.normal(0, 1.0, size=n)
    y_true[5] = np.nan
    y_true[10] = np.inf
    y_pred[15] = -np.inf
    audit = audit_residuals(y_true, y_pred)
    assert audit.n == 1_000 - 3  # 3 dropped
    assert any("non-finite" in w for w in audit.warnings_)


def test_sample_size_honored():
    """Large input gets sampled to sample_size."""
    rng = np.random.default_rng(0)
    n = 100_000
    y_pred = rng.uniform(-3, 3, size=n)
    y_true = y_pred + rng.normal(0, 1.0, size=n)
    audit = audit_residuals(y_true, y_pred, sample_size=5_000)
    assert audit.sampled is True
    assert audit.n == 5_000
    assert audit.n_total == 100_000


def test_sample_before_finite_filter_on_large_input():
    """audit_residuals must subsample BEFORE the np.isfinite + finite-mask
    boolean-index pass on inputs larger than sample_size. Pre-fix the
    function ran finite_mask on the full N-row input (~4x 16MB allocations
    on 1M f64 rows: the mask itself, n_finite reduction, and the two
    boolean-indexed output arrays). Surfaced as ~0.5-1s of cumtime in the
    1M-row regression suite profile.

    A regression that reorders the operations back to filter-then-sample
    would fail this test by triggering np.isfinite on the full array. We
    pin the new ordering by counting np.isfinite invocations and asserting
    the y_true argument it receives has at most sample_size + slack
    elements.
    """
    rng = np.random.default_rng(0)
    n = 200_000
    sample_size = 5_000
    y_pred = rng.uniform(-3, 3, size=n)
    y_true = y_pred + rng.normal(0, 1.0, size=n)

    captured: dict[str, int] = {"max_size_seen": 0}
    _orig_isfinite = np.isfinite

    def _spy_isfinite(arr, *a, **kw):
        try:
            sz = int(np.asarray(arr).size)
            if sz > captured["max_size_seen"]:
                captured["max_size_seen"] = sz
        except Exception:
            pass
        return _orig_isfinite(arr, *a, **kw)

    import unittest.mock as _mock
    with _mock.patch(
        "mlframe.training.targets.regression_residual_audit.np.isfinite",
        side_effect=_spy_isfinite,
    ):
        audit = audit_residuals(y_true, y_pred, sample_size=sample_size)

    assert audit.sampled is True
    assert audit.n == sample_size
    assert audit.n_total == n
    # The finite filter must run AFTER subsampling -- the largest array
    # np.isfinite ever sees should be at most ``sample_size`` (plus a
    # tiny slack for any incidental probe; tighten to exactly
    # ``sample_size`` if no probes exist).
    assert captured["max_size_seen"] <= sample_size, (
        f"np.isfinite ran on an array of size {captured['max_size_seen']} > "
        f"sample_size={sample_size}; this re-introduces the pre-fix full-N "
        f"finite-mask pass."
    )


def test_sample_size_none_no_subsample():
    rng = np.random.default_rng(0)
    n = 1_000
    y_pred = rng.uniform(-3, 3, size=n)
    y_true = y_pred + rng.normal(0, 1.0, size=n)
    audit = audit_residuals(y_true, y_pred, sample_size=None)
    assert audit.sampled is False
    assert audit.n == n


def test_too_few_observations_raises():
    with pytest.raises(ValueError, match=">= 5 finite observations"):
        audit_residuals(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_mismatched_shapes_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        audit_residuals(np.zeros(10), np.zeros(11))


def test_pandas_series_input():
    rng = np.random.default_rng(0)
    n = 1_000
    y_pred = pd.Series(rng.uniform(-3, 3, size=n))
    y_true = y_pred + rng.normal(0, 1.0, size=n)
    audit = audit_residuals(y_true, y_pred)
    assert audit.n == n


# -----------------------------------------------------------------------------
# Format report
# -----------------------------------------------------------------------------


def test_format_report_includes_all_diagnostic_sections():
    rng = np.random.default_rng(0)
    n = 5_000
    y_pred = rng.uniform(-3, 3, size=n)
    y_true = y_pred + rng.laplace(0, 1.0, size=n)
    audit = audit_residuals(y_true, y_pred)
    report = format_residual_audit_report(audit)
    assert "residual_audit" in report
    assert "moments:" in report
    assert "shape:" in report
    assert "hetero:" in report
    assert "hypothesis:" in report
    assert "suggested:" in report
    assert "Laplace" in report or "Student-t" in report


def test_to_dict_round_trips_to_json():
    import orjson
    rng = np.random.default_rng(0)
    n = 1_000
    y_pred = rng.uniform(-3, 3, size=n)
    y_true = y_pred + rng.normal(0, 1.0, size=n)
    audit = audit_residuals(y_true, y_pred)
    d = audit.to_dict()
    s = orjson.dumps(d).decode()  # must be JSON-safe (no numpy scalars leaking)
    parsed = orjson.loads(s)
    assert parsed["hypothesis"] == audit.hypothesis
    assert parsed["n"] == audit.n


# -----------------------------------------------------------------------------
# Plot integration
# -----------------------------------------------------------------------------


def test_plot_residual_diagnostics_writes_to_axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    n = 2_000
    y_pred = rng.uniform(0, 50, size=n)
    y_true = y_pred * np.exp(rng.normal(0, 0.5, size=n))  # LogNormal pattern
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    audit = plot_residual_diagnostics(y_true, y_pred,
                                      ax_hist=axes[0], ax_resid_vs_pred=axes[1])
    plt.close(fig)
    assert isinstance(audit, ResidualAudit)
    assert audit.hetero_significant is True


def test_plot_passes_when_one_axis_is_none():
    """Caller can request only one panel by passing None to the other."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    y_pred = rng.uniform(-3, 3, size=500)
    y_true = y_pred + rng.normal(0, 1.0, size=500)
    fig, ax = plt.subplots()
    plot_residual_diagnostics(y_true, y_pred, ax_hist=ax, ax_resid_vs_pred=None)
    plt.close(fig)
    # No exception means pass.


def test_plot_handles_too_few_obs_gracefully():
    """Fewer than 5 obs → return None, no crash."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    result = plot_residual_diagnostics(
        np.array([1.0, 2.0]), np.array([1.0, 2.0]),
        ax_hist=ax, ax_resid_vs_pred=None,
    )
    plt.close(fig)
    assert result is None


def test_default_diag_sample_size_constant():
    assert DEFAULT_DIAG_SAMPLE_SIZE == 50_000
    assert HETERO_SPEARMAN_THRESHOLD == 0.30
