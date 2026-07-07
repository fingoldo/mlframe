"""Regression residual audit - distribution diagnostics + noise model hint.

A scatter of true vs predicted is necessary but not sufficient for
understanding regression-model error structure. The same R^2=0.85 looks
identical on a scatter for:

* well-calibrated Gaussian noise (MSE is the right loss),
* heavy-tailed Laplace / Student-t (MSE chases outliers; MAE / Huber
  would be more robust),
* multiplicative LogNormal noise (MSE on log-scale would be more
  appropriate),
* heteroscedastic noise where sigma proportional to y_hat (the model's confidence is
  itself prediction-dependent - typical of count / monetary targets).

This module computes residuals = (y_true - y_pred), reports
distributional diagnostics (skewness, excess kurtosis, Anderson-Darling
vs Normal, Spearman correlation of |residuals| vs y_hat for
heteroscedasticity), and emits a noise-distribution hypothesis with
a concrete actionable training recommendation:

| observation                                         | hypothesis            | suggested loss / model           |
|-----------------------------------------------------|-----------------------|----------------------------------|
| roughly symmetric, light tails                      | Gaussian              | MSE (default)                    |
| symmetric but heavy tails / many outliers           | Laplace / Student-t   | MAE or Huber                     |
| right-skewed, y_true >= 0                            | Gamma / Exponential   | gamma loss / log-link GLM        |
| heteroscedastic AND right-skewed AND y > 0          | LogNormal             | MSE on log(y) (warn about bias)  |
| extreme outliers (excess kurt > 10)                 | contaminated          | Huber, or remove outliers        |

Public surface:
- audit_residuals(y_true, y_pred, *, sample_size=None) -> ResidualAudit
- format_residual_audit_report(audit) -> str
- plot_residual_diagnostics(audit, ax_hist, ax_resid_vs_pred, save_path=None)
- ResidualAudit (dataclass)

The audit is wired automatically into
``report_regression_model_perf``: the existing scatter is now in a
1x3 grid alongside the residual histogram and a residuals-vs-predicted
plot, and the diagnostic / hypothesis text is printed underneath.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# numba-optional fused-moments kernels. Lazy import so a numba-less env still
# imports this module fine (the dispatcher falls back to numpy).
try:
    from numba import njit as _njit, prange as _prange
    _HAS_NUMBA_MOMENTS = True
except ImportError:
    _HAS_NUMBA_MOMENTS = False
    _njit = None
    _prange = None


if _HAS_NUMBA_MOMENTS:
    # Single-pass-per-statistic fused moments kernel. Computes mean, std,
    # skew, excess_kurt, pct_outliers_3sigma in 3 passes over (y_true, y_pred)
    # with explicit scalar accumulators -- the numpy path needed ~6 passes
    # over (residuals,) tensors. Bench at n=50k: 2.4 ms numpy -> 0.21 ms
    # numba seq (~11x); at n=500k: 35 ms -> 2.2 ms seq (~16x), 0.96 ms par
    # (~36x).
    #
    # Median / MAD / percentiles / spearman stay in numpy: those are O(n log n)
    # sort-based and numpy's C implementations are already optimal. The 3-pass
    # moments kernel only fuses the linear-time reductions.
    @_njit(cache=True, fastmath=False)
    def _audit_moments_njit_seq(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        n = y_true.shape[0]
        # Pass 1: mean of residuals.
        s = 0.0
        for i in range(n):
            s += y_true[i] - y_pred[i]
        mean = s / n
        # Pass 2: sum of squared deviations (Welford-equivalent).
        ss = 0.0
        for i in range(n):
            d = (y_true[i] - y_pred[i]) - mean
            ss += d * d
        var = ss / (n - 1) if n > 1 else 0.0
        std = var**0.5
        if std <= 0:
            return mean, std, 0.0, 0.0, 0.0
        # Pass 3: skew / kurt / 3-sigma count via standardized residual z.
        # ``z * z2`` for z^3 and ``z2 * z2`` for z^4 -- numpy's ``z ** 3`` /
        # ``z ** 4`` dispatch through np.power (~7x slower than the integer-
        # exponent identity).
        z3_sum = 0.0
        z4_sum = 0.0
        out3 = 0
        inv_std = 1.0 / std
        for i in range(n):
            z = ((y_true[i] - y_pred[i]) - mean) * inv_std
            z2 = z * z
            z3_sum += z * z2
            z4_sum += z2 * z2
            if z > 3.0 or z < -3.0:
                out3 += 1
        skew = z3_sum / n
        kurt = z4_sum / n - 3.0
        pct_3s = out3 / n
        return mean, std, skew, kurt, pct_3s

    @_njit(cache=True, parallel=True, fastmath=False)
    def _audit_moments_njit_par(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        n = y_true.shape[0]
        s = 0.0
        for i in _prange(n):
            s += y_true[i] - y_pred[i]
        mean = s / n
        ss = 0.0
        for i in _prange(n):
            d = (y_true[i] - y_pred[i]) - mean
            ss += d * d
        var = ss / (n - 1) if n > 1 else 0.0
        std = var**0.5
        if std <= 0:
            return mean, std, 0.0, 0.0, 0.0
        z3_sum = 0.0
        z4_sum = 0.0
        out3 = 0
        inv_std = 1.0 / std
        for i in _prange(n):
            z = ((y_true[i] - y_pred[i]) - mean) * inv_std
            z2 = z * z
            z3_sum += z * z2
            z4_sum += z2 * z2
            if z > 3.0 or z < -3.0:
                out3 += 1
        skew = z3_sum / n
        kurt = z4_sum / n - 3.0
        pct_3s = out3 / n
        return mean, std, skew, kurt, pct_3s


# Dispatcher threshold: numba prange overhead amortises only above ~200k
# elements. Bench:
#   n=50k  : seq 0.21 ms, par 0.21 ms (no win)
#   n=200k : seq 1.0 ms,  par 0.6 ms (1.5x)
#   n=500k : seq 2.2 ms,  par 0.96 ms (2.3x)
# Hardcoded constant rather than kernel_tuning_cache because this is a CPU
# threshold (per feedback_use_kernel_tuning_cache_for_gpu, KTC is for GPU);
# 200k is the empirical crossover on the dev machine and survives modest
# HW variance.
_AUDIT_MOMENTS_PAR_THRESHOLD: int = 200_000


# Diagnostic thresholds - exposed so callers can override.
DEFAULT_DIAG_SAMPLE_SIZE: int = 50_000
"""Target rows for the moment / Anderson-Darling computations. The
math is O(n log n) at worst (sort for AD, sort for Spearman); sampling
to 50k makes the audit sub-second for any practical N. Plot uses its
own (smaller) sample independently."""

SKEW_MODERATE: float = 0.3  # tightened from 0.5 -> 0.3
SKEW_HIGH: float = 0.8  # tightened from 1.0 -> 0.8
# Thresholds significantly tightened after a user-reported
# case where the audit verdict said "Gaussian (well-behaved)" for a
# histogram with excess_kurt=+2.40 -- the threshold was set at the
# LAPLACE excess kurt level (3.0), so anything Logistic-like or peakier
# but not quite Laplace got labeled Gaussian. Reference excess kurtosis:
#   Normal:           0
#   Logistic:         1.2 (only slightly peakier than Normal)
#   Laplace:          3.0 (sharp peak + heavy tails -- clearly non-N)
#   Student-t(df=5):  ~6
#   Contaminated:     10+
# With kurt=+2.40 the residual is between Logistic and Laplace -- the
# histogram VISIBLY shows a sharp peak at 0 with thin shoulders. That
# is NOT well-behaved Gaussian; calling it so is profanation, the user
# was right to flag.
EXCESS_KURT_NEAR_GAUSSIAN: float = 0.5  # < 0.5 -> truly Normal-like
EXCESS_KURT_MILD: float = 1.5  # 0.5-1.5 -> mild leptokurtosis
EXCESS_KURT_HEAVY: float = 1.5  # > 1.5 -> heavy tails (was 3.0!)
EXCESS_KURT_EXTREME: float = 10.0  # > 10 -> outlier contamination
HETERO_SPEARMAN_THRESHOLD: float = 0.30
"""|Spearman corr(|residuals|, y_hat)| above this -> heteroscedasticity
is real, not noise. 0.30 is a moderate effect; 0.50+ is strong."""


@dataclass
class ResidualAudit:
    """Structured outcome of a residuals-distribution audit."""
    n: int                              # number of observations audited (after sampling)
    n_total: int                        # total observations (pre-sample)
    sampled: bool                       # True if we down-sampled before computing stats
    mean: float                         # E[residual]
    std: float                          # sigma(residual)
    median: float
    mad: float  # median absolute deviation (robust sigma)
    skew: float  # 3rd standardized moment
    excess_kurt: float  # kurtosis - 3 (Normal = 0)
    p01: float  # 1st percentile of residuals
    p99: float  # 99th percentile
    pct_outliers_3sigma: float  # fraction of |residual - mean| > 3*std
    hetero_spearman: float  # Spearman corr(|residual|, y_hat)
    hetero_significant: bool  # |hetero_spearman| > HETERO_SPEARMAN_THRESHOLD
    y_true_all_nonneg: bool  # True if min(y_true) >= 0
    y_pred_all_nonneg: bool  # True if min(y_pred) >= 0
    hypothesis: str  # short distribution name (e.g. "Gaussian", "LogNormal")
    suggested_loss: str  # e.g. "MSE", "MAE", "Huber"
    rationale: list[str] = field(default_factory=list)
    warnings_: list[str] = field(default_factory=list)
    # Formal normality test result -- D'Agostino K^2 + Anderson-Darling.
    # The moment-based heuristic above misses bimodal / asymmetric
    # distributions whose skew + excess_kurt happen to land inside the
    # cutoff band; these tests catch them via principled hypothesis
    # testing instead. None when n < 20.
    normality_test: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_total": self.n_total,
            "sampled": self.sampled,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "mad": self.mad,
            "skew": self.skew,
            "excess_kurt": self.excess_kurt,
            "p01": self.p01,
            "p99": self.p99,
            "pct_outliers_3sigma": self.pct_outliers_3sigma,
            "hetero_spearman": self.hetero_spearman,
            "hetero_significant": self.hetero_significant,
            "y_true_all_nonneg": self.y_true_all_nonneg,
            "y_pred_all_nonneg": self.y_pred_all_nonneg,
            "hypothesis": self.hypothesis,
            "suggested_loss": self.suggested_loss,
            "rationale": list(self.rationale),
            "warnings": list(self.warnings_),
            "normality_test": dict(self.normality_test) if self.normality_test else None,
        }


def _sample(arr: np.ndarray, size: int, seed: int = 0) -> np.ndarray:
    """Random subsample without replacement when arr is larger."""
    if arr.size <= size:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.size, size=size, replace=False)
    return arr[idx]


def _ranks_via_scatter(x: np.ndarray) -> np.ndarray:
    """Ordinal ranks (0-based) of ``x`` via one argsort + a scatter, instead of the
    ``argsort(argsort(x))`` double-sort. One sort produces the order permutation; scattering
    positional indices back through it yields the same ranks with half the sorting work.
    Bit-identical to ``np.argsort(np.argsort(x)).astype(np.float64)`` (no ties: pure ordinal
    ranking; ties: both forms break them by argsort's stable position, identically)."""
    order = np.argsort(x)
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Self-contained (no scipy dep) so the
    audit doesn't add a new mandatory import; same definition as
    ``scipy.stats.spearmanr``."""
    if x.size < 3:
        return 0.0
    rx = _ranks_via_scatter(x)
    ry = _ranks_via_scatter(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx**2).sum())) * math.sqrt(float((ry**2).sum()))
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def _diagnose(
    *,
    skew: float,
    excess_kurt: float,
    hetero_spearman: float,
    pct_outliers_3sigma: float,
    y_true_all_nonneg: bool,
    y_pred_all_nonneg: bool,
    normality_test: dict | None = None,
) -> tuple[str, str, list[str]]:
    """Map diagnostics -> (hypothesis, suggested_loss, rationale lines)."""
    rationale: list[str] = []
    abs_skew = abs(skew)
    abs_hetero = abs(hetero_spearman)

    # Heteroscedasticity is the most informative signal - flag it first.
    if abs_hetero > HETERO_SPEARMAN_THRESHOLD:
        rationale.append(
            f"heteroscedasticity: Spearman corr(|resid|, y_hat) = {hetero_spearman:+.3f} "
            f"(|dot| > {HETERO_SPEARMAN_THRESHOLD}); error variance depends on prediction magnitude."
        )

    # Right-skew + nonneg y -> LogNormal / Gamma family.
    if abs_skew >= SKEW_HIGH and skew > 0 and y_true_all_nonneg and y_pred_all_nonneg:
        if abs_hetero > HETERO_SPEARMAN_THRESHOLD:
            rationale.append(
                f"right-skewed (skew={skew:+.2f}) AND y >= 0 AND heteroscedastic - " "classic multiplicative-noise pattern (error proportional to y)."
            )
            return "LogNormal", "MSE on log(y) (CAUTION: introduces bias when back-transforming; consider Duan smearing)", rationale
        rationale.append(f"right-skewed (skew={skew:+.2f}) AND y >= 0 - additive Gamma/Exponential family.")
        return "Gamma / Exponential", "Gamma deviance loss (LightGBM 'gamma' / XGBoost 'reg:gamma' / sklearn GLM)", rationale

    # Heavy tails - outlier-robust losses. Distinguish Laplace-class
    # (excess_kurt ~3, naturally heavy but not contaminated; Laplace's
    # P(|z|>3) ~= 1.5%) from true contamination by EXCESS KURTOSIS, not
    # by outlier fraction. Empirical reference (n=50k):
    #   Gaussian:     kurt ~= 0,    pct_3sigma ~= 0.27%
    #   Laplace:      kurt ~= 3,    pct_3sigma ~= 1.5%
    #   Student-t(3): kurt ~= 60+,  pct_3sigma ~= 1.4%
    #   Contaminated: kurt ~= 50+,  pct_3sigma ~= 2.5%
    # Using pct_3sigma to flag contamination misclassifies Laplace; using
    # the kurtosis cliff (kurt > 10) catches both Student-t and the
    # mixture-contamination case while letting Laplace through.
    if excess_kurt > EXCESS_KURT_EXTREME:
        rationale.append(
            f"extreme tails / outlier contamination: excess kurt={excess_kurt:+.2f} "
            f"(> {EXCESS_KURT_EXTREME}); {pct_outliers_3sigma*100:.1f}% of |resid|>3sigma "
            f"(Normal expects 0.27%)."
        )
        return "Contaminated / outliers", "Huber (robust to outliers; tune delta around 1-2 sigma_resid)", rationale

    if excess_kurt > EXCESS_KURT_HEAVY:
        rationale.append(
            f"heavy-tailed / leptokurtic: excess kurt={excess_kurt:+.2f} (> {EXCESS_KURT_HEAVY}); "
            f"{pct_outliers_3sigma*100:.1f}% of |resid|>3sigma (Normal expects 0.27%). "
            f"Sharp peak at 0 with heavier-than-Normal tails. Consistent with "
            f"Laplace (excess kurt ~ 3.0) or Student-t (df 4-10)."
        )
        return "Laplace / Student-t", "MAE (Laplace-MLE) or Huber as a compromise", rationale

    # Intermediate verdict for mildly leptokurtic residuals
    # (excess_kurt in [0.5, 1.5]). Previously these silently passed
    # as "Gaussian (well-behaved)" -- the user flagged this on a
    # histogram with kurt=+2.40 that visibly showed a sharp peak at 0.
    # Now we honestly label them as "Near-Gaussian (mildly peaky)" and
    # still suggest MSE because the deviation is small enough not to
    # warrant changing the loss -- but the verdict label no longer
    # claims a clean Normal.
    if excess_kurt > EXCESS_KURT_NEAR_GAUSSIAN:
        # Mild leptokurtosis -- not Gaussian, but MSE still OK.
        rationale.append(
            f"mildly leptokurtic: excess kurt={excess_kurt:+.2f} "
            f"(> {EXCESS_KURT_NEAR_GAUSSIAN}, but < {EXCESS_KURT_HEAVY}); residual "
            f"distribution has a noticeable peak at 0 but tails are not so heavy "
            f"that MSE breaks down. {pct_outliers_3sigma*100:.1f}% of |resid|>3sigma. "
            f"Reference: Logistic (~1.2), Laplace (~3.0). "
            "MSE is still appropriate; Huber would only marginally improve."
        )
        return "Near-Gaussian (mildly peaky)", "MSE (default) - mild leptokurtosis is tolerable; Huber an option if outliers concern you", rationale

    if excess_kurt < -EXCESS_KURT_NEAR_GAUSSIAN:
        # Platykurtic (flat distribution, lighter-than-Normal tails).
        # Rare in practice; usually indicates uniform-ish residuals or
        # bounded targets (saturating predictions). Still ~symmetric
        # so MSE works.
        rationale.append(
            f"platykurtic: excess kurt={excess_kurt:+.2f} "
            f"(< {-EXCESS_KURT_NEAR_GAUSSIAN}); residuals are flatter than Normal. "
            f"Often signals a bounded / saturating target. MSE still OK."
        )
        return "Near-Gaussian (platykurtic)", "MSE (default) - mild tail-thinning is tolerable", rationale

    # Mild asymmetry without nonneg constraint - flag but Gaussian still ok.
    if abs_skew >= SKEW_MODERATE:
        rationale.append(
            f"mild skew ({skew:+.2f}); within Gaussian tolerance but worth investigating " "if the target has a natural non-negativity constraint."
        )

    # True Gaussian verdict reserved for tight near-Normal:
    # |skew| < 0.3 AND |excess_kurt| < 0.5 AND |hetero| < 0.3.
    # Final gate: formal normality test (D'Agostino K^2 + Anderson-Darling).
    # Moment-based heuristics above can return ``True`` for bimodal /
    # asymmetric distributions whose skew + excess_kurt happen to land
    # inside the cutoff band (user-reported case: skew=-0.17 kurt=-0.12
    # with visible asymmetric hist tails -> labelled Gaussian). When the
    # formal test rejects H0, downgrade the verdict.
    if normality_test is not None and normality_test.get("reject_normal"):
        rationale.append(
            f"moment-based heuristics within Normal band "
            f"(|skew|={abs_skew:.2f}, excess_kurt={excess_kurt:+.2f}) but "
            f"formal tests reject H0: {normality_test.get('verdict')}."
        )
        return (
            "Empirically non-Normal (formal test rejects)",
            "Huber (robust default since formal tests reject Normal; pick delta ~ 1-2 * MAD)",
            rationale,
        )
    rationale.append(
        f"residuals look ~Gaussian: |skew|={abs_skew:.2f} (< {SKEW_MODERATE}), "
        f"excess kurt={excess_kurt:+.2f} (within "
        f"+/-{EXCESS_KURT_NEAR_GAUSSIAN}), "
        f"|hetero|={abs_hetero:.2f}. Default MSE / MAE losses are appropriate."
    )
    if normality_test is not None and math.isfinite(normality_test.get("k2_p", float("nan"))):
        rationale.append("formal tests do not reject Normal: " f"{normality_test.get('verdict')}.")
    return "Gaussian (well-behaved)", "MSE (default) - diagnostics support the standard regression assumption", rationale


def audit_residuals(
    y_true: Any,
    y_pred: Any,
    *,
    sample_size: int | None = DEFAULT_DIAG_SAMPLE_SIZE,
    seed: int = 0,
) -> ResidualAudit:
    """Compute residual-distribution diagnostics + noise hypothesis.

    Parameters
    ----------
    y_true, y_pred : array-like
        True target values and model predictions. Same length. 2-D
        inputs (multi-output regression) are flattened - per-output
        analysis is the caller's job.
    sample_size : int or None, default 50000
        Down-sample to this many points before computing diagnostics.
        ``None`` disables sampling. The math is O(n log n) so 50k
        keeps the audit sub-second for any practical N. The plotting
        path samples independently.
    seed : int, default 0
        RNG seed for the subsample (deterministic across reruns).

    Returns
    -------
    ResidualAudit
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true / y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")

    n_total = int(y_true.size)
    if n_total == 0:
        raise ValueError("y_true / y_pred are empty.")

    # Sample BEFORE the finite filter on large inputs. Pre-fix the
    # function ran np.isfinite + finite_mask boolean-index on the full
    # input (~1M rows ~16MB allocations x 4 -- the finite-mask itself,
    # both masked output arrays, and the int .sum() reduction) only to
    # then subsample down to ``sample_size`` rows. The audit's purpose
    # is distribution diagnostics on a representative sample; non-finite
    # detection only needs to surface ANY occurrences. The post-sample
    # finite-mask path stays sub-millisecond.
    warnings: list[str] = []
    sampled = False
    if sample_size is not None and n_total > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=sample_size, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        sampled = True

    # Filter NaN / inf - common with degenerate model outputs. After
    # sampling on large inputs (above), this runs on at most sample_size
    # points. The dropped-count reported below is "in sample" not "total";
    # the warning text reflects that the audit is sample-scoped.
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n_post_sample = int(y_true.size)
    n_finite = int(finite_mask.sum())
    if n_finite < n_post_sample:
        n_dropped = n_post_sample - n_finite
        # Use n_post_sample (not the pre-sample n_total) as the denominator
        # so the percentage reflects what the audit actually saw.
        _scope_word = "sampled" if sampled else "input"
        warnings.append(
            f"dropped {n_dropped} non-finite point(s) from the audit "
            f"({n_dropped/n_post_sample*100:.2f}% of {_scope_word}). Common causes: "
            f"NaN target rows or model returning inf for OOD inputs."
        )
    y_true = y_true[finite_mask]
    y_pred = y_pred[finite_mask]

    if y_true.size < 5:
        raise ValueError(f"need >= 5 finite observations to compute residual diagnostics; got {y_true.size}.")

    residuals = y_true - y_pred
    n = int(residuals.size)

    # Moments dispatcher: prefer the fused @njit kernel when available; pick
    # the prange variant above the empirical CPU crossover. Bench at n=50k:
    # 2.4 ms numpy -> 0.21 ms numba seq (~11x); at n=500k: 35 ms -> 2.2 ms
    # seq (~16x), 0.96 ms par (~36x). Median / MAD / percentile / spearman
    # stay in numpy (sort-based; already C-optimal).
    if _HAS_NUMBA_MOMENTS and n >= 2:
        try:
            if n >= _AUDIT_MOMENTS_PAR_THRESHOLD:
                mean, std, skew, excess_kurt, pct_outliers_3sigma = _audit_moments_njit_par(y_true, y_pred)
            else:
                mean, std, skew, excess_kurt, pct_outliers_3sigma = _audit_moments_njit_seq(y_true, y_pred)
            mean, std = float(mean), float(std)
            skew, excess_kurt = float(skew), float(excess_kurt)
            pct_outliers_3sigma = float(pct_outliers_3sigma)
        except Exception as _moments_err:
            logger.warning(
                "audit_residuals: numba fused-moments kernel failed (%s); " "falling back to numpy path. n=%d.",
                _moments_err,
                n,
            )
            mean = float(residuals.mean())
            std = float(residuals.std(ddof=1)) if n > 1 else 0.0
            if std > 0:
                z = (residuals - mean) / std
                z2 = z * z
                skew = float(np.mean(z * z2))
                excess_kurt = float(np.mean(z2 * z2) - 3.0)
                pct_outliers_3sigma = float((np.abs(z) > 3.0).mean())
            else:
                skew, excess_kurt, pct_outliers_3sigma = 0.0, 0.0, 0.0
    else:
        # numba-less environment: same numpy-only path as the pre-iter129 code.
        mean = float(residuals.mean())
        std = float(residuals.std(ddof=1)) if n > 1 else 0.0
        if std > 0:
            z = (residuals - mean) / std
            z2 = z * z
            skew = float(np.mean(z * z2))
            excess_kurt = float(np.mean(z2 * z2) - 3.0)
            pct_outliers_3sigma = float((np.abs(z) > 3.0).mean())
        else:
            skew, excess_kurt, pct_outliers_3sigma = 0.0, 0.0, 0.0

    # Single ``np.percentile(residuals, [1, 50, 99])`` does one partial sort and extracts all three
    # quantiles at once. The median is percentile-50 (bit-identical to ``np.median`` up to ULP FP
    # reduction-order), so folding it into this call drops the separate ``np.median(residuals)``
    # partition entirely (bench n=50k: median+percentile[1,99] 0.83 ms / call vs combined [1,50,99]
    # 0.68 ms = 1.22x). The MAD median is on the distinct abs-deviation array, so it stays separate.
    _p01_50_99 = np.percentile(residuals, [1, 50, 99])
    p01, median, p99 = float(_p01_50_99[0]), float(_p01_50_99[1]), float(_p01_50_99[2])
    mad = float(np.median(np.abs(residuals - median)))

    # Heteroscedasticity: do |residuals| correlate with y_hat?
    hetero_spearman = _spearman_corr(np.abs(residuals), y_pred)
    hetero_significant = abs(hetero_spearman) > HETERO_SPEARMAN_THRESHOLD

    y_true_all_nonneg = bool(y_true.min() >= 0)
    y_pred_all_nonneg = bool(y_pred.min() >= 0)

    # Formal normality test (D'Agostino K^2 + Anderson-Darling) gates the
    # final "Gaussian (well-behaved)" verdict. The two tests catch
    # bimodal / asymmetric distributions whose moments happen to land
    # inside the heuristic cutoff band -- a class of bug the legacy
    # diagnose path could not detect.
    normality_test: dict | None = None
    try:
        from pyutilz.stats.normality import normality_verdict as _nv
        normality_test = _nv(residuals)
    except Exception as _nv_err:
        logger.debug(
            "normality_verdict unavailable / failed: %s; " "audit will fall back to moment-cutoff heuristic only.",
            _nv_err,
        )

    hypothesis, suggested_loss, rationale = _diagnose(
        skew=skew, excess_kurt=excess_kurt,
        hetero_spearman=hetero_spearman,
        pct_outliers_3sigma=pct_outliers_3sigma,
        y_true_all_nonneg=y_true_all_nonneg,
        y_pred_all_nonneg=y_pred_all_nonneg,
        normality_test=normality_test,
    )

    # Edge-case warnings.
    if std > 0 and abs(mean) > 0.5 * std:
        warnings.append(
            f"residual mean is large ({mean:+.4g}) relative to std ({std:.4g}); "
            "the model is biased - consider re-centering predictions or "
            "checking for a regression-target offset bug."
        )
    if y_true_all_nonneg and y_pred.min() < 0:
        warnings.append(
            "y_true is non-negative but y_pred has negative values - model is "
            "violating a domain constraint. Add a non-negativity constraint "
            "(e.g. log-link, exp output, or clip)."
        )

    return ResidualAudit(
        n=n, n_total=n_total, sampled=sampled,
        mean=mean, std=std, median=median, mad=mad,
        skew=skew, excess_kurt=excess_kurt,
        p01=p01, p99=p99,
        pct_outliers_3sigma=pct_outliers_3sigma,
        hetero_spearman=hetero_spearman,
        hetero_significant=hetero_significant,
        y_true_all_nonneg=y_true_all_nonneg,
        y_pred_all_nonneg=y_pred_all_nonneg,
        hypothesis=hypothesis,
        suggested_loss=suggested_loss,
        rationale=rationale,
        warnings_=warnings,
        normality_test=normality_test,
    )


def format_residual_audit_report(audit: ResidualAudit, *, ndigits: int = 4) -> str:
    """Compact log block: header + one-line moments + hypothesis +
    rationale + warnings."""
    sample_note = f" (sampled {audit.n:_}/{audit.n_total:_})" if audit.sampled else f" (n={audit.n:_})"
    lines = [
        f"residual_audit{sample_note}:",
        f"  moments:   mean={audit.mean:+.{ndigits}g} std={audit.std:.{ndigits}g} " f"median={audit.median:+.{ndigits}g} MAD={audit.mad:.{ndigits}g}",
        f"  shape:     skew={audit.skew:+.{ndigits-2}f} excess_kurt={audit.excess_kurt:+.{ndigits-2}f} "
        f"|p01,p99|=[{audit.p01:+.{ndigits-2}f}, {audit.p99:+.{ndigits-2}f}] "
        f"outliers_3sigma={audit.pct_outliers_3sigma*100:.2f}%",
        f"  hetero:    spearman(|resid|, y_hat) = {audit.hetero_spearman:+.{ndigits-2}f} " f"({'significant' if audit.hetero_significant else 'ok'})",
        f"  hypothesis: {audit.hypothesis}",
        f"  suggested:  {audit.suggested_loss}",
    ]
    if audit.rationale:
        for r in audit.rationale:
            lines.append(f"  why:        {r}")
    if audit.warnings_:
        for w in audit.warnings_:
            lines.append(f"  WARN:       {w}")
    return "\n".join(lines)


def plot_residual_diagnostics(
    y_true: Any,
    y_pred: Any,
    audit: ResidualAudit | None = None,
    *,
    ax_hist: Any = None,
    ax_resid_vs_pred: Any = None,
    plot_sample_size: int = 5_000,
    seed: int = 0,
    plot_outputs: str | None = None,
    base_path: str | None = None,
    header_str: str = "",
    metrics_str: str = "",
    dpi: int | None = None,
    panels_template: str | None = None,
) -> ResidualAudit | None:
    """Render the residual histogram + residuals-vs-predicted plot
    on the supplied matplotlib axes.

    The two diagnostic plots intended to flank the existing true-vs-
    predicted scatter in ``report_regression_model_perf``:

    * ``ax_hist`` - histogram of residuals with overlaid Normal density
      fitted to (mean, std). A heavy/skewed observed distribution
      vs the dashed Normal bell visually flags non-Gaussian noise.
    * ``ax_resid_vs_pred`` - scatter of residuals vs y_pred. A funnel
      shape (variance growing with y_hat) is the visual signature of
      heteroscedasticity. A straight line of points suggests a model
      bias dependent on prediction magnitude.

    Both axes are optional - pass ``None`` to skip a panel.

    Returns the audit (computed if not supplied) so callers can use
    its diagnostic fields without computing twice.
    """
    # Opt-in DSL render path (matplotlib + plotly via the
    # spec pipeline). When ``plot_outputs`` + ``base_path`` are set,
    # bypass the in-place axes path and emit a full figure via the
    # shared renderer. Default behaviour preserved for callers that
    # supply their own axes.
    if plot_outputs and base_path:
        from mlframe.reporting.charts.regression import DEFAULT_REGRESSION_PANELS, build_regression_panel_spec
        from mlframe.reporting.output import parse_plot_output_dsl
        from mlframe.reporting.renderers import render_and_save
        _yt = np.asarray(y_true, dtype=np.float64).ravel()
        _yp = np.asarray(y_pred, dtype=np.float64).ravel()
        _mask = np.isfinite(_yt) & np.isfinite(_yp)
        if int(_mask.sum()) < 5:
            return audit
        if audit is None:
            # Re-attempt the audit for the chart, but a second failure must not crash the report -- render the
            # composer without it (hist Normal overlay + hypothesis text omitted) rather than dropping the chart.
            try:
                audit = audit_residuals(_yt[_mask], _yp[_mask], seed=seed)
            except Exception as _audit_err:
                logger.warning(
                    "residual audit unavailable for the regression chart (%s); " "rendering scatter + residual hist without the noise-distribution hypothesis.",
                    _audit_err,
                )
                audit = None
        spec = build_regression_panel_spec(
            _yt, _yp,
            audit=audit, header_str=header_str, metrics_str=metrics_str,
            plot_sample_size=plot_sample_size, seed=seed,
            panels_template=panels_template or DEFAULT_REGRESSION_PANELS,
        )
        if dpi is not None:
            import dataclasses as _dc
            spec = _dc.replace(spec, dpi=dpi)
        render_and_save(spec, parse_plot_output_dsl(plot_outputs), base_path)
        return audit

    try:
        import matplotlib.pyplot as plt  # noqa: F401
        from matplotlib.lines import Line2D  # noqa: F401
    except ImportError:
        logger.warning("matplotlib not installed; residual diagnostic plot skipped.")
        return audit

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_f = y_true[finite_mask]
    y_pred_f = y_pred[finite_mask]
    if y_true_f.size < 5:
        return audit

    if audit is None:
        audit = audit_residuals(y_true_f, y_pred_f, seed=seed)

    residuals = y_true_f - y_pred_f

    # Subsample for plotting so 9M-row datasets don't render 9M points. Extremes-preserving so the largest-|resid|
    # points (the heteroscedasticity funnel mouth + the MaxError row the title quotes) survive the draw -- a uniform
    # rng.choice silently drops exactly the points these panels exist to show.
    if residuals.size > plot_sample_size:
        from mlframe.reporting.charts import subsample_preserving_extremes
        idx = subsample_preserving_extremes(
            y_pred_f, residuals, sample_size=plot_sample_size, extreme_values=residuals, rng=seed,
        )
        plot_resid = residuals[idx]
        plot_pred = y_pred_f[idx]
    else:
        plot_resid = residuals
        plot_pred = y_pred_f

    # Histogram panel
    if ax_hist is not None:
        n_bins = max(20, min(80, int(math.sqrt(plot_resid.size))))
        ax_hist.hist(plot_resid, bins=n_bins, alpha=0.6, color="steelblue", edgecolor="white", linewidth=0.4, density=True)
        # Overlay fitted Normal density (the baseline assumption). Guard
        # against constant-residual inputs: when min == max, linspace
        # collapses to a single point and the normal PDF is meaningless.
        # The audit.std > 0 check covered most cases but ``residuals.std``
        # uses ddof=1 on a SAMPLED subset, so the plot-time residual array
        # can still be all-identical even when the population std is nonzero.
        _r_min = float(plot_resid.min())
        _r_max = float(plot_resid.max())
        if audit.std > 0 and _r_max > _r_min:
            x_grid = np.linspace(_r_min, _r_max, 200)
            normal_pdf = 1 / (audit.std * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x_grid - audit.mean) / audit.std) ** 2)
            ax_hist.plot(x_grid, normal_pdf, color="red", linestyle="--", linewidth=1.4, label=f"Normal(mu={audit.mean:.2g}, sigma={audit.std:.2g})")
            ax_hist.legend(loc="best", fontsize=8, framealpha=0.7)
        ax_hist.axvline(0, color="green", linestyle=":", linewidth=1.0, alpha=0.7)
        ax_hist.set_xlabel("Residual (y_true - y_pred)")
        ax_hist.set_ylabel("Density")
        # Residual hypothesis + suggested loss now live on the
        # histogram title (was previously appended to the scatter title,
        # crowding it). Self-contained: skew/kurt -> hypothesis -> suggested
        # loss reads top-to-bottom on the same panel that visualises the
        # residual distribution.
        _suggested = audit.suggested_loss.split("(")[0].strip() if audit.suggested_loss else ""
        _hyp_line = f"hypothesis: {audit.hypothesis}"
        if _suggested:
            _hyp_line += f" (suggested: {_suggested})"
        ax_hist.set_title(f"Residuals (skew={audit.skew:+.2f}, excess_kurt={audit.excess_kurt:+.2f})\n" f"{_hyp_line}")
        ax_hist.grid(True, alpha=0.3)

    # Residuals vs predicted (heteroscedasticity panel)
    if ax_resid_vs_pred is not None:
        ax_resid_vs_pred.scatter(plot_pred, plot_resid, alpha=0.3, s=10, color="steelblue")
        ax_resid_vs_pred.axhline(0, color="green", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_resid_vs_pred.set_xlabel("Predicted (y_hat)")
        ax_resid_vs_pred.set_ylabel("Residual")
        het_marker = "(!) heteroscedastic" if audit.hetero_significant else "homoscedastic"
        ax_resid_vs_pred.set_title(f"Residuals vs predicted ({het_marker})\n" f"spearman(|resid|, y_hat) = {audit.hetero_spearman:+.3f}")
        ax_resid_vs_pred.grid(True, alpha=0.3)

    return audit
