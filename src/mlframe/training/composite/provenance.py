"""CompositeProvenance dataclass + report-to-markdown helper. Production-grade metadata for one composite-target spec: human-readable formula, fitted params, baseline metrics, ensemble weight, selection-path audit trail. Split out of composite.py for clean separation between discovery internals and stakeholder-facing audit artefacts; composite.py re-exports every symbol below at its bottom for full back-compat."""


from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple,
)

import numpy as np

if TYPE_CHECKING:
    from .spec import CompositeSpec  # used as a forward annotation in CompositeProvenance.from_spec; importing at runtime is unnecessary and risks circular load.

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# CompositeProvenance + report helpers
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CompositeProvenance:
    """Production-grade metadata for one composite-target spec.

    Carries everything a downstream consumer needs to (a) understand
    *why* this composite was selected, (b) reproduce the inverse at
    serving time without consulting source code, and (c) audit the
    selection trail months later when the original DS has moved on
    and stakeholders ask "what does this number mean".

    Why this exists. Without provenance, ``y__linear_residual__lag1``
    is an opaque key. With provenance, the same key reads as
    "predicts residual after subtracting fitted alpha=0.952 of the
    previous-period lag1 value (R^2_train = 0.91), selected
    because removing the linear contribution exposed a residual MI
    of 0.165 against the remaining features".

    Convert to dict via :meth:`to_dict` (JSON-serialisable) or to a
    stakeholder-ready paragraph via :meth:`to_audit_trail`.
    """

    # Identity
    composite_id: str
    discovery_timestamp: str  # ISO 8601, no datetime obj to keep dict-pickle clean
    discovery_random_state: int | None

    # Origin
    name: str  # canonical spec name (matches CompositeSpec.name); the legacy target__transform__base key no longer matches it.
    target_col: str
    transform_name: str
    base_column: str

    # Human-readable formula
    forward_formula_human: str
    inverse_formula_human: str
    stakeholder_description: str

    # Fitted parameters (reproducible inversion)
    fitted_params: dict[str, Any]

    # Justification numbers
    mi_y: float
    mi_t: float
    mi_gain: float
    valid_domain_frac: float
    n_train_rows: int

    # Multi-base extension; empty tuple = single-base spec (base_column authoritative).
    extra_base_columns: tuple[str, ...] = ()

    # Optional: weight in cross-target ensemble (filled at integration time).
    ensemble_weight: float | None = None
    ensemble_strategy: str | None = None

    @classmethod
    def from_spec(
        cls,
        spec: CompositeSpec,
        random_state: int | None,
        *,
        ensemble_weight: float | None = None,
        ensemble_strategy: str | None = None,
    ) -> CompositeProvenance:
        """Construct provenance from a discovered :class:`CompositeSpec`.

        Pulls human-readable formula text from the registered transform
        and the spec's fitted parameters, plus a deterministic
        ``composite_id`` (sha256 prefix) so the same spec recurring in
        future runs is recognisable.
        """
        # Stable id derived from (target, transform, base, fitted_params).
        canonical = json.dumps(
            {
                "target_col": spec.target_col,
                "transform_name": spec.transform_name,
                "base_column": spec.base_column,
                "fitted_params": spec.fitted_params,
            },
            sort_keys=True,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o),
        )
        composite_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

        forward, inverse, description = _format_transform_formulas(
            transform_name=spec.transform_name,
            base_column=spec.base_column,
            target_col=spec.target_col,
            fitted_params=spec.fitted_params,
        )

        return cls(
            composite_id=composite_id,
            discovery_timestamp=datetime.now(timezone.utc).isoformat(),
            discovery_random_state=random_state,
            name=spec.name,
            target_col=spec.target_col,
            transform_name=spec.transform_name,
            base_column=spec.base_column,
            extra_base_columns=tuple(spec.extra_base_columns),
            forward_formula_human=forward,
            inverse_formula_human=inverse,
            stakeholder_description=description,
            fitted_params=dict(spec.fitted_params),
            mi_y=spec.mi_y,
            mi_t=spec.mi_t,
            mi_gain=spec.mi_gain,
            valid_domain_frac=spec.valid_domain_frac,
            n_train_rows=spec.n_train_rows,
            ensemble_weight=ensemble_weight,
            ensemble_strategy=ensemble_strategy,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable plain dict (for ``metadata`` storage)."""
        return {
            "composite_id": self.composite_id,
            "discovery_timestamp": self.discovery_timestamp,
            "discovery_random_state": self.discovery_random_state,
            "name": self.name,
            "target_col": self.target_col,
            "transform_name": self.transform_name,
            "base_column": self.base_column,
            "extra_base_columns": list(self.extra_base_columns),
            "forward_formula_human": self.forward_formula_human,
            "inverse_formula_human": self.inverse_formula_human,
            "stakeholder_description": self.stakeholder_description,
            "fitted_params": dict(self.fitted_params),
            "mi_y": float(self.mi_y),
            "mi_t": float(self.mi_t),
            "mi_gain": float(self.mi_gain),
            "valid_domain_frac": float(self.valid_domain_frac),
            "n_train_rows": int(self.n_train_rows),
            "ensemble_weight": (None if self.ensemble_weight is None
                                else float(self.ensemble_weight)),
            "ensemble_strategy": self.ensemble_strategy,
        }

    def to_audit_trail(self) -> str:
        """Single-paragraph human-readable summary suitable for a Slack
        message or a code-review comment. Quotes the exact numbers
        that justified inclusion so the reader can cross-check."""
        ensemble_clause = ""
        if self.ensemble_weight is not None and self.ensemble_strategy is not None:
            ensemble_clause = (
                f" In the cross-target {self.ensemble_strategy} ensemble it "
                f"received weight {self.ensemble_weight:.3f}."
            )
        return (
            f"Composite '{self.name}' "
            f"(id={self.composite_id}) was discovered using "
            f"random_state={'unspecified' if self.discovery_random_state is None else self.discovery_random_state} on "
            f"{self.n_train_rows} train rows ({self.valid_domain_frac:.1%} of valid domain). "
            f"It was selected because MI(T, X\\base)={self.mi_t:.4f} vs "
            f"MI(y, X\\base)={self.mi_y:.4f} (gain={self.mi_gain:+.4f}), "
            f"meaning the transform '{self.stakeholder_description}' exposed "
            f"residual structure the remaining features can predict more easily. "
            f"Forward: {self.forward_formula_human}. "
            f"Inverse: {self.inverse_formula_human}.{ensemble_clause}"
        )


# Friendly transform-name-to-paragraph table.
#
# Covers EVERY registered transform (audit S12: the old 4-entry table left
# ~32 of ~36 registered transforms rendering a generic stub, contradicting
# the "reproduce the inverse at serving time without consulting source
# code" promise in :class:`CompositeProvenance`). A meta-test
# (``test_provenance_formula_coverage_future``) pins this table + the
# builder table below against the live ``TRANSFORMS_REGISTRY`` so a newly
# registered transform that forgets a formula entry fails fast instead of
# silently regressing to the opaque generic stub.
_TRANSFORM_DESCRIPTIONS: dict[str, str] = {
    "diff": ("predicts the residual after subtracting the base feature "
             "from the target"),
    "additive_residual": ("predicts the residual after subtracting the base "
                          "feature and a fitted constant offset (alpha fixed at 1)"),
    "median_residual": ("predicts the residual after subtracting the per-bin "
                        "median of the target within quantile bins of the base"),
    "y_quantile_clip": ("clips the target to its train 0.5%/99.5% quantiles to "
                       "bound the downstream model's effective range"),
    "ratio": ("predicts the multiplicative factor relating target to "
              "base feature"),
    "logratio": ("predicts the log-ratio of target to base feature, "
                 "stabilising heavy-tail distributions"),
    "linear_residual": ("predicts the residual after subtracting a "
                        "fitted linear contribution of the base feature"),
    "linear_residual_robust": ("predicts the residual after subtracting an "
                              "outlier-robust (trimmed-LS) linear contribution of the base"),
    "linear_residual_multi": ("predicts the residual after subtracting a fitted "
                             "linear combination of several base features"),
    "linear_residual_grouped": ("predicts the residual after subtracting a "
                               "per-group linear contribution of the base (James-Stein shrunk to global)"),
    "quantile_residual": ("predicts the heteroscedasticity-standardised residual: "
                         "per-bin median removed then divided by per-bin IQR"),
    "monotonic_residual": ("predicts the residual after subtracting a fitted "
                          "monotone (PCHIP) function of the base feature"),
    "ewma_residual": ("predicts the residual after subtracting an exponentially-"
                     "weighted moving average of the base feature"),
    "rolling_quantile_ratio": ("predicts the ratio of target to a rolling median "
                              "of the base feature (local multiplicative level)"),
    "frac_diff": ("predicts the fractionally-differenced target (long-memory-"
                 "preserving stationarising transform)"),
    "cbrt_y": ("predicts the signed cube root of the target, compressing heavy "
              "tails without breaking sign"),
    "log_y": ("predicts the shifted log of the target, compressing right-skew"),
    "yeo_johnson_y": ("predicts the Yeo-Johnson power transform of the target "
                     "(lambda fitted by MLE), normalising mixed-sign targets"),
    "quantile_normal_y": ("predicts the target mapped through its empirical CDF "
                         "into a standard Normal"),
    "chain_linres_cbrt": ("predicts the cube-root of the linear-residual of the "
                         "target (linear base absorber then tail compression)"),
    "chain_linres_yj": ("predicts the Yeo-Johnson transform of the linear-residual "
                       "of the target"),
    "chain_monres_cbrt": ("predicts the cube-root of the monotone-residual of the "
                         "target"),
    "chain_monres_yj": ("predicts the Yeo-Johnson transform of the monotone-residual "
                       "of the target"),
    "chain_linres_cbrt_qn": ("predicts the quantile-normalised cube-root of the "
                            "linear-residual of the target (3-stage chain)"),
    "asinh_residual": ("predicts the residual after subtracting a fitted linear "
                      "contribution of the base in arcsinh space (signed-base logratio)"),
    "centered_ratio": ("predicts the ratio of target to a shifted base feature "
                      "(ratio extended to signed bases)"),
    "polynomial_residual_deg2": ("predicts the residual after subtracting a fitted "
                                "quadratic (degree-2) function of the base feature"),
    "rank_residual": ("predicts the distribution-free rank-space linear residual "
                     "of the target against the base feature"),
    "smoothing_spline_residual": ("predicts the residual after subtracting a fitted "
                                 "smoothing spline of the base feature"),
    "reciprocal_residual": ("predicts the residual of 1/target against 1/base "
                          "(reciprocal-scale dynamics)"),
    "geometric_mean_residual": ("predicts the ratio of target to the geometric mean "
                              "of several positive base features"),
    "pairwise_interaction_residual": ("predicts the residual after subtracting a "
                                     "fitted multiple of the product of several base features"),
}


def _fmt(value: Any, default: float = 0.0) -> str:
    """Render a fitted scalar param compactly (``%.4g``), tolerating a
    missing key or a non-numeric stored value (returns the raw repr)."""
    if value is None:
        value = default
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _join_base_columns(base_column: str, fitted_params: dict[str, Any]) -> str:
    """Render the multi-base matrix label. The wrapper threads the extra
    base columns via the spec (``extra_base_columns``); at formula-render
    time we only have the primary ``base_column`` so we annotate the count
    when the fit recorded ``n_bases`` / ``alphas`` for honesty."""
    n = fitted_params.get("n_bases")
    if n is None:
        alphas = fitted_params.get("alphas")
        n = len(alphas) if hasattr(alphas, "__len__") else None
    if n and n > 1:
        return f"[{base_column}, ...{int(n) - 1} more]"
    return base_column


# ----------------------------------------------------------------------
# Per-transform formula builders. Each returns (forward_human,
# inverse_human) given (target_col, base_column, fitted_params). The
# ``diff`` / ``ratio`` / ``logratio`` / ``linear_residual`` four keep their
# exact original strings (bit-identical -- pinned by ``TestFormulas``); the
# rest fill the audit S12 gap with faithful, fitted-param-interpolating
# formulas so an inverse can be reproduced at serving time without source.
# ----------------------------------------------------------------------


def _f_diff(t: str, b: str, p: dict) -> tuple[str, str]:
    return (f"T = {t} - {b}", f"y_hat = T_hat + {b}")


def _f_additive_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    beta = _fmt(p.get("beta"))
    return (
        f"T = {t} - {b} - ({beta})",
        f"y_hat = T_hat + {b} + ({beta})",
    )


def _f_median_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    return (
        f"T = {t} - median({t} | quantile_bin({b}))",
        f"y_hat = T_hat + bin_median[bin({b})]  (per-bin lookup, edge-bin fallback)",
    )


def _f_y_quantile_clip(t: str, b: str, p: dict) -> tuple[str, str]:
    q_lo = _fmt(p.get("q_lo"))
    q_hi = _fmt(p.get("q_hi"))
    return (
        f"T = clip({t}, {q_lo}, {q_hi})  (train 0.5%/99.5% quantiles)",
        f"y_hat = clip(T_hat, {q_lo}, {q_hi})",
    )


def _f_ratio(t: str, b: str, p: dict) -> tuple[str, str]:
    eps = float(p.get("eps", 1e-12))
    return (
        f"T = {t} / {b}  (with |{b}| >= {eps:.3g})",
        f"y_hat = T_hat * {b}",
    )


def _f_logratio(t: str, b: str, p: dict) -> tuple[str, str]:
    median_t = float(p.get("median_t", 0.0))
    mad_eff = float(p.get("mad_eff", 0.0))
    k = float(p.get("soft_cap_k", 10.0))
    return (
        f"T = log({t}) - log({b})  (requires {t}, {b} > 0)",
        f"y_hat = {b} * exp(clip(T_hat, {median_t:.4g} +/- {k:.4g}*{mad_eff:.4g}))",
    )


def _f_linear_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    alpha = float(p.get("alpha", 0.0))
    beta = float(p.get("beta", 0.0))
    return (
        f"T = {t} - {alpha:.4g} * {b} - ({beta:.4g})",
        f"y_hat = T_hat + {alpha:.4g} * {b} + ({beta:.4g})",
    )


def _f_linear_residual_multi(t: str, b: str, p: dict) -> tuple[str, str]:
    bases = _join_base_columns(b, p)
    beta = _fmt(p.get("beta"))
    if p.get("collinear_fallback"):
        note = "  (collinear fallback: alphas=0, beta=train_mean)"
    else:
        note = ""
    return (
        f"T = {t} - alphas . {bases} - ({beta}){note}",
        f"y_hat = T_hat + alphas . {bases} + ({beta})",
    )


def _f_linear_residual_grouped(t: str, b: str, p: dict) -> tuple[str, str]:
    a_g = _fmt(p.get("alpha_global"))
    b_g = _fmt(p.get("beta_global"))
    return (
        f"T = {t} - alpha_g * {b} - beta_g  (per-group OLS, James-Stein shrunk "
        f"to global alpha={a_g}, beta={b_g}; small/unseen groups use global)",
        f"y_hat = T_hat + alpha_g * {b} + beta_g",
    )


def _f_quantile_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    return (
        f"T = ({t} - median_bin({t})) / IQR_bin({t})  (quantile bins of {b})",
        f"y_hat = T_hat * IQR_bin + median_bin  (sparse/constant bins -> global)",
    )


def _f_monotonic_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    return (
        f"T = {t} - g({b})  (g = monotone PCHIP fitted on quantile-knot medians)",
        f"y_hat = T_hat + g({b})  (out-of-range {b} clips to edge knot)",
    )


def _f_ewma_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    k = int(p.get("k", 7))
    alpha = 2.0 / (k + 1.0)
    return (
        f"T = {t} - EWMA_k({b})  (k={k}, alpha=2/(k+1)={alpha:.4g}; chronological order required)",
        f"y_hat = T_hat + EWMA_k({b})",
    )


def _f_rolling_quantile_ratio(t: str, b: str, p: dict) -> tuple[str, str]:
    k = int(p.get("k", 0))
    eps = float(p.get("eps", 0.0))
    return (
        f"T = {t} / RollingMedian_k({b})  (centred k={k}, eps floor {eps:.3g}; LOOK-AHEAD)",
        f"y_hat = T_hat * RollingMedian_k({b})",
    )


def _f_frac_diff(t: str, b: str, p: dict) -> tuple[str, str]:
    d = _fmt(p.get("d"))
    lags = int(p.get("lags", 0))
    return (
        f"T_i = sum_k w_k * {t}_(i-k)  (Lopez de Prado frac-diff, d={d}, {lags} lags; "
        f"w_k = -w_(k-1)*(d-k+1)/k, pre-window padded with train mean)",
        f"y_hat_i = (T_i - sum_(k>=1) w_k * y_hat_(i-k)) / w_0  (iterative reconstruction)",
    )


def _f_cbrt_y(t: str, b: str, p: dict) -> tuple[str, str]:
    return (f"T = sign({t}) * |{t}|^(1/3)", "y_hat = T_hat^3")


def _f_log_y(t: str, b: str, p: dict) -> tuple[str, str]:
    offset = _fmt(p.get("offset"))
    return (
        f"T = log({t} + {offset})  (offset fitted so min({t}_train) + offset > 0)",
        f"y_hat = exp(T_hat) - {offset}",
    )


def _f_yeo_johnson_y(t: str, b: str, p: dict) -> tuple[str, str]:
    lam = _fmt(p.get("lambda"))
    return (
        f"T = YeoJohnson({t}; lambda={lam})  (lambda fitted by MLE, clipped to [-2, 4])",
        f"y_hat = YeoJohnson_inverse(T_hat; lambda={lam})",
    )


def _f_quantile_normal_y(t: str, b: str, p: dict) -> tuple[str, str]:
    return (
        f"T = Phi^-1(rank({t}) / (n+1))  (empirical-CDF -> standard Normal via knots)",
        "y_hat = empirical_CDF_inverse(Phi(T_hat))  (knot interpolation, scale lost)",
    )


def _f_asinh_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    alpha = _fmt(p.get("alpha"))
    beta = _fmt(p.get("beta"))
    return (
        f"T = asinh({t}) - {alpha} * asinh({b}) - ({beta})",
        f"y_hat = sinh(T_hat + {alpha} * asinh({b}) + ({beta}))",
    )


def _f_centered_ratio(t: str, b: str, p: dict) -> tuple[str, str]:
    c = _fmt(p.get("c"))
    eps = float(p.get("eps", 0.0))
    return (
        f"T = {t} / ({b} + {c})  (c fitted so {b}+c>0, eps floor {eps:.3g})",
        f"y_hat = T_hat * ({b} + {c})",
    )


def _f_polynomial_residual_deg2(t: str, b: str, p: dict) -> tuple[str, str]:
    a1 = _fmt(p.get("alpha1"))
    a2 = _fmt(p.get("alpha2"))
    beta = _fmt(p.get("beta"))
    return (
        f"T = {t} - {a1} * {b} - {a2} * {b}^2 - ({beta})",
        f"y_hat = T_hat + {a1} * {b} + {a2} * {b}^2 + ({beta})",
    )


def _f_rank_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    alpha = _fmt(p.get("alpha"))
    beta = _fmt(p.get("beta"))
    return (
        f"T = rank({t})/n - {alpha} * rank({b})/n - ({beta})  (train sorted-y/base lookup)",
        f"y_hat = train_sorted_y[clip(rank_recovered, 0, 1)]",
    )


def _f_smoothing_spline_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    s = _fmt(p.get("s"))
    return (
        f"T = {t} - g({b})  (g = UnivariateSpline, smoothing s={s})",
        f"y_hat = T_hat + g({b})",
    )


def _f_reciprocal_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    eps_y = float(p.get("eps_y", 0.0))
    eps_b = float(p.get("eps_b", 0.0))
    return (
        f"T = 1/{t} - 1/{b}  (eps floors {eps_y:.3g}/{eps_b:.3g})",
        f"y_hat = 1 / (T_hat + 1/{b})",
    )


def _f_geometric_mean_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    bases = _join_base_columns(b, p)
    return (
        f"T = {t} / geomean({bases})  (requires every base > 0)",
        f"y_hat = T_hat * geomean({bases})",
    )


def _f_pairwise_interaction_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    bases = _join_base_columns(b, p)
    alpha = _fmt(p.get("alpha"))
    beta = _fmt(p.get("beta"))
    return (
        f"T = {t} - {alpha} * prod({bases}) - ({beta})",
        f"y_hat = T_hat + {alpha} * prod({bases}) + ({beta})",
    )


def _f_linear_residual_robust(t: str, b: str, p: dict) -> tuple[str, str]:
    # Same forward/inverse algebra as linear_residual once (alpha, beta) fitted.
    return _f_linear_residual(t, b, p)


def _f_chain_linres_cbrt(t: str, b: str, p: dict) -> tuple[str, str]:
    bp = p.get("bivariate_params", {}) or {}
    alpha = _fmt(bp.get("alpha"))
    beta = _fmt(bp.get("beta"))
    return (
        f"T1 = {t} - {alpha} * {b} - ({beta}); T = sign(T1) * |T1|^(1/3)",
        f"T1_hat = T_hat^3; y_hat = T1_hat + {alpha} * {b} + ({beta})",
    )


def _f_chain_linres_yj(t: str, b: str, p: dict) -> tuple[str, str]:
    bp = p.get("bivariate_params", {}) or {}
    up = p.get("unary_params", {}) or {}
    alpha = _fmt(bp.get("alpha"))
    beta = _fmt(bp.get("beta"))
    lam = _fmt(up.get("lambda"))
    return (
        f"T1 = {t} - {alpha} * {b} - ({beta}); T = YeoJohnson(T1; lambda={lam})",
        f"T1_hat = YeoJohnson_inverse(T_hat; lambda={lam}); y_hat = T1_hat + {alpha} * {b} + ({beta})",
    )


def _f_chain_monres_cbrt(t: str, b: str, p: dict) -> tuple[str, str]:
    return (
        f"T1 = {t} - g({b}) (monotone PCHIP); T = sign(T1) * |T1|^(1/3)",
        f"T1_hat = T_hat^3; y_hat = T1_hat + g({b})",
    )


def _f_chain_monres_yj(t: str, b: str, p: dict) -> tuple[str, str]:
    up = p.get("unary_params", {}) or {}
    lam = _fmt(up.get("lambda"))
    return (
        f"T1 = {t} - g({b}) (monotone PCHIP); T = YeoJohnson(T1; lambda={lam})",
        f"T1_hat = YeoJohnson_inverse(T_hat; lambda={lam}); y_hat = T1_hat + g({b})",
    )


def _f_chain_linres_cbrt_qn(t: str, b: str, p: dict) -> tuple[str, str]:
    bp = p.get("bivariate_params", {}) or {}
    alpha = _fmt(bp.get("alpha"))
    beta = _fmt(bp.get("beta"))
    return (
        f"T1 = {t} - {alpha} * {b} - ({beta}); T2 = sign(T1) * |T1|^(1/3); "
        f"T = Phi^-1(rank(T2)/(n+1))",
        f"T2_hat = empirical_CDF_inverse(Phi(T_hat)); T1_hat = T2_hat^3; "
        f"y_hat = T1_hat + {alpha} * {b} + ({beta})  (scale lost in quantile-normal stage)",
    )


# name -> builder. Source of truth alongside ``_TRANSFORM_DESCRIPTIONS``;
# both are coverage-pinned against the live registry by the S12 meta-test.
_TRANSFORM_FORMULA_BUILDERS: dict[str, Any] = {
    "diff": _f_diff,
    "additive_residual": _f_additive_residual,
    "median_residual": _f_median_residual,
    "y_quantile_clip": _f_y_quantile_clip,
    "ratio": _f_ratio,
    "logratio": _f_logratio,
    "linear_residual": _f_linear_residual,
    "linear_residual_robust": _f_linear_residual_robust,
    "linear_residual_multi": _f_linear_residual_multi,
    "linear_residual_grouped": _f_linear_residual_grouped,
    "quantile_residual": _f_quantile_residual,
    "monotonic_residual": _f_monotonic_residual,
    "ewma_residual": _f_ewma_residual,
    "rolling_quantile_ratio": _f_rolling_quantile_ratio,
    "frac_diff": _f_frac_diff,
    "cbrt_y": _f_cbrt_y,
    "log_y": _f_log_y,
    "yeo_johnson_y": _f_yeo_johnson_y,
    "quantile_normal_y": _f_quantile_normal_y,
    "asinh_residual": _f_asinh_residual,
    "centered_ratio": _f_centered_ratio,
    "polynomial_residual_deg2": _f_polynomial_residual_deg2,
    "rank_residual": _f_rank_residual,
    "smoothing_spline_residual": _f_smoothing_spline_residual,
    "reciprocal_residual": _f_reciprocal_residual,
    "geometric_mean_residual": _f_geometric_mean_residual,
    "pairwise_interaction_residual": _f_pairwise_interaction_residual,
    "chain_linres_cbrt": _f_chain_linres_cbrt,
    "chain_linres_yj": _f_chain_linres_yj,
    "chain_monres_cbrt": _f_chain_monres_cbrt,
    "chain_monres_yj": _f_chain_monres_yj,
    "chain_linres_cbrt_qn": _f_chain_linres_cbrt_qn,
}


def _registered_transform_names() -> frozenset[str]:
    """Live registry keys, lazily imported (no import-time coupling; the
    registry is only needed for the S12 coverage meta-test and graceful
    fallback bookkeeping, never at provenance module load)."""
    try:
        from .transforms import TRANSFORMS_REGISTRY  # local import: avoid cycle
        return frozenset(TRANSFORMS_REGISTRY.keys())
    except Exception:  # pragma: no cover - registry should always import
        return frozenset()


def _format_transform_formulas(
    transform_name: str, base_column: str, target_col: str,
    fitted_params: dict[str, Any],
) -> tuple[str, str, str]:
    """Return (forward_human, inverse_human, description) strings.

    Strings interpolate fitted parameter values where applicable. Used
    by :class:`CompositeProvenance` to render audit-friendly formula
    descriptions without forcing the caller to know the registry.

    Every transform registered in ``TRANSFORMS_REGISTRY`` has a dedicated
    builder in ``_TRANSFORM_FORMULA_BUILDERS`` (audit S12); a genuinely
    unknown / future name falls back to the generic ``forward(...)`` /
    ``inverse(...)`` stub so the function never raises.
    """
    description = _TRANSFORM_DESCRIPTIONS.get(transform_name, "")
    builder = _TRANSFORM_FORMULA_BUILDERS.get(transform_name)
    if builder is not None:
        forward, inverse = builder(target_col, base_column, fitted_params or {})
        return (forward, inverse, description or f"transform '{transform_name}'")
    # Unknown / future transform: fall back to a generic description.
    return (
        f"T = forward({target_col}, {base_column}) [{transform_name}]",
        f"y_hat = inverse(T_hat, {base_column}) [{transform_name}]",
        description or f"transform '{transform_name}'",
    )


def report_to_markdown(
    *,
    target_col: str,
    specs: Sequence[CompositeSpec],
    failures: Sequence[dict[str, Any]] = (),
    ensemble_metadata: dict[str, Any] | None = None,
    random_state: int | None = None,
) -> str:
    """Render a stakeholder-ready Markdown report for one target's
    composite-target discovery output.

    Sections:

    1. Summary line: target name, count of kept specs, count of rejected.
    2. Discovered specs table with mi_y / mi_t / mi_gain / valid_frac.
    3. Per-spec audit paragraph (one per spec).
    4. Rejected candidates table with reason.
    5. Ensemble metadata if provided.

    All user-controlled strings (column names, target names) are NOT
    HTML-escaped in this version because Markdown is plain text by
    default; if the caller renders to HTML elsewhere they should
    escape there.
    """
    lines: list[str] = []
    lines.append(f"# Composite-target discovery report: `{target_col}`")
    lines.append("")
    lines.append(
        f"**{len(specs)}** discovered spec(s); **{len(failures)}** rejected candidate(s)."
    )
    lines.append("")

    if specs:
        lines.append("## Discovered specs")
        lines.append("")
        lines.append("| name | base | transform | mi_y | mi_t | mi_gain | valid_frac | n_train |")
        lines.append("|------|------|-----------|------|------|---------|-----------|---------|")
        for spec in specs:
            lines.append(
                f"| `{spec.name}` | `{spec.base_column}` | `{spec.transform_name}` | "
                f"{spec.mi_y:.4f} | {spec.mi_t:.4f} | {spec.mi_gain:+.4f} | "
                f"{spec.valid_domain_frac:.1%} | {spec.n_train_rows} |"
            )
        lines.append("")
        lines.append("## Per-spec audit")
        lines.append("")
        for spec in specs:
            ensemble_w = None
            ensemble_strat = None
            if ensemble_metadata:
                # The spec contributes one component per ensemble model, named ``{spec.name}#{i}``; its true mass is the SUM over those, not the first match.
                _w_sum = 0.0
                _k = 0
                for nm, w in zip(
                    ensemble_metadata.get("component_names", []),
                    ensemble_metadata.get("weights", []),
                ):
                    if nm.rsplit("#", 1)[0] == spec.name:
                        _w_sum += float(w)
                        _k += 1
                if _k:
                    ensemble_w = _w_sum
                    ensemble_strat = ensemble_metadata.get("strategy")
            prov = CompositeProvenance.from_spec(
                spec=spec, random_state=random_state,
                ensemble_weight=ensemble_w,
                ensemble_strategy=ensemble_strat,
            )
            lines.append(f"### `{spec.name}`")
            lines.append("")
            lines.append(prov.to_audit_trail())
            lines.append("")

    if failures:
        lines.append("## Rejected candidates")
        lines.append("")
        lines.append("| base | transform | reason |")
        lines.append("|------|-----------|--------|")
        for f in failures:
            base = f.get("base_column", "?")
            transform = f.get("transform_name", "?")
            reason = f.get("reason", "")
            lines.append(f"| `{base}` | `{transform}` | {reason} |")
        lines.append("")

    if ensemble_metadata:
        lines.append("## Cross-target ensemble")
        lines.append("")
        lines.append(f"Strategy: **{ensemble_metadata.get('strategy', '?')}**")
        lines.append("")
        lines.append("| component | weight |")
        lines.append("|-----------|-------:|")
        for nm, w in zip(
            ensemble_metadata.get("component_names", []),
            ensemble_metadata.get("weights", []),
        ):
            lines.append(f"| `{nm}` | {w:.4f} |")
        lines.append("")

    return "\n".join(lines)
