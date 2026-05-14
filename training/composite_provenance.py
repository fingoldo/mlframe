"""CompositeProvenance dataclass + report-to-markdown helper. Production-grade metadata for one composite-target spec: human-readable formula, fitted params, baseline metrics, ensemble weight, selection-path audit trail. Split out of composite.py for clean separation between discovery internals and stakeholder-facing audit artefacts; composite.py re-exports every symbol below at its bottom for full back-compat."""


from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple,
)

import numpy as np

from .composite_transforms import _TRANSFORMS_REGISTRY

if TYPE_CHECKING:
    from .composite_spec import CompositeSpec  # used as a forward annotation in CompositeProvenance.from_spec; importing at runtime is unnecessary and risks circular load.

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

    Why this exists. Without provenance, ``TVT__linear_residual__TVT_prev``
    is an opaque key. With provenance, the same key reads as
    "predicts residual after subtracting fitted alpha=0.952 of the
    previous-period TVT_prev value (R^2_train = 0.91), selected
    because removing the linear contribution exposed a residual MI
    of 0.165 against the remaining features".

    Convert to dict via :meth:`to_dict` (JSON-serialisable) or to a
    stakeholder-ready paragraph via :meth:`to_audit_trail`.
    """

    # Identity
    composite_id: str
    discovery_timestamp: str  # ISO 8601, no datetime obj to keep dict-pickle clean
    discovery_random_state: int

    # Origin
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

    # Optional: weight in cross-target ensemble (filled at integration time).
    ensemble_weight: float | None = None
    ensemble_strategy: str | None = None

    @classmethod
    def from_spec(
        cls,
        spec: CompositeSpec,
        random_state: int,
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
        from datetime import timezone

        # Stable id derived from (target, transform, base, fitted_params).
        canonical = json.dumps(
            {
                "target_col": spec.target_col,
                "transform_name": spec.transform_name,
                "base_column": spec.base_column,
                "fitted_params": spec.fitted_params,
            },
            sort_keys=True, default=str,
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
            target_col=spec.target_col,
            transform_name=spec.transform_name,
            base_column=spec.base_column,
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
            "target_col": self.target_col,
            "transform_name": self.transform_name,
            "base_column": self.base_column,
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
            f"Composite '{self.target_col}__{self.transform_name}__{self.base_column}' "
            f"(id={self.composite_id}) was discovered using "
            f"random_state={self.discovery_random_state} on "
            f"{self.n_train_rows} train rows ({self.valid_domain_frac:.1%} of valid domain). "
            f"It was selected because MI(T, X\\base)={self.mi_t:.4f} vs "
            f"MI(y, X\\base)={self.mi_y:.4f} (gain={self.mi_gain:+.4f}), "
            f"meaning the transform '{self.stakeholder_description}' exposed "
            f"residual structure the remaining features can predict more easily. "
            f"Forward: {self.forward_formula_human}. "
            f"Inverse: {self.inverse_formula_human}.{ensemble_clause}"
        )


# Friendly transform-name-to-paragraph table.
_TRANSFORM_DESCRIPTIONS: dict[str, str] = {
    "diff": ("predicts the residual after subtracting the base feature "
             "from the target"),
    "ratio": ("predicts the multiplicative factor relating target to "
              "base feature"),
    "logratio": ("predicts the log-ratio of target to base feature, "
                 "stabilising heavy-tail distributions"),
    "linear_residual": ("predicts the residual after subtracting a "
                        "fitted linear contribution of the base feature"),
}


def _format_transform_formulas(
    transform_name: str, base_column: str, target_col: str,
    fitted_params: dict[str, Any],
) -> tuple[str, str, str]:
    """Return (forward_human, inverse_human, description) strings.

    Strings interpolate fitted parameter values where applicable. Used
    by :class:`CompositeProvenance` to render audit-friendly formula
    descriptions without forcing the caller to know the registry.
    """
    description = _TRANSFORM_DESCRIPTIONS.get(transform_name, "")
    if transform_name == "diff":
        return (
            f"T = {target_col} - {base_column}",
            f"y_hat = T_hat + {base_column}",
            description,
        )
    if transform_name == "ratio":
        eps = fitted_params.get("eps", 1e-12)
        return (
            f"T = {target_col} / {base_column}  (with |{base_column}| >= {eps:.3g})",
            f"y_hat = T_hat * {base_column}",
            description,
        )
    if transform_name == "logratio":
        median_t = fitted_params.get("median_t", 0.0)
        mad_eff = fitted_params.get("mad_eff", 0.0)
        return (
            f"T = log({target_col}) - log({base_column})  (requires {target_col}, {base_column} > 0)",
            f"y_hat = {base_column} * exp(softcap(T_hat, {median_t:.4g} +/- 10*{mad_eff:.4g}))",
            description,
        )
    if transform_name == "linear_residual":
        alpha = fitted_params.get("alpha", 0.0)
        beta = fitted_params.get("beta", 0.0)
        return (
            f"T = {target_col} - {alpha:.4g} * {base_column} - ({beta:.4g})",
            f"y_hat = T_hat + {alpha:.4g} * {base_column} + ({beta:.4g})",
            description,
        )
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
    random_state: int = 42,
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
                # Look up this spec's weight if it appears in the
                # ensemble's component list.
                for nm, w in zip(
                    ensemble_metadata.get("component_names", []),
                    ensemble_metadata.get("weights", []),
                ):
                    if nm.startswith(spec.name + "#"):
                        ensemble_w = float(w)
                        ensemble_strat = ensemble_metadata.get("strategy")
                        break
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
