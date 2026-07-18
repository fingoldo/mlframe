"""CompositeProvenance dataclass + report-to-markdown helper. Production-grade metadata for one composite-target spec: human-readable formula, fitted params, baseline metrics, ensemble weight, selection-path audit trail. ``composite.py`` re-exports every symbol below at its bottom for full back-compat."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING, Any, Sequence,
)

import numpy as np

if TYPE_CHECKING:
    from .spec import CompositeSpec  # used as a forward annotation in CompositeProvenance.from_spec; importing at runtime is unnecessary and risks circular load.

logger = logging.getLogger(__name__)


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
            "ensemble_weight": (None if self.ensemble_weight is None else float(self.ensemble_weight)),
            "ensemble_strategy": self.ensemble_strategy,
        }

    def to_audit_trail(self) -> str:
        """Single-paragraph human-readable summary suitable for a Slack
        message or a code-review comment. Quotes the exact numbers
        that justified inclusion so the reader can cross-check."""
        ensemble_clause = ""
        if self.ensemble_weight is not None and self.ensemble_strategy is not None:
            ensemble_clause = f" In the cross-target {self.ensemble_strategy} ensemble it " f"received weight {self.ensemble_weight:.3f}."
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


# The transform-formula-text registry (per-transform description table, forward/inverse formula
# builders, dynamic auto-chain registration) lives in the sibling ``provenance_formulas`` module,
# carved out to keep this file under the 1k LOC ceiling; re-imported here so every existing
# ``from .provenance import ...`` call site keeps working unchanged.
from .provenance_formulas import (
    _TRANSFORM_DESCRIPTIONS,
    _TRANSFORM_FORMULA_BUILDERS,
    _format_transform_formulas,
    _registered_transform_names,
    register_chain_provenance,
)

__all__ = [
    "CompositeProvenance",
    "report_to_markdown",
    "register_chain_provenance",
    "_format_transform_formulas",
    "_TRANSFORM_DESCRIPTIONS",
    "_TRANSFORM_FORMULA_BUILDERS",
    "_registered_transform_names",
]


# Ordered (substring, gate-label) pairs used to classify a rejection ``reason``
# string into the named discovery gate that produced it. First match wins, so
# more-specific markers precede generic ones. Lets the decision-trail table show
# *which* gate dropped each candidate at a glance without parsing free text.
_GATE_MARKERS: tuple[tuple[str, str], ...] = (
    ("forbidden_pattern", "name-filter"),
    ("non_numeric", "dtype-filter"),
    ("insufficient_finite_rows", "finite-rows"),
    ("constant_or_near_constant", "constant"),
    ("forbidden_base_corr", "leak-guard"),
    ("BH-FDR", "fdr"),
    ("BY-FDR", "fdr"),
    ("bootstrap p", "fdr"),
    ("mi_gain", "mi-gain"),
    ("eps", "mi-gain"),
    ("valid_domain_frac", "domain"),
    ("raw_baseline", "raw-baseline"),
    ("tiny", "tiny-cv"),
    ("rmse", "tiny-cv"),
    ("alpha", "alpha-drift"),
    ("collapse", "collapse"),
    ("gate", "tiny-cv"),
)


def _classify_gate(reason: str) -> str:
    """Map a rejection ``reason`` string to the named gate that produced it.

    Pure substring routing over :data:`_GATE_MARKERS` (case-insensitive,
    first-match-wins). Returns ``"?"`` for an empty / unrecognised reason so
    the decision-trail column is never blank.
    """
    if not reason:
        return "kept"
    low = reason.lower()
    for marker, label in _GATE_MARKERS:
        if marker.lower() in low:
            return label
    return "other"


def _fmt_metric(value: Any, fmt: str = "{:.4f}") -> str:
    """Format an optional numeric cell, rendering missing / non-finite as a dash.

    Keeps the metrics matrix readable when a spec carries a metric (tiny-CV
    RMSE, raw-baseline delta) that another spec does not.
    """
    if value is None:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "-"
    return fmt.format(f)


def _spec_metric(
    name: str, attr: str, spec: Any, extra: dict[str, Any] | None,
) -> Any:
    """Resolve one per-spec metric, preferring an explicit ``spec_metrics``
    override, then a same-named attribute on the spec, else ``None``.

    ``extra`` is the caller-supplied ``spec_metrics[name]`` mapping (e.g.
    ``{"tiny_cv_rmse": 0.31, "raw_delta": -0.04}``); discovery already has
    these numbers from the tiny-model rerank but does not store them on the
    frozen :class:`CompositeSpec`, so they ride in via this side channel.
    """
    if extra is not None and attr in extra:
        return extra[attr]
    return getattr(spec, attr, None)


def report_to_markdown(
    *,
    target_col: str,
    specs: Sequence[CompositeSpec],
    failures: Sequence[dict[str, Any]] = (),
    ensemble_metadata: dict[str, Any] | None = None,
    random_state: int | None = None,
    spec_metrics: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render a stakeholder-ready Markdown report for one target's
    composite-target discovery output.

    Sections:

    1. Summary line: target name, count of kept specs, count of rejected.
    2. Discovered specs table with mi_y / mi_t / mi_gain / valid_frac.
    3. Metrics matrix: one row per spec / rejected candidate with
       mi_gain, raw-baseline delta, tiny-CV RMSE, kept/rejected + reason.
    4. Decision trail: which gate each rejected candidate failed.
    5. Per-spec audit paragraph (one per spec).
    6. Rejected candidates table with reason.
    7. Ensemble metadata if provided.

    ``spec_metrics`` is an optional ``{spec_name: {metric: value}}`` side
    channel for per-spec numbers the frozen :class:`CompositeSpec` does not
    carry (``raw_delta`` = composite-vs-raw-baseline tiny-CV RMSE delta,
    negative is better; ``tiny_cv_rmse`` = the composite's own tiny-CV RMSE
    on the y-scale). Missing metrics render as a dash, so callers without
    the rerank numbers still get a valid report.

    All user-controlled strings (column names, target names) are NOT
    HTML-escaped in this version because Markdown is plain text by
    default; if the caller renders to HTML elsewhere they should
    escape there.
    """
    spec_metrics = spec_metrics or {}
    lines: list[str] = []
    lines.append(f"# Composite-target discovery report: `{target_col}`")
    lines.append("")
    lines.append(f"**{len(specs)}** discovered spec(s); **{len(failures)}** rejected candidate(s).")
    lines.append("")

    if specs:
        lines.append("## Discovered specs")
        lines.append("")
        lines.append("| name | base | transform | mi_y | mi_t | mi_gain | valid_frac | n_train |")
        lines.append("|------|------|-----------|------|------|---------|-----------|---------|")
        lines.extend(
            f"| `{spec.name}` | `{spec.base_column}` | `{spec.transform_name}` | "
            f"{spec.mi_y:.4f} | {spec.mi_t:.4f} | {spec.mi_gain:+.4f} | "
            f"{spec.valid_domain_frac:.1%} | {spec.n_train_rows} |"
            for spec in specs
        )
        lines.append("")

    # Metrics matrix + decision trail: kept specs AND rejected candidates in one
    # at-a-glance view, so a user sees WHY each candidate survived or was dropped.
    if specs or failures:
        lines.append("## Metrics matrix")
        lines.append("")
        lines.append(
            "Per-candidate decision metrics. `raw_delta` is composite-vs-raw-baseline "
            "tiny-CV RMSE (negative = composite wins); `tiny_cv_rmse` is the "
            "composite's own y-scale tiny-CV RMSE. `-` = metric not recorded."
        )
        lines.append("")
        lines.append("| name | status | mi_gain | raw_delta | tiny_cv_rmse | gate | reason |")
        lines.append("|------|--------|---------|-----------|--------------|------|--------|")
        for spec in specs:
            extra = spec_metrics.get(spec.name)
            raw_delta = _spec_metric(spec.name, "raw_delta", spec, extra)
            tiny_rmse = _spec_metric(spec.name, "tiny_cv_rmse", spec, extra)
            lines.append(
                f"| `{spec.name}` | kept | {spec.mi_gain:+.4f} | " f"{_fmt_metric(raw_delta, '{:+.4f}')} | " f"{_fmt_metric(tiny_rmse)} | gate-passed | - |"
            )
        for f in failures:
            _f_name = f.get("name")
            name = _f_name if _f_name else (f"__{f.get('transform_name', '?')}__{f.get('base_column', '?')}")
            extra = spec_metrics.get(name)
            reason = f.get("reason", "")
            mi_gain = f.get("mi_gain")
            raw_delta = extra.get("raw_delta") if extra else f.get("raw_delta")
            tiny_rmse = extra.get("tiny_cv_rmse") if extra else f.get("tiny_cv_rmse")
            lines.append(
                f"| `{name}` | rejected | {_fmt_metric(mi_gain, '{:+.4f}')} | "
                f"{_fmt_metric(raw_delta, '{:+.4f}')} | "
                f"{_fmt_metric(tiny_rmse)} | {_classify_gate(reason)} | {reason if reason else '-'} |"
            )
        lines.append("")

        if failures:
            lines.append("## Decision trail")
            lines.append("")
            lines.append("Which gate each rejected candidate failed (first failing gate).")
            lines.append("")
            lines.append("| candidate | failed_gate | detail |")
            lines.append("|-----------|-------------|--------|")
            for f in failures:
                _f_name2 = f.get("name")
                name = _f_name2 if _f_name2 else (f"__{f.get('transform_name', '?')}__{f.get('base_column', '?')}")
                reason = f.get("reason", "")
                lines.append(f"| `{name}` | {_classify_gate(reason)} | {reason if reason else '-'} |")
            lines.append("")

    if specs:
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


# --- Carve re-export (sibling pattern): the per-transform formula-builder block ->
# provenance_formulas.py (carved VERBATIM under the 1k ceiling). Rebind EVERY moved name (public AND
# underscore-private) into THIS namespace so every existing ``from .provenance import X`` path still
# resolves byte-for-byte.
from . import provenance_formulas as _pf
for _n in dir(_pf):
    if not _n.startswith("__") and _n not in globals():
        globals()[_n] = getattr(_pf, _n)
del _n
