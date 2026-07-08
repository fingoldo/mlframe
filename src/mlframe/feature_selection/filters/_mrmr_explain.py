"""One-call human-readable ``explain_selection()`` report for the MRMR / Hybrid selector.

User-facing question this layer answers, in ONE accessor and under one screen:
  * WHY these features?   -> the surviving selected features + their engineered recipe kinds
                             (assembled from ``fe_provenance_``, the Layer-54 survivor surface).
  * WHAT did FE build / WHICH gate dropped what? -> the BINDING rejection gate -- the single
                             biggest killer from the Layer-? ``fe_rejection_ledger_`` (b1a1048a),
                             with how many candidates it dropped and the margin band by which
                             they missed (reuses ``get_fe_rejection_report``'s ledger).
  * WHICH knob would I turn? -> the meta-FE recommender's CHOSEN fe_* flags
                             (``_fe_recommended_flags_``, captured from Layer-99
                             ``recommend_fe_flags_by_rules``) + a one-line actionable hint mapping
                             the binding gate to the knob that would admit more candidates.

PURE ADDITIVE, FIT-ONLY
-----------------------
This layer touches NO decision logic. It only ASSEMBLES already-populated fit artifacts
(``fe_provenance_``, ``fe_rejection_ledger_``, ``_fe_recommended_flags_``) into a narrative
string. Cost is post-fit string assembly (~0). Degrades gracefully (empty-but-valid report)
when FE is disabled, there are no rejections, or the estimator is a plain classification fit.
There is no opt-in flag: ``explain_selection`` is a method you call on a fitted MRMR.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# Map a binding rejection gate to the single knob a user would relax to admit more
# candidates through it. Keys are the canonical ``FE_GATE_LABELS``. Used for the
# one-line actionable hint; an unmapped gate falls back to a generic message.
_GATE_TO_HINT_KNOB: dict[str, str] = {
    "engineered_mi_prevalence": "lower fe_min_engineered_mi_prevalence (e.g. 0.90 -> 0.80) to admit weaker engineered uplift",
    "marginal_pair_mi_prescreen": "lower fe_synergy_min_prevalence to let lower-synergy pairs through the pre-screen",
    "order2_maxt_floor": "raise the order-2 permutation budget / lower its null floor to admit borderline joint-MI pairs",
    "marginal_uplift_floor": "lower the marginal-uplift floor to admit pairs with weaker joint recovery",
    "cmi_redundancy": "raise the CMI redundancy tolerance to keep partly-redundant engineered features",
    "stability_vote": "lower the cross-fold stability quorum to admit features that pass on fewer folds",
    "ledger_capped": "(diagnostic) raise FE_REJECTION_LEDGER_CAP -- the ledger truncated, this is not a real gate",
}

# Per-gate one-band relaxation delta, used by the WHAT-IF-FLIP preview. A "one band"
# is the additive step by which a user would loosen the gate's threshold; the ledger
# records ``margin = observed - threshold``, so loosening the threshold by ``delta``
# re-admits a recorded candidate iff this gate ACTUALLY BLOCKED it (``margin < 0``, observed
# below the floor) AND it would clear the relaxed floor (``margin > -delta``) -- i.e.
# ``-delta < margin < 0``. Candidates recorded with ``margin >= 0`` cleared this gate and were
# dropped downstream, so relaxing this gate does not re-admit them (they stay dropped on refit).
# Bands are sized to the gate's natural scale: ratio gates (prevalence) step in ~0.10 ratio;
# CMI / uplift / maxT step in ~0.05 of their measured unit. The mapped knob name (for the
# narrative) is the canonical fe_* flag the user would actually turn.
_GATE_TO_FLIP_BAND: dict[str, tuple[str, float]] = {
    "engineered_mi_prevalence": ("fe_min_engineered_mi_prevalence (0.90 -> 0.80)", 0.10),
    "marginal_pair_mi_prescreen": ("fe_synergy_min_prevalence (1.5 -> 1.4)", 0.10),
    "order2_maxt_floor": ("order-2 maxT null floor (one band)", 0.05),
    "marginal_uplift_floor": ("marginal-uplift floor (one band)", 0.05),
    "cmi_redundancy": ("CMI redundancy tolerance (one band)", 0.05),
    "stability_vote": ("cross-fold stability quorum (one fold)", 1.0),
}


# Hard cap on the assembled report so it always fits under ~1 screen. Survivor / ledger
# detail beyond these row budgets is summarised, never dumped in full.
_MAX_SURVIVOR_ROWS = 12
_MAX_LEDGER_GATES = 5
_SCREEN_CHAR_CAP = 2600


def _fmt_margin_band(series: pd.Series) -> str:
    """Render a "-0.04..-0.11" style margin band from a numeric margin column; '' if none."""
    try:
        _coerced = pd.to_numeric(series, errors="coerce")
        _n_dropped = int(_coerced.isna().sum() - series.isna().sum()) if hasattr(series, "isna") else int(_coerced.isna().sum())
        if _n_dropped > 0:
            logger.warning("mrmr-explain: margin-band render dropped %d unparseable margin value(s).", _n_dropped)
        vals = _coerced.dropna()
    except Exception:
        return ""
    if vals.empty:
        return ""
    lo = float(vals.min())
    hi = float(vals.max())
    if lo == hi:
        return f"margin {lo:+.2f}"
    return f"margin {hi:+.2f}..{lo:+.2f}"


def _survivor_section(mrmr_self: Any) -> str:
    """Narrate the surviving selected features + their engineered recipe kinds."""
    prov = getattr(mrmr_self, "fe_provenance_", None)
    if prov is None or not isinstance(prov, pd.DataFrame) or prov.empty:
        return "Surviving features: none recorded (estimator unfitted, or provenance wiped)."
    by_origin = prov.groupby("origin", dropna=False).size().sort_values(ascending=False)
    n_total = len(prov)
    n_eng = int((prov["origin"].astype(str) != "raw").sum())
    kinds = [str(o) for o in by_origin.index if str(o) != "raw"]
    kinds_str = ", ".join(f"{k}={int(by_origin[k])}" for k in kinds) if kinds else "none"
    lines = [
        f"Surviving features: {n_total} selected ({n_eng} engineered, {n_total - n_eng} raw).",
        f"  engineered recipe kinds: {kinds_str}",
    ]
    # PER-FEATURE MI/gain ATTRIBUTION: order survivors by their MRMR gain to y (the
    # selection score the greedy screen already stored in ``mrmr_gain``) so the user
    # sees WHICH survivors carry the signal. Gain is non-NaN for selected columns;
    # screened-out produced columns carry NaN and are excluded from the attribution
    # roster. Falls back to provenance order when no gains are recorded.
    # NOTE: the helper gain column MUST NOT start with an underscore -- ``itertuples``
    # rewrites leading-underscore / invalid-identifier column names to positional
    # ``_N``, which would make ``getattr(row, ...)`` silently miss the gain.
    if "mrmr_gain" in prov.columns:
        gains = pd.to_numeric(prov["mrmr_gain"], errors="coerce")
        _n_bad_gain = int(gains.isna().sum() - pd.isna(prov["mrmr_gain"]).sum())
        if _n_bad_gain > 0:
            logger.warning("mrmr-explain: %d mrmr_gain value(s) were unparseable and excluded from the attribution roster.", _n_bad_gain)
        ranked = prov.assign(gain_attr=gains)
        scored = ranked[ranked["gain_attr"].notna()].sort_values("gain_attr", ascending=False)
        roster = scored if not scored.empty else ranked
    else:
        roster = prov.assign(gain_attr=float("nan"))
    head = roster.head(_MAX_SURVIVOR_ROWS)

    def _attr(row: Any) -> str:
        g = getattr(row, "gain_attr", float("nan"))
        try:
            gv = float(g)
        except (TypeError, ValueError):
            gv = float("nan")
        gain_str = f" gain={gv:.4f}" if gv == gv else ""  # NaN-safe
        return f"{row.feature_name}[{row.origin}]{gain_str}"

    named = ", ".join(_attr(r) for r in head.itertuples(index=False))
    suffix = "" if n_total <= _MAX_SURVIVOR_ROWS else f", ... (+{n_total - _MAX_SURVIVOR_ROWS} more)"
    lines.append(f"  by MI/gain attribution: {named}{suffix}")
    return "\n".join(lines)


def _rejection_section(mrmr_self: Any) -> tuple[str, str | None]:
    """Narrate the BINDING rejection gate. Returns (text, binding_gate_or_None)."""
    led = getattr(mrmr_self, "fe_rejection_ledger_", None)
    if led is None or not isinstance(led, pd.DataFrame) or led.empty:
        return ("Rejections: none -- no FE candidate was dropped by a gate this fit.", None)
    by_gate = led.groupby("gate", dropna=False).size().sort_values(ascending=False)
    n_total = len(led)
    binding_gate = str(by_gate.index[0])
    binding_n = int(by_gate.iloc[0])
    band = _fmt_margin_band(led.loc[led["gate"].astype(str) == binding_gate, "margin"])
    band_str = f" ({band})" if band else ""
    lines = [
        f"Rejections: {n_total} FE candidates dropped across {len(by_gate)} gate(s).",
        f"  BINDING gate: {binding_gate} -- dropped {binding_n}{band_str}.",
    ]
    others = [f"{g}={int(c)}" for g, c in list(by_gate.items())[1:_MAX_LEDGER_GATES]]
    if others:
        lines.append("  other gates: " + ", ".join(others))
    return ("\n".join(lines), binding_gate)


def _recommender_section(mrmr_self: Any) -> str:
    """Narrate the meta-FE recommender's CHOSEN fe_* flags (Layer-99)."""
    rec = getattr(mrmr_self, "_fe_recommended_flags_", None)
    if not rec:
        return "FE recommender: not consulted this fit (fe_auto=False or no rule matched)."
    chosen = sorted(f for f, on in rec.items() if on)
    if not chosen:
        return "FE recommender (Layer-99): inspected the data shape, chose NO fe_* generators (clean continuous frame)."
    return "FE recommender (Layer-99) chose fe_* flags: " + ", ".join(chosen)


def _whatif_section(mrmr_self: Any, binding_gate: str | None) -> str:
    """WHAT-IF-FLIP preview: how many ledger candidates a one-band relaxation re-admits.

    PURE COUNT over the recorded ledger -- NO refit. For the binding gate (and any other
    gate that maps to an fe_* knob), relaxing the threshold by the gate's one-band ``delta``
    re-admits exactly the recorded candidates this gate ACTUALLY BLOCKED (``margin < 0``,
    observed below the floor) that would clear the relaxed floor (``margin > -delta``) --
    i.e. ``-delta < margin < 0`` (margin == observed - threshold). Candidates recorded with
    ``margin >= 0`` cleared this gate and were dropped downstream, so relaxing this gate does
    not re-admit them. The preview reports that count per knob.
    """
    led = getattr(mrmr_self, "fe_rejection_ledger_", None)
    if led is None or not isinstance(led, pd.DataFrame) or led.empty:
        return "What-if-flip: ledger empty -- no relaxation would re-admit anything this fit."
    gate_col = led["gate"].astype(str)
    margin_col = pd.to_numeric(led["margin"], errors="coerce")
    _n_bad_margin = int(margin_col.isna().sum() - pd.isna(led["margin"]).sum())
    if _n_bad_margin > 0:
        logger.warning("mrmr-explain: %d rejection-ledger margin value(s) were unparseable and treated as NaN in the what-if-flip preview.", _n_bad_margin)
    lines: list[str] = []
    # Lead with the binding gate, then any other recorded gate that maps to a knob.
    ordered = []
    if binding_gate is not None and binding_gate in _GATE_TO_FLIP_BAND:
        ordered.append(binding_gate)
    for g in led.groupby(gate_col).size().sort_values(ascending=False).index:
        if g in _GATE_TO_FLIP_BAND and g not in ordered:
            ordered.append(g)
    for gate in ordered:
        knob, delta = _GATE_TO_FLIP_BAND[gate]
        mask = gate_col.eq(gate) & (margin_col > -delta) & (margin_col < 0)
        n_readmit = int(mask.sum())
        lines.append(f"  relaxing {knob} would re-admit {n_readmit} candidate(s) at -{delta:g} < margin < 0 [{gate}]")
        if len(lines) >= 3:  # keep under 1.5 screens
            break
    if not lines:
        return "What-if-flip: no recorded gate maps to a relaxable fe_* knob this fit."
    return "What-if-flip preview (count over ledger, no refit):\n" + "\n".join(lines)


def _hint_line(binding_gate: str | None, mrmr_self: Any) -> str:
    """One-line actionable hint mapping the binding gate to the knob to relax."""
    if binding_gate is None:
        return "Hint: no gate is binding -- FE admitted everything it built; turn ON more fe_* generators to build more candidates."
    knob = _GATE_TO_HINT_KNOB.get(binding_gate)
    if knob is None:
        return f"Hint: '{binding_gate}' is the binding gate; relax its threshold to admit more candidates."
    return f"Hint: to admit more synergy, {knob}."


def explain_selection(mrmr_self: Any) -> str:
    """Assemble a one-screen human-readable explanation of this MRMR fit.

    Reads only already-populated fit artifacts (``fe_provenance_``,
    ``fe_rejection_ledger_``, ``_fe_recommended_flags_``); never recomputes a
    selection, never mutates state, never raises. Returns a capped string a
    domain user can read to answer "why these features / what did FE build /
    what would I turn" WITHOUT reading source.
    """
    try:
        survivors = _survivor_section(mrmr_self)
    except Exception as exc:  # pragma: no cover - assembly must never break
        survivors = f"Surviving features: (unavailable: {type(exc).__name__})."
    try:
        rejections, binding_gate = _rejection_section(mrmr_self)
    except Exception as exc:  # pragma: no cover
        rejections, binding_gate = (f"Rejections: (unavailable: {type(exc).__name__}).", None)
    try:
        recommender = _recommender_section(mrmr_self)
    except Exception as exc:  # pragma: no cover
        recommender = f"FE recommender: (unavailable: {type(exc).__name__})."
    try:
        whatif = _whatif_section(mrmr_self, binding_gate)
    except Exception as exc:  # pragma: no cover
        whatif = f"What-if-flip: (unavailable: {type(exc).__name__})."
    try:
        hint = _hint_line(binding_gate, mrmr_self)
    except Exception as exc:  # pragma: no cover
        hint = f"Hint: (unavailable: {type(exc).__name__})."

    report = "\n".join(
        [
            "=== MRMR.explain_selection() ===",
            survivors,
            rejections,
            whatif,
            recommender,
            hint,
        ]
    )
    if len(report) > _SCREEN_CHAR_CAP:
        report = report[: _SCREEN_CHAR_CAP - 3].rstrip() + "..."
    return report


__all__ = ["explain_selection"]
