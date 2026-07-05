"""Per-gate FE REJECTION LEDGER (the rejection side of the Layer-54 provenance surface).

User-facing question this layer answers: "WHY was this engineered candidate dropped?".
``_mrmr_fe_provenance.py`` records the SURVIVING FE recipes (and -- via
``_produced_recipes_`` -- the produced-but-screened survivors with their origin). The
FE pair / candidate search, however, runs every candidate through ~6 GATES and SILENTLY
discards the ones that miss; the session has repeatedly had to hand-instrument the search
to recover which gate killed a given candidate and by how much. This layer makes that
self-diagnosing: for every candidate a gate kills, it records the candidate (operands +
operator/kind), WHICH gate killed it, the observed value, the threshold, and the MARGIN
(how far it missed), plus the FE step index.

DESIGN
------
``fe_rejection_ledger_`` is a pandas DataFrame populated at the very end of ``fit()`` from
the raw records appended to ``self._fe_rejection_records_`` during the FE loop. One row per
rejected candidate-per-gate. Columns:

- ``candidate``    : engineered-column name / pair label the gate rejected.
- ``operands``     : tuple-as-string of the source column names / indices.
- ``operator``     : the operator / transform / kind under test (``"pair"`` for the
                     pre-screen, the binary func name for the per-pair winner gate, etc.).
- ``gate``         : which gate killed it (one of ``FE_GATE_LABELS``).
- ``observed``     : the value the gate measured (ratio / CMI / pass-count / ...).
- ``threshold``    : the bar the candidate had to clear.
- ``margin``       : ``observed - threshold`` (negative => missed; for ratio gates this is
                     the additive gap, e.g. ``0.94 - 0.97 = -0.03``). NaN when not meaningful.
- ``reason``       : a short machine-readable reason string (gate sub-leg).
- ``step``         : the FE step index (0-based) the rejection happened in.

PURE ADDITIVE
-------------
This layer touches NO gate decision logic. Every record is built from values the gate
ALREADY computed at its drop site (the ratio it compared, the floor it compared against,
the per-name diagnostics the CMI gate already returns, the per-recipe fold pass-counts the
stability vote already counts). The ledger only RECORDS -- it never recomputes an MI, a
permutation null, or a CMI. Cost is one ``list.append`` of a small dict per rejected
candidate.

DEFAULT-ON, MEMORY-CAPPED
-------------------------
Every fitted MRMR carries ``fe_rejection_ledger_`` -- there is no opt-in flag (mirrors
``fe_provenance_``). To bound memory on pathological wide frames (tens of thousands of
rejected pair candidates) the raw record list is capped at ``FE_REJECTION_LEDGER_CAP``
records; once the cap is hit further records are dropped and a single ``_capped`` marker
row is recorded so the truncation is never silent (per the no-silent-truncation rule). The
cap is generous (50k) so realistic fits keep the full ledger.
"""
from __future__ import annotations

import logging as _logging
from typing import Any

import numpy as np
import pandas as pd

logger = _logging.getLogger("mlframe.feature_selection.filters.mrmr")


# Canonical gate labels. Public surface; tests pin the membership.
FE_GATE_LABELS = (
    "marginal_pair_mi_prescreen",  # gate 1: joint-MI prevalence ratio pre-screen (~1.05 / synergy 1.5)
    "order2_maxt_floor",  # gate 2: order-2/3 max-T permutation null floor on the joint MI
    "engineered_mi_prevalence",  # gate 6: best_mi / pair_mi vs fe_min_engineered_mi_prevalence (~0.97)
    "marginal_uplift_floor",  # gate 5: marginal-uplift / joint-recovery (abs-MAD) fallback floor
    "cmi_redundancy",  # gate 3: S5 conditional-MI redundancy gate
    "stability_vote",  # gate 4: cross-fold stability quorum vote
    "ledger_capped",  # synthetic marker: records dropped past the memory cap
)


# Raw-record list memory cap. Realistic fits stay far below this; the cap only guards a
# pathological wide-frame fit from accumulating an unbounded ledger. When exceeded a single
# ``ledger_capped`` marker row is recorded so the truncation is visible (never silent).
FE_REJECTION_LEDGER_CAP: int = 50_000


# Ledger DataFrame schema. ``compute_fe_rejection_ledger`` returns an empty frame with this
# column order when there are no records, so downstream callers can ``.iterrows()`` without a
# column-existence check (mirrors the provenance helper).
_LEDGER_COLUMNS: tuple[str, ...] = (
    "candidate",
    "operands",
    "operator",
    "gate",
    "observed",
    "threshold",
    "margin",
    "reason",
    "step",
)


def _safe_float(value: Any) -> float:
    """Coerce to float; NaN on failure so a malformed record never raises at DataFrame build."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _operands_str(operands: Any) -> str:
    """Stable stringification of the operand collection (sorted dicts not needed; preserve order)."""
    try:
        if operands is None:
            return "()"
        if isinstance(operands, (list, tuple)):
            return "(" + ", ".join(str(o) for o in operands) + ")"
        return str(operands)
    except Exception:
        return "<unstringifiable>"


def record_fe_rejection(
    mrmr_self: Any,
    *,
    gate: str,
    candidate: Any,
    operands: Any = None,
    operator: Any = None,
    observed: Any = float("nan"),
    threshold: Any = float("nan"),
    margin: Any = None,
    reason: str = "",
    step: int = -1,
) -> None:
    """Append ONE rejection record to ``mrmr_self._fe_rejection_records_``.

    Pure-record: never recomputes anything. ``margin`` defaults to ``observed - threshold``
    when both are finite and ``margin`` is not supplied. Memory-capped: once the list reaches
    ``FE_REJECTION_LEDGER_CAP`` a single ``ledger_capped`` marker is appended (so the cap is
    never silent) and subsequent records are dropped. Swallows its OWN errors -- an
    instrumentation failure must never break the FE search.
    """
    try:
        records = getattr(mrmr_self, "_fe_rejection_records_", None)
        if records is None:
            records = []
            mrmr_self._fe_rejection_records_ = records
        n = len(records)
        if n >= FE_REJECTION_LEDGER_CAP:
            # Record the truncation EXACTLY once (no silent truncation), then stop appending.
            if n == FE_REJECTION_LEDGER_CAP:
                records.append(
                    {
                        "candidate": "<ledger cap reached>",
                        "operands": "()",
                        "operator": None,
                        "gate": "ledger_capped",
                        "observed": float(FE_REJECTION_LEDGER_CAP),
                        "threshold": float(FE_REJECTION_LEDGER_CAP),
                        "margin": 0.0,
                        "reason": (
                            f"rejection ledger reached its {FE_REJECTION_LEDGER_CAP}-record cap; " "further rejected candidates are not recorded for this fit"
                        ),
                        "step": int(step),
                    }
                )
                logger.warning(
                    "MRMR FE rejection ledger hit its %d-record cap; further rejected " "candidates this fit are not recorded.",
                    FE_REJECTION_LEDGER_CAP,
                )
            return
        obs = _safe_float(observed)
        thr = _safe_float(threshold)
        if margin is None:
            mrg = obs - thr if (np.isfinite(obs) and np.isfinite(thr)) else float("nan")
        else:
            mrg = _safe_float(margin)
        records.append(
            {
                "candidate": str(candidate),
                "operands": _operands_str(operands),
                "operator": (None if operator is None else str(operator)),
                "gate": str(gate),
                "observed": obs,
                "threshold": thr,
                "margin": mrg,
                "reason": str(reason),
                "step": int(step),
            }
        )
    except Exception as exc:  # pragma: no cover - instrumentation must never break the fit
        logger.debug("MRMR FE rejection ledger record failed (%s); skipping.", exc)


def compute_fe_rejection_ledger(mrmr_self: Any) -> pd.DataFrame:
    """Build the ``fe_rejection_ledger_`` DataFrame from ``_fe_rejection_records_``.

    Pure-read; returns an empty (correctly-shaped) frame when there are no records.
    """
    columns = list(_LEDGER_COLUMNS)
    empty = pd.DataFrame({col: pd.Series([], dtype=object) for col in columns})
    records = getattr(mrmr_self, "_fe_rejection_records_", None)
    if not records:
        return empty
    try:
        df = pd.DataFrame(list(records), columns=columns)
    except Exception as exc:
        logger.warning("MRMR fe_rejection_ledger_ build failed (%s); returning empty.", exc)
        return empty
    return df


def populate_fe_rejection_ledger(mrmr_self: Any) -> None:
    """Run ``compute_fe_rejection_ledger`` and stash the result on the estimator.

    Mirrors ``populate_fe_provenance``: wrapped here so the parent module stays under its LOC
    budget and the fallback schema is centralised. Never raises.
    """
    try:
        mrmr_self.fe_rejection_ledger_ = compute_fe_rejection_ledger(mrmr_self)
    except Exception as exc:
        logger.warning(
            "MRMR.fit: fe_rejection_ledger_ population failed (%s: %s); " "get_fe_rejection_report() will surface the empty-DataFrame message.",
            type(exc).__name__,
            exc,
        )
        mrmr_self.fe_rejection_ledger_ = pd.DataFrame(
            {col: pd.Series([], dtype=object) for col in _LEDGER_COLUMNS},
        )


def get_fe_rejection_report(mrmr_self: Any) -> str:
    """Render ``fe_rejection_ledger_`` as a single human-readable string.

    Header summarises rejection counts per gate; body is the full DataFrame. Returns an
    explanatory message (never raises) when the ledger is empty / missing.
    """
    led = getattr(mrmr_self, "fe_rejection_ledger_", None)
    if led is None or not isinstance(led, pd.DataFrame) or led.empty:
        return (
            "MRMR.fe_rejection_ledger_ is empty: estimator is unfitted, no FE candidate was "
            "rejected this fit, or the fitted attributes have been wiped. Call fit() first."
        )
    by_gate = led.groupby("gate", dropna=False).size().sort_values(ascending=False)
    header_parts = [f"{gate}={count}" for gate, count in by_gate.items()]
    header = "MRMR FE rejection ledger: " + ", ".join(header_parts)
    table = led.to_string(index=False)
    return header + "\n" + table
