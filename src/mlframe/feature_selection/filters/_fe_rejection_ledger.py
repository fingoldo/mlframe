"""Per-gate FE REJECTION LEDGER (the rejection side of the Layer-54 provenance surface).

User-facing question this layer answers: "WHY was this engineered candidate dropped?".
``_mrmr_fe_provenance.py`` records the SURVIVING FE recipes (and - via
``_produced_recipes_`` - the produced-but-screened survivors with their origin). The
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
stability vote already counts). The ledger only RECORDS - it never recomputes an MI, a
permutation null, or a CMI. Cost is one ``list.append`` of a small dict per rejected
candidate.

DEFAULT-ON, MEMORY-CAPPED
-------------------------
Every fitted MRMR carries ``fe_rejection_ledger_`` - there is no opt-in flag (mirrors
``fe_provenance_``). To bound memory on pathological wide frames (hundreds of thousands of
rejected pair candidates on very-wide-p fits) the raw record list is capped at
``FE_REJECTION_LEDGER_CAP`` records (or ``MRMR(fe_rejection_ledger_cap=...)`` when set on the
instance); once the cap is hit further records are dropped and a single ``_capped`` marker
row is recorded so the truncation is never silent (per the no-silent-truncation rule). The
default cap is 500k records (~100-200 MB worst case) so realistic fits, including very-wide-p
ones, keep the full ledger; raise it further via the constructor param if needed.
"""
from __future__ import annotations

import logging as _logging
from typing import Any

import numpy as np
import pandas as pd
import logging
from mlframe.utils.log_throttle import log_throttle

logger = _logging.getLogger("mlframe.feature_selection.filters.mrmr")


# Canonical gate labels. Public surface; tests pin the membership.
FE_GATE_LABELS = (
    "marginal_pair_mi_prescreen",  # gate 1: joint-MI prevalence ratio pre-screen (~1.05 / synergy 1.5)
    "order2_maxt_floor",  # gate 2: order-2/3 max-T permutation null floor on the joint MI
    "engineered_mi_prevalence",  # gate 6: best_mi / pair_mi vs fe_min_engineered_mi_prevalence (~0.97)
    "marginal_uplift_floor",  # gate 5: marginal-uplift / joint-recovery (abs-MAD) fallback floor
    "cmi_redundancy",  # gate 3: S5 conditional-MI redundancy gate
    "stability_vote",  # gate 4: cross-fold stability quorum vote
    "operand_uplift_floor",  # gate: engineered MI must beat the larger operand by its permutation-null uplift bar
    "binagg_source_redundancy",  # binned-numeric-agg candidate dropped as redundant with its own source column
    "fast_search_cross_group_overmaterialization",  # fast-search cross-group candidate over the materialisation budget
    "gate_composite_overmaterialization",  # gate composite over the materialisation budget
    "offset_product_gain",  # offset-product candidate below the gain floor
    "oof_warp_gain",  # out-of-fold warp candidate below the gain floor
    "pair_candidate_generation",  # pair never proposed by candidate generation
    "row_stat_gain",  # row-statistic candidate below the gain floor
    "inert_leaf",  # pair candidate with a raw leaf that carries no information about it beyond its other leaves
    "dominated_by_operands",  # pair candidate that adds no information about the target beyond its other raw leaves
    "additive_fusion",  # two additively separable halves whose fusion does not beat the stronger half
    "residual_rescue_prune",  # residual rescue: a raw or engineered column without held-out gain given the rest of the selection
    "final_subsumption",  # end-of-fit pass: a column carrying no information beyond the compounds
    "raw_redundancy_drop",  # raw operand conditionally subsumed by a surviving engineered child
    "ledger_capped",  # synthetic marker: records dropped past the memory cap
)


# Raw-record list memory cap (module-level fallback; override per-instance via
# ``MRMR(fe_rejection_ledger_cap=...)``). Realistic fits stay far below this; the cap only
# guards a pathological wide-frame fit from accumulating an unbounded ledger. When exceeded a
# single ``ledger_capped`` marker row is recorded so the truncation is visible (never silent).
FE_REJECTION_LEDGER_CAP: int = 500_000


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
    "operand_names",
)


def _safe_float(value: Any) -> float:
    """Coerce to float; NaN on failure so a malformed record never raises at DataFrame build."""
    try:
        return float(value)
    except (TypeError, ValueError) as _exc:
        logger.debug("_safe_float fell back to its default after %s", type(_exc).__name__)
        return float("nan")


def _operands_str(operands: Any) -> str:
    """Stable stringification of the operand collection (sorted dicts not needed; preserve order)."""
    try:
        if operands is None:
            return "()"
        if isinstance(operands, (list, tuple)):
            return "(" + ", ".join(str(o) for o in operands) + ")"
        return str(operands)
    except Exception as e:
        log_throttle(logger, "_fe_rejection_ledger._operands_str", logging.WARNING, "_operands_str: could not stringify operand collection %r: %s", type(operands), e)
        return "<unstringifiable>"


_ADMISSION_COLUMNS: tuple[str, ...] = ("candidate", "operands", "operator", "stage", "observed", "best_mi", "pair_mi", "reason", "step", "operand_names")


def record_fe_admission(
    mrmr_self: Any, *, candidate: Any, operands: Any = None, operator: Any = None, stage: str = "pair_gate", observed: Any = float("nan"),
    best_mi: Any = float("nan"), pair_mi: Any = float("nan"), reason: str = "", step: int = -1,
) -> None:
    """Append ONE admission record (a candidate that passed the gates of ``stage``) to ``mrmr_self._fe_admission_records_``.

    The answer to "where did this column come from, and by how much did it pass": ``observed`` is the gate ratio (engineered MI over the pair's joint MI) and ``best_mi`` / ``pair_mi`` its
    two terms. Capped like the rejection list; swallows its own errors.
    """
    try:
        records = getattr(mrmr_self, "_fe_admission_records_", None)
        if records is None:
            records = []
            mrmr_self._fe_admission_records_ = records
        _cap_cfg = getattr(mrmr_self, "fe_rejection_ledger_cap", None)
        if len(records) >= int(FE_REJECTION_LEDGER_CAP if _cap_cfg is None else _cap_cfg):
            return
        records.append(
            {
                "candidate": str(candidate), "operands": _operands_str(operands), "operator": (None if operator is None else str(operator)), "stage": str(stage),
                "observed": _safe_float(observed), "best_mi": _safe_float(best_mi), "pair_mi": _safe_float(pair_mi), "reason": str(reason), "step": int(step), "operand_names": "",
            }
        )
    except Exception as exc:  # pragma: no cover - instrumentation must never break the fit
        logger.debug("MRMR FE admission ledger record failed (%s); skipping.", exc)


def drain_pair_records(mrmr_self: Any, records: Any, step: int) -> None:
    """Route the per-pair records of one FE chunk: admitted candidates to the admission list, every other record to the rejection ledger."""
    for rec in records or ():
        if rec.get("verdict") == "admitted":
            record_fe_admission(
                mrmr_self, candidate=rec.get("candidate"), operands=rec.get("operands"), operator=rec.get("operator"), stage=rec.get("stage", "pair_gate"),
                observed=rec.get("observed", float("nan")), best_mi=rec.get("best_mi", float("nan")), pair_mi=rec.get("pair_mi", float("nan")), reason=rec.get("reason", ""), step=step,
            )
        else:
            record_fe_rejection(
                mrmr_self, gate=rec["gate"], candidate=rec.get("candidate"), operands=rec.get("operands"), operator=rec.get("operator"),
                observed=rec.get("observed", float("nan")), threshold=rec.get("threshold", float("nan")), reason=rec.get("reason", ""), step=step,
            )


def record_raw_redundancy_drop(mrmr_self: Any, name: str, cmi: float, floor: float, excess: float, marginal_excess: float) -> None:
    """Ledger row for a raw operand the redundancy sweep dropped: ``observed`` is its debiased conditional excess, ``threshold`` the share of its own marginal excess it had to retain."""
    record_fe_rejection(
        mrmr_self, gate="raw_redundancy_drop", candidate=name, operator="raw", observed=excess, threshold=marginal_excess,
        reason=f"conditional MI {cmi:.5f} against permutation floor {floor:.5f}: subsumed by a surviving engineered child",
    )


def record_fe_rejection(
    mrmr_self: Any,
    *,
    gate: str,
    candidate: Any,
    operands: Any = None,
    operand_names: str = "",
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
    the effective cap (``mrmr_self.fe_rejection_ledger_cap`` if set, else module-default
    ``FE_REJECTION_LEDGER_CAP``) a single ``ledger_capped`` marker is appended (so the cap is
    never silent) and subsequent records are dropped. Swallows its OWN errors - an
    instrumentation failure must never break the FE search.
    """
    try:
        records = getattr(mrmr_self, "_fe_rejection_records_", None)
        if records is None:
            records = []
            mrmr_self._fe_rejection_records_ = records
        # Configurable per-instance (``fe_rejection_ledger_cap``, default None -> module default). At
        # very wide p (hundreds-of-thousands-of-columns use case) a fixed 50k cap is reached
        # after tens of thousands of ALREADY-expensive candidate evaluations, losing FE debugging
        # visibility for the rest of the fit even though each record is small (~200-400 bytes): raising
        # the cap 10x costs low-single-digit-MB extra, cheap on any real host.
        _cap_cfg = getattr(mrmr_self, "fe_rejection_ledger_cap", None)
        effective_cap = int(FE_REJECTION_LEDGER_CAP if _cap_cfg is None else _cap_cfg)
        n = len(records)
        if n >= effective_cap:
            # Record the truncation EXACTLY once (no silent truncation), then stop appending.
            if n == effective_cap:
                records.append(
                    {
                        "candidate": "<ledger cap reached>",
                        "operands": "()",
                        "operator": None,
                        "gate": "ledger_capped",
                        "observed": float(effective_cap),
                        "threshold": float(effective_cap),
                        "margin": 0.0,
                        "reason": (f"rejection ledger reached its {effective_cap}-record cap; " "further rejected candidates are not recorded for this fit"),
                        "step": int(step),
                    }
                )
                logger.warning(
                    "MRMR FE rejection ledger hit its %d-record cap; further rejected " "candidates this fit are not recorded. "
                    "Raise via MRMR(fe_rejection_ledger_cap=...) if full-ledger visibility is needed on very wide pools.",
                    effective_cap,
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
                "operand_names": str(operand_names or ""),
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
    _resolve_index_columns(mrmr_self, df)
    return df


def _resolve_index_columns(mrmr_self: Any, df: pd.DataFrame) -> None:
    """Add readable ``operand_names`` beside the raw positional ``operands``, in place.

    The pair-level gates record positions into the FE working matrix, not names, so a row reads
    ``(2, 3)`` and the reader has to reconstruct the column order to learn it means ``(c, d)``. Worse, a
    position at or beyond the raw column count is an ENGINEERED column appended to the pool, which is
    indistinguishable from a raw one by looking at the number. Both cost real debugging time, so resolve
    what can be resolved and leave the rest visibly marked rather than silently ambiguous.
    """
    try:
        # feature_names_in_ is an ndarray after fit; ``arr or []`` raised "truth value ... ambiguous".
        _fni = getattr(mrmr_self, "feature_names_in_", None)
        names = [] if _fni is None else [str(n) for n in _fni]
        if not names or "operands" not in df.columns:
            return

        def _name(v: Any) -> str:
            """One operand: a raw name, an engineered marker, or the value unchanged if it is not a position."""
            if isinstance(v, str):
                return v
            try:
                i = int(v)
            except (TypeError, ValueError) as _exc:
                logger.debug("_name fell back to its default after %s", type(_exc).__name__)
                return str(v)
            return names[i] if 0 <= i < len(names) else f"<engineered#{i}>"

        def _row(ops: Any) -> str:
            """The whole operand tuple, rendered."""
            if ops is None:
                return ""
            if isinstance(ops, str) and ops.startswith("(") and ops.endswith(")"):
                # the records store ``_operands_str`` text, "(0, 1)"; resolve each piece
                ops = [int(p) if p.isdigit() else p for p in ops[1:-1].split(", ") if p]
            if isinstance(ops, np.ndarray):
                ops = ops.tolist()
            if isinstance(ops, (list, tuple)):
                return "(" + ", ".join(_name(o) for o in ops) + ")" if len(ops) else ""
            return _name(ops)

        _resolved = [_row(o) for o in df["operands"]]
        if any(_resolved):
            df["operand_names"] = _resolved
    except Exception as exc:  # nosec B110 - a readability aid must never break the ledger
        logger.debug("fe_rejection_ledger_: operand-name resolution failed: %s", exc)


def populate_fe_admission_ledger(mrmr_self: Any) -> None:
    """Stash ``fe_admission_ledger_`` (one row per candidate a gate admitted) on the estimator; never raises."""
    try:
        records = getattr(mrmr_self, "_fe_admission_records_", None) or []
        df = pd.DataFrame(list(records), columns=list(_ADMISSION_COLUMNS)) if records else pd.DataFrame({c: pd.Series([], dtype=object) for c in _ADMISSION_COLUMNS})
        _resolve_index_columns(mrmr_self, df)
        mrmr_self.fe_admission_ledger_ = df
    except Exception as exc:
        logger.warning("MRMR.fit: fe_admission_ledger_ population failed (%s: %s); it stays empty.", type(exc).__name__, exc)
        mrmr_self.fe_admission_ledger_ = pd.DataFrame({c: pd.Series([], dtype=object) for c in _ADMISSION_COLUMNS})


def populate_fe_rejection_ledger(mrmr_self: Any) -> None:
    """Run ``compute_fe_rejection_ledger`` and stash the result on the estimator.

    Mirrors ``populate_fe_provenance``: wrapped here so the parent module stays under its LOC
    budget and the fallback schema is centralised. Never raises.
    """
    try:
        _led = compute_fe_rejection_ledger(mrmr_self)
        _resolve_index_columns(mrmr_self, _led)
        mrmr_self.fe_rejection_ledger_ = _led
        populate_fe_admission_ledger(mrmr_self)
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
    return str(header + "\n" + table)


def _drain_fe_rejection_records(mrmr_self: Any, records: list, step: int) -> None:
    """Append caller-collected rejection dicts to the MRMR ledger, one ``record_fe_rejection`` call each."""
    for rec in records:
        record_fe_rejection(mrmr_self, step=int(step), **rec)
