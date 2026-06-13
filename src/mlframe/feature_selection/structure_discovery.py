"""Standalone structure-discovery / EDA tool: ``discover_structure(X, y) -> StructureReport``.

Surfaces the hidden DISCRETE structural relationships the four FE detectors find -- turning them from silent feature-emitters into a
user-facing insight tool. A correlation matrix or a tree-importance plot never reveals that ``y`` depends on ``gcd(price, quantity)``,
on ``(a + b) mod 7``, on a regime switch ``c > tau ? a : b``, or on ``argmax(a, b, c)``; these relationships are number-theoretic /
non-smooth / comparison-based, so smooth bases and a single correlation coefficient cannot express them. This runs the four shipped
detectors on ``(X, y)`` and returns a ranked, human-readable report of what it found.

It REUSES the detectors verbatim -- it does NOT reimplement detection, the MI scoring, the permutation-null gate, or the
best-existing-op baseline. Each detector already returns only ``responded`` hits (cleared both the smooth-basis / best-existing-op
baseline AND a permutation-null upper band), so the 0-false-discovery guarantee on structureless data is INHERITED: a smooth / linear /
noise frame returns an EMPTY report.

Usage
-----
>>> import numpy as np, pandas as pd
>>> from mlframe.feature_selection import discover_structure
>>> rng = np.random.default_rng(0)
>>> a = rng.integers(1, 40, 2000); b = rng.integers(1, 40, 2000)
>>> X = pd.DataFrame({"price": a, "quantity": b, "noise": rng.normal(size=2000)})
>>> y = np.gcd(a, b)                      # y is literally gcd(price, quantity)
>>> report = discover_structure(X, y)
>>> print(report)                         # ranked human-readable block
>>> report.relations[0].kind, report.relations[0].columns
('gcd', ('price', 'quantity'))

Cheap: it is the same bounded cheap-first scans the operators run inside MRMR (budget-guarded by ``max_int_cols`` / the operator
internal caps), with no full MRMR fit. On n=2000, p<=20 it runs in ~1-2s.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "DiscoveredRelation",
    "StructureReport",
    "discover_structure",
    "structure_report_from_recipes",
]

# The families this tool can surface. ``include`` accepts the coarse family names; each maps to one or more fine ``kind`` labels.
_FAMILIES = ("modular", "lattice", "argmax", "gate")


@dataclass(frozen=True)
class DiscoveredRelation:
    """One discovered structural relationship between source columns of X and the target y.

    Attributes
    ----------
    kind
        Fine relationship label: one of ``gcd`` / ``lcm`` / ``bitwise_and`` / ``modular`` / ``parity`` / ``period`` / ``argmax`` /
        ``gate_select`` / ``gate_mask``. (``modular`` covers ``(a+b) mod m`` style residues; ``parity`` is ``mod 2``; ``period`` is a
        single-column hidden integer period.)
    columns
        The SOURCE column NAMES (not indices) the relationship is over, in operand order.
    parameter
        The detected structural parameter: the modulus ``m`` for modular kinds, the threshold ``tau`` for gate kinds, ``None`` otherwise.
    mi
        Engineered-column plug-in MI vs y (nats) -- how strongly the discovered structure explains y.
    baseline_mi
        MI of the best EXISTING op / raw operand the relationship is gated against -- the floor the structure had to beat.
    lift
        ``mi / baseline_mi`` (clamped; ``inf`` when the baseline is ~0). How much the discovered structure adds over what a selector
        already had from the raw columns / cheap ops.
    description
        One-line human-readable summary, e.g. ``"y depends on gcd(price, quantity)  [MI 0.47, lift 6.9x]"``.
    """

    kind: str
    columns: tuple[str, ...]
    parameter: Optional[float]
    mi: float
    baseline_mi: float
    lift: float
    description: str


@dataclass
class StructureReport:
    """Ranked report of the discrete structural relationships ``discover_structure`` found in ``(X, y)``.

    ``relations`` is ranked by MI descending (then lift), capped at ``top_k``. Empty when nothing responded -- the headline
    anti-false-discovery guarantee on smooth / linear / noise frames. ``str(report)`` / ``report.summary()`` render a readable block;
    ``skipped`` carries a reason string when the scan was skipped (e.g. 2D y)."""

    relations: list[DiscoveredRelation] = field(default_factory=list)
    n_columns: int = 0
    n_integer_columns: int = 0
    skipped: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.relations)

    def __len__(self) -> int:
        return len(self.relations)

    def __iter__(self):
        return iter(self.relations)

    def summary(self) -> str:
        """Human-readable ranked block."""
        header = (
            f"StructureReport: {len(self.relations)} discovered relationship(s) "
            f"over {self.n_columns} columns ({self.n_integer_columns} integer-eligible)."
        )
        if self.skipped:
            return f"{header}\n  (scan skipped: {self.skipped})"
        if not self.relations:
            return f"{header}\n  no discrete structural relationships detected."
        lines = [header]
        for i, r in enumerate(self.relations, 1):
            lines.append(f"  {i:>2}. {r.description}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _lift(mi: float, baseline_mi: float) -> float:
    """``mi / baseline_mi`` clamped to a finite-ish lift; ``inf`` when the baseline is ~0 (the structure is the ENTIRE signal)."""
    if baseline_mi <= 1e-9:
        return float("inf")
    return float(mi / baseline_mi)


def _fmt_lift(lift: float) -> str:
    return "inf" if not np.isfinite(lift) else f"{lift:.1f}x"


def _modular_kind(op: str, modulus: int) -> str:
    """Fine kind label for a modular hit: ``parity`` (mod 2), ``period`` (single-column self op), else ``modular``."""
    if int(modulus) == 2:
        return "parity"
    if op == "self":
        return "period"
    return "modular"


def _modular_relations(X, y, names_ok, nbins, seed, max_int_cols):
    from .filters._pairwise_modular_fe import cheap_modular_scan, escalate_modulus, _is_integer_col

    int_cols = [c for c in names_ok if _is_integer_col(np.asarray(X[c]))]
    if len(int_cols) > int(max_int_cols) or len(int_cols) < 1:
        return []
    out = []
    for h in cheap_modular_scan(X, y, int_cols, nbins=nbins, seed=seed):
        if not h.responded:
            continue
        best_m, best_mi, _ = escalate_modulus(X, y, h, nbins=nbins)
        kind = _modular_kind(h.op, best_m)
        lift = _lift(best_mi, h.baseline_mi)
        cols = tuple(str(c) for c in h.cols)
        opdesc = {"self": cols[0], "sum": " + ".join(cols), "diff": " - ".join(cols),
                  "prod": " * ".join(cols), "sum3": " + ".join(cols)}.get(h.op, " ? ".join(cols))
        desc = f"y depends on ({opdesc}) mod {best_m}  [{kind}, MI {best_mi:.3f}, lift {_fmt_lift(lift)}]"
        out.append(DiscoveredRelation(kind, cols, float(best_m), float(best_mi), float(h.baseline_mi), lift, desc))
    return out


def _lattice_relations(X, y, names_ok, nbins, seed, max_int_cols):
    from .filters._integer_lattice_fe import cheap_integer_lattice_scan
    from .filters._pairwise_modular_fe import _is_integer_col

    int_cols = [c for c in names_ok if _is_integer_col(np.asarray(X[c]))]
    if len(int_cols) > int(max_int_cols) or len(int_cols) < 2:
        return []
    out = []
    for h in cheap_integer_lattice_scan(X, y, int_cols, nbins=nbins, seed=seed):
        if not h.responded:
            continue
        lift = _lift(h.feat_mi, h.operand_floor)
        cols = tuple(str(c) for c in h.cols)
        desc = f"y depends on {h.op}({', '.join(cols)})  [MI {h.feat_mi:.3f}, lift {_fmt_lift(lift)}]"
        out.append(DiscoveredRelation(h.op, cols, None, float(h.feat_mi), float(h.operand_floor), lift, desc))
    return out


def _argmax_relations(X, y, names_ok, nbins, seed):
    from .filters._conditional_gate_fe import cheap_row_argmax_scan

    out = []
    for h in cheap_row_argmax_scan(X, y, names_ok, nbins=nbins, seed=seed):
        if not h.responded:
            continue
        lift = _lift(h.feat_mi, h.operand_floor)
        cols = tuple(str(c) for c in h.cols)
        desc = f"y depends on which of ({', '.join(cols)}) is largest (argmax)  [MI {h.feat_mi:.3f}, lift {_fmt_lift(lift)}]"
        out.append(DiscoveredRelation("argmax", cols, None, float(h.feat_mi), float(h.operand_floor), lift, desc))
    return out


def _gate_relations(X, y, names_ok, nbins, seed):
    from .filters._conditional_gate_fe import cheap_conditional_gate_scan

    out = []
    for h in cheap_conditional_gate_scan(X, y, names_ok, nbins=nbins, seed=seed):
        if not h.responded:
            continue
        lift = _lift(h.feat_mi, h.baseline_mi)
        cols = tuple(str(c) for c in h.cols)
        if h.mode == "select":
            a, b, c = cols
            kind = "gate_select"
            desc = f"regime switch: y ~ ({a} if {c}>{h.tau:.3g} else {b})  [MI {h.feat_mi:.3f}, lift {_fmt_lift(lift)}]"
        else:
            a, c = cols
            kind = "gate_mask"
            desc = f"masked interaction: y ~ {a} * 1[{c}>{h.tau:.3g}]  [MI {h.feat_mi:.3f}, lift {_fmt_lift(lift)}]"
        out.append(DiscoveredRelation(kind, cols, float(h.tau), float(h.feat_mi), float(h.baseline_mi), lift, desc))
    return out


def discover_structure(
    X,
    y,
    *,
    nbins: int = 12,
    max_int_cols: int = 30,
    top_k: int = 20,
    include: Sequence[str] = _FAMILIES,
    seed: int = 0,
) -> StructureReport:
    """Discover hidden DISCRETE structural relationships between the columns of ``X`` and the target ``y``.

    Runs the four shipped FE detectors -- pairwise/n-way MODULAR (``(a+b) mod m`` / parity / hidden period), integer-LATTICE
    (``gcd`` / ``lcm`` / ``bitwise_and``), row-ARGMAX (which of a, b, c is largest), and conditional-GATE (regime switch
    ``c>tau ? a : b`` / masked interaction ``1[c>tau]*a``) -- and returns a ranked, human-readable :class:`StructureReport` of what
    responded. Each detector's own permutation-null + best-existing-op gate is reused unchanged, so a smooth / linear / noise frame
    yields an EMPTY report (0 false discovery, inherited).

    Parameters
    ----------
    X : pandas.DataFrame (or numpy ndarray)
        Feature frame. An ndarray is wrapped with positional string names ``f0, f1, ...``.
    y : 1D array-like
        Target (classification or regression). A continuous y is quantile-binned internally (via the operators' shared
        ``bin_y_for_class_mi``) before MI scoring. A 2D y (multilabel / multi-target) is SKIPPED with a warning (mirrors the operators).
    nbins : int, default 12
        Binning resolution for the plug-in MI relevance scoring.
    max_int_cols : int, default 30
        Budget guard: the modular / lattice sweeps are skipped when the integer-eligible column count exceeds this (the wide-frame cost
        guard the operators use). Argmax / gate carry their own internal relevance-pruned budgets.
    top_k : int, default 20
        Cap on the number of returned relations (ranked by MI descending, then lift).
    include : sequence of str, default ("modular", "lattice", "argmax", "gate")
        Which detector families to run.
    seed : int, default 0
        Permutation-null seed (passed to each detector).

    Returns
    -------
    StructureReport
        Ranked list of :class:`DiscoveredRelation`; empty when nothing responds.

    Examples
    --------
    >>> report = discover_structure(X, y)
    >>> print(report)
    >>> for rel in report:
    ...     print(rel.kind, rel.columns, rel.parameter, rel.mi)
    """
    from .filters._fe_accuracy_gate import class_mi_fe_applicable, bin_y_for_class_mi

    import pandas as pd

    if not isinstance(X, pd.DataFrame):
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"discover_structure: X must be 2D; got shape {X_arr.shape}")
        X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])

    n_columns = X.shape[1]
    y_arr = np.asarray(y)
    if not class_mi_fe_applicable(y_arr):
        warnings.warn(
            "discover_structure: 2D / multi-target y is unsupported by the discrete structure detectors; "
            "returning an empty report. Pass a 1D classification or regression target.",
            stacklevel=2,
        )
        return StructureReport(relations=[], n_columns=n_columns, n_integer_columns=0, skipped="2D / multi-target y")

    yb = bin_y_for_class_mi(y_arr.ravel(), nbins=nbins)
    names = [str(c) for c in X.columns]

    from .filters._pairwise_modular_fe import _is_integer_col

    n_int = sum(1 for c in X.columns if _is_integer_col(np.asarray(X[c])))

    include = tuple(include)
    relations: list[DiscoveredRelation] = []
    try:
        if "modular" in include:
            relations += _modular_relations(X, yb, names, nbins, seed, max_int_cols)
        if "lattice" in include:
            relations += _lattice_relations(X, yb, names, nbins, seed, max_int_cols)
        if "argmax" in include:
            relations += _argmax_relations(X, yb, names, nbins, seed)
        if "gate" in include:
            relations += _gate_relations(X, yb, names, nbins, seed)
    except Exception as exc:  # pragma: no cover - a detector failure must not break the EDA call
        logger.warning("discover_structure: detector raised (%s); returning partial report", exc)

    relations.sort(key=lambda r: (r.mi, r.lift if np.isfinite(r.lift) else 1e18), reverse=True)
    return StructureReport(
        relations=relations[: int(top_k)],
        n_columns=n_columns,
        n_integer_columns=int(n_int),
    )


def structure_report_from_recipes(recipes, *, n_columns: int = 0) -> StructureReport:
    """Assemble a :class:`StructureReport` from the frozen ``EngineeredRecipe`` objects a fitted MRMR already emitted.

    Reads ONLY the metadata the four detectors froze at fit time (``kind`` / ``src_names`` / ``extra`` op / modulus / tau / mode) -- no
    re-scan, no y, near-free. Used by ``MRMR.discovered_structure_`` so a user who already fit MRMR can read what discrete structure was
    found. MI / baseline / lift are not available from the recipe (the fit did not freeze the scan's MI), so they are reported as ``nan``;
    the kind + columns + parameter (the structural facts) are exact."""
    relations: list[DiscoveredRelation] = []
    for r in recipes:
        kind = getattr(r, "kind", None)
        cols = tuple(str(c) for c in getattr(r, "src_names", ()))
        extra = getattr(r, "extra", {}) or {}
        if kind == "pairwise_modular":
            op, m = str(extra.get("op", "")), int(extra.get("modulus", 0) or 0)
            fine = _modular_kind(op, m) if m else "modular"
            opdesc = " ".join(cols) if op == "self" else f" {op} ".join(cols)
            desc = f"y depends on ({opdesc}) mod {m}  [{fine}]"
            relations.append(DiscoveredRelation(fine, cols, float(m), float("nan"), float("nan"), float("nan"), desc))
        elif kind == "pairwise_integer_lattice":
            op = str(extra.get("op", ""))
            desc = f"y depends on {op}({', '.join(cols)})"
            relations.append(DiscoveredRelation(op, cols, None, float("nan"), float("nan"), float("nan"), desc))
        elif kind == "row_argmax":
            desc = f"y depends on which of ({', '.join(cols)}) is largest (argmax)"
            relations.append(DiscoveredRelation("argmax", cols, None, float("nan"), float("nan"), float("nan"), desc))
        elif kind == "conditional_gate":
            mode, tau = str(extra.get("mode", "")), float(extra.get("tau", float("nan")))
            if mode == "select" and len(cols) == 3:
                a, b, c = cols
                desc = f"regime switch: y ~ ({a} if {c}>{tau:.3g} else {b})"
                relations.append(DiscoveredRelation("gate_select", cols, tau, float("nan"), float("nan"), float("nan"), desc))
            elif mode == "mask" and len(cols) == 2:
                a, c = cols
                desc = f"masked interaction: y ~ {a} * 1[{c}>{tau:.3g}]"
                relations.append(DiscoveredRelation("gate_mask", cols, tau, float("nan"), float("nan"), float("nan"), desc))
    return StructureReport(relations=relations, n_columns=int(n_columns), n_integer_columns=0)
