"""How many engineered columns each FE group produced, and how many each FE stage added.

A group is the recipe's ``kind``, refined by its basis where the kind has several (``orth_univariate:rbf``, ``orth_univariate:He``, ``hermite_pair:He``). Three counts per group are kept in
``MRMR.fe_family_counts_``: ``emitted`` by the cascades, ``kept_for_screen`` (what the dedup and the accuracy gate left for the screen) and ``selected`` (in the final support). The same
numbers by stage are in ``MRMR.fe_step_counts_``: the cascades first, then one row per FE step from 1 to ``fe_max_steps``. Both are also in ``fe_provenance_.attrs`` and in ``get_fe_report()``.
Pure metadata: nothing here reads the data or changes the selection.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Iterable, Optional

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

__all__ = ["recipe_group", "group_counts", "registry_recipes", "record_fe_step_counts", "record_cascade_emitted", "record_family_counts", "family_counts_section"]


def recipe_group(recipe: Any) -> str:
    """``kind`` of ``recipe``, with ``:<basis>`` appended when its ``extra`` names a basis."""
    kind = str(getattr(recipe, "kind", "") or "unknown")
    extra = getattr(recipe, "extra", None)
    basis = extra.get("basis") if isinstance(extra, Mapping) else None
    return f"{kind}:{basis}" if basis else kind


def group_counts(recipes: Iterable[Any]) -> dict[str, int]:
    """``{group: number of distinct recipe names}`` over ``recipes``, largest first."""
    by_name: dict[str, Any] = {}
    for r in recipes:
        name = getattr(r, "name", None)
        if name is not None:
            by_name.setdefault(str(name), r)
    counts: dict[str, int] = {}
    for r in by_name.values():
        g = recipe_group(r)
        counts[g] = counts.get(g, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def registry_recipes(store: Any) -> list[Any]:
    """Every recipe held by the per-family registries (dicts of ``name -> recipe``) of one fit."""
    out: list[Any] = []
    for value in vars(store).values():
        if isinstance(value, dict):
            out.extend(r for r in value.values() if hasattr(r, "kind") and hasattr(r, "name"))
    return out


def record_cascade_emitted(mrmr_self: Any, store: Any) -> None:
    """Remember the groups the cascades emitted (called when they finish, before the dedup and the gates remove anything)."""
    try:
        mrmr_self._fe_emitted_by_group_ = group_counts(registry_recipes(store))
        mrmr_self._fe_step_counts_acc_ = []
    except Exception as exc:  # metadata only
        mrmr_self._fe_emitted_by_group_ = {}
        mrmr_self._fe_step_counts_acc_ = []
        log_throttle(logger, "fe_group_counts.cascade", logging.WARNING, "fe_group_counts: the cascade counts could not be taken (%r)", exc)


def record_fe_step_counts(mrmr_self: Any, step: int, recommended: int, new_recipes: Iterable[Any]) -> None:
    """One FE step finished: its recommended-feature count and the groups of the recipes it added."""
    try:
        new = list(new_recipes)
        acc = getattr(mrmr_self, "_fe_step_counts_acc_", None)
        if acc is None:
            acc = mrmr_self._fe_step_counts_acc_ = []
        acc.append({"stage": int(step), "recommended": int(recommended), "new_columns": len(new), "by_group": group_counts(new)})
    except Exception as exc:  # metadata only
        log_throttle(logger, "fe_group_counts.step", logging.WARNING, "fe_group_counts: the counts of FE step %s could not be taken (%r)", step, exc)


def record_family_counts(mrmr_self: Any) -> None:
    """Fill ``fe_family_counts_`` and ``fe_step_counts_`` from what the fit recorded, and put both in ``fe_provenance_.attrs``."""
    try:
        emitted = dict(getattr(mrmr_self, "_fe_emitted_by_group_", None) or {})
        kept = group_counts(getattr(mrmr_self, "_produced_recipes_", None) or [])
        selected = group_counts(getattr(mrmr_self, "_engineered_recipes_", None) or [])
        groups = sorted(set(emitted) | set(kept) | set(selected), key=lambda g: (-emitted.get(g, 0), -kept.get(g, 0), g))
        family = {g: {"emitted": emitted.get(g, 0), "kept_for_screen": kept.get(g, 0), "selected": selected.get(g, 0)} for g in groups}
        steps = [{"stage": "cascades", "recommended": None, "new_columns": sum(emitted.values()), "by_group": emitted}]
        steps.extend(list(getattr(mrmr_self, "_fe_step_counts_acc_", None) or []))
        mrmr_self.fe_family_counts_ = family
        mrmr_self.fe_step_counts_ = steps
        prov = getattr(mrmr_self, "fe_provenance_", None)
        if prov is not None and hasattr(prov, "attrs"):
            prov.attrs["family_counts"] = family
            prov.attrs["step_counts"] = steps
    except Exception as exc:  # metadata only
        mrmr_self.fe_family_counts_ = {}
        mrmr_self.fe_step_counts_ = []
        log_throttle(logger, "fe_group_counts.final", logging.WARNING, "fe_group_counts: fe_family_counts_ could not be built (%r)", exc)


def family_counts_section(mrmr_self: Any) -> str:
    """The two blocks of :func:`get_fe_report`: engineered columns by group, then by FE stage (empty when the fit recorded none)."""
    family: Optional[dict] = getattr(mrmr_self, "fe_family_counts_", None)
    steps: Optional[list] = getattr(mrmr_self, "fe_step_counts_", None)
    if not family and not steps:
        return ""
    lines: list[str] = []
    if family:
        lines.append("FE columns by group (emitted by the cascades / kept for the screen / selected):")
        lines.extend(f"  {g}: {row['emitted']} / {row['kept_for_screen']} / {row['selected']}" for g, row in family.items())
    if steps:
        lines.append("FE columns by stage:")
        for row in steps:
            rec = "" if row.get("recommended") is None else f", {row['recommended']} recommended"
            top = ", ".join(f"{g}={n}" for g, n in list((row.get("by_group") or {}).items())[:6])
            lines.append(f"  {row['stage']}: {row['new_columns']} new{rec}" + (f" ({top})" if top else ""))
    return "\n" + "\n".join(lines)
