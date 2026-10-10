"""Wire a new FE operator into MRMR: the mechanical part of the factory checklist (README section "Checklist for adopting an operator").

An operator that has a cascade stage is registered in about fifteen places (recipe kind and dispatch, provenance, roster, dedup lists, gates, state, recipe registry, cascade call, constructor
parameters, setstate defaults, fuzz-combo axis). Each edit inserts lines AFTER the lines of an operator that is already wired (``prev``), so the anchors are the previous operator's own lines.
``plan`` lists the edits and checks that every anchor occurs exactly as often as expected (a dry run); ``apply`` writes them. What stays manual: the operator's own module(s), its stage module
(a copy of ``_mrmr_fit_impl/_fe_stage_row_stat.py`` with the names changed), the tests, ``python -m mlframe.training.fs_params._generate`` and the CHANGELOG / audit entries.

Usage::

    python -m mlframe.feature_selection._benchmarks.fe_operator_factory.wiring --kind my_kind --family my_family --flag fe_my_family_enable --fn apply_my_recipe --module _my_family_fe --prev row_stat

``--prev`` names a previous operator by its family (``row_stat`` or ``oof_warp``); the others are derived from the table in ``PREVIOUS``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["Operator", "PREVIOUS", "plan", "apply", "main"]

REPO = Path(__file__).resolve().parents[5]
FILTERS = REPO / "src" / "mlframe" / "feature_selection" / "filters"
FUZZ = REPO / "tests" / "training"


@dataclass(frozen=True)
class Operator:
    """Names of one operator: recipe ``kind``, ``family`` (registry / stage name), constructor ``flag``, roster attribute, replay function and its module."""

    kind: str
    family: str
    flag: str
    fn: str
    module: str

    @property
    def attr(self) -> str:
        """Roster attribute written by the stage."""
        return f"{self.family}_features_"

    @property
    def cfg(self) -> str:
        """Fuzz-combo field name."""
        return f"mrmr_{self.flag}_cfg"


PREVIOUS = {
    "row_stat": Operator("row_stat", "row_stat", "fe_row_stat_enable", "apply_row_stat_recipe", "_row_stat_fe"),
    "oof_warp": Operator("oof_warp1d", "oof_warp", "fe_oof_warp_enable", "apply_oof_warp1d_recipe", "_oof_warp_fe"),
}


def _edits(new: Operator, prev: Operator, params: "list[tuple[str, str]]", comment: str) -> "list[tuple[Path, str, str, int]]":
    """``(file, anchor, replacement, expected occurrences)`` for every registration of ``new`` after ``prev``."""
    p, n = prev, new
    ctor = "".join(f"        {n.flag.replace('_enable', '')}_{name}: {typ}\n" for name, typ in params)
    ctor_defaults = "".join(f'    "{n.flag.replace("_enable", "")}_{name}": {typ.split("=")[1].strip().rstrip(",")},\n' for name, typ in params)
    stage_call = lambda f: f"    X_acc = _fe_merge_new_columns(X_acc, _stage_{f.family}(self, _fe_family_on, X, _y_np, _raw_input_cols_pre_fe, _{f.family}_pre_recipes, verbose), X)\n"  # noqa: E731
    return [
        (FILTERS / "engineered_recipes/_recipe_core.py", f'"{p.kind}"', f'"{p.kind}", "{n.kind}"', 1),
        (FILTERS / "engineered_recipes/_recipe_dispatch.py", f'    "{p.kind}": _routed(".._{p.module.lstrip("_")}", "{p.fn}"),\n', f'    "{p.kind}": _routed(".._{p.module.lstrip("_")}", "{p.fn}"),\n    "{n.kind}": _routed(".._{n.module.lstrip("_")}", "{n.fn}"),\n', 1),
        (FILTERS / "_mrmr_fe_provenance.py", f'    "{p.kind}": "extra_fe",\n', f'    "{p.kind}": "extra_fe",\n    "{n.kind}": "extra_fe",\n', 1),
        (FILTERS / "_mrmr_fe_provenance.py", f'    ("{p.attr}", "extra_fe"),\n', f'    ("{p.attr}", "extra_fe"),\n    ("{n.attr}", "extra_fe"),\n', 1),
        (FILTERS / "_mrmr_fit_impl/_fe_roster_attrs.py", f'    "{p.attr}",\n', f'    "{p.attr}",\n    "{n.attr}",\n', 1),
        (FILTERS / "_mrmr_fit_impl/_fit_impl_stages/_engineered_dedup.py", f'    "{p.family}",\n', f'    "{p.family}",\n    "{n.family}",\n', 2),
        (FILTERS / "_mrmr_fit_impl/_fit_impl_stages/_state.py", f'    "{p.family}",\n', f'    "{p.family}",\n    "{n.family}",\n', 1),
        (FILTERS / "_mrmr_fit_impl/_fit_impl_stages/_engineered_gates.py", f"                        recipes.{p.family},\n", f"                        recipes.{p.family},\n                        recipes.{n.family},\n", 1),
        (FILTERS / "_mrmr_fit_impl/_fit_impl_core.py", f"    recipes.{p.family} = {{}}\n", f"    recipes.{p.family} = {{}}\n    recipes.{n.family} = {{}}\n", 1),
        (FILTERS / "_mrmr_fit_impl/_fe_stage_cascade_run.py", f"        _{p.family}_pre_recipes=recipes.{p.family},\n", f"        _{p.family}_pre_recipes=recipes.{p.family},\n        _{n.family}_pre_recipes=recipes.{n.family},\n", 1),
        (FILTERS / "_mrmr_fit_impl/_fe_stage_cascade_mid_b.py", f"from ._fe_stage_{p.family} import _stage_{p.family}\n", f"from ._fe_stage_{p.family} import _stage_{p.family}\nfrom ._fe_stage_{n.family} import _stage_{n.family}\n", 1),
        (FILTERS / "_mrmr_fit_impl/_fe_stage_cascade_mid_b.py", f"    _{p.family}_pre_recipes,\n", f"    _{p.family}_pre_recipes,\n    _{n.family}_pre_recipes,\n", 1),
        (FILTERS / "_mrmr_fit_impl/_fe_stage_cascade_mid_b.py", f"    self.{p.attr} = []\n", f"    self.{p.attr} = []\n    self.{n.attr} = []\n", 1),
        (FILTERS / "_mrmr_fit_impl/_fe_stage_cascade_mid_b.py", stage_call(p), stage_call(p) + f"\n    # {comment}\n" + stage_call(n), 1),
        (FILTERS / "mrmr/_mrmr_class.py", f"        {p.flag.replace('_enable', '')}_min_relative_gain: float = 0.05,\n", f"        {p.flag.replace('_enable', '')}_min_relative_gain: float = 0.05,\n        # {comment}\n        {n.flag}: bool = True,\n{ctor}", 1),
        (FILTERS / "mrmr/_mrmr_class.py", f'            "{p.flag}",  # legacy OFF; ctor ON\n', f'            "{p.flag}",  # legacy OFF; ctor ON\n            "{n.flag}",  # legacy OFF; ctor ON\n', 1),
        (FILTERS / "mrmr/_mrmr_setstate_defaults.py", f'    "{p.attr}": [],\n', f'    "{p.attr}": [],\n    "{n.flag}": False,\n{ctor_defaults}    "{n.attr}": [],\n', 1),
        (FUZZ / "_fuzz_combo/axes.py", f'    "{p.cfg}": (True, False),\n', f'    "{p.cfg}": (True, False),\n    "{n.cfg}": (True, False),\n', 1),
        (FUZZ / "_fuzz_combo/combo.py", f"    {p.cfg}: bool = True\n", f"    {p.cfg}: bool = True\n    {n.cfg}: bool = True\n", 1),
        (FUZZ / "_fuzz_combo/combo.py", f"            self.{p.cfg} if self.use_mrmr_fs else True,\n", f"            self.{p.cfg} if self.use_mrmr_fs else True,\n            self.{n.cfg} if self.use_mrmr_fs else True,\n", 1),
        (FUZZ / "_fuzz_combo/builders.py", f"    {p.flag}: bool = True,\n", f"    {p.flag}: bool = True,\n    {n.flag}: bool = True,\n", 1),
        (FUZZ / "_fuzz_combo/builders.py", f'        "{p.flag}": {p.flag},\n', f'        "{p.flag}": {p.flag},\n        "{n.flag}": {n.flag},\n', 1),
        (FUZZ / "_fuzz_combo/builders.py", f"        {p.flag}=combo.{p.cfg},\n", f"        {p.flag}=combo.{p.cfg},\n        {n.flag}=combo.{n.cfg},\n", 1),
        (FUZZ / "_fuzz_combo/enumerator.py", f'        {p.cfg}=axes.get("{p.cfg}", True),\n', f'        {p.cfg}=axes.get("{p.cfg}", True),\n        {n.cfg}=axes.get("{n.cfg}", True),\n', 1),
    ]


def plan(new: Operator, prev: Operator, params: "Optional[list[tuple[str, str]]]" = None, comment: str = "", root: Optional[Path] = None) -> "list[str]":
    """Check every anchor (a dry run) and return the problems found: an empty list means ``apply`` will succeed. ``root`` re-bases the repository (tests use a copy)."""
    problems = []
    for path, anchor, _repl, expected in _edits(new, prev, params or [], comment or f"{new.family} operator"):
        target = _rebase(path, root)
        if not target.exists():
            problems.append(f"missing file {target}")
            continue
        found = target.read_bytes().decode("utf-8").replace("\r\n", "\n").count(anchor)
        if found != expected:
            problems.append(f"{target.name}: anchor {anchor.strip()[:60]!r} found {found} times, expected {expected}")
    return problems


def _rebase(path: Path, root: Optional[Path]) -> Path:
    """``path`` under ``root`` instead of the repository (identity when ``root`` is None)."""
    return path if root is None else root / path.relative_to(REPO)


def apply(new: Operator, prev: Operator, params: "Optional[list[tuple[str, str]]]" = None, comment: str = "", root: Optional[Path] = None) -> int:
    """Write every edit (line endings of each file preserved); returns the number of edits. Raises ``RuntimeError`` listing the problems when the dry run fails (nothing is written)."""
    problems = plan(new, prev, params, comment, root)
    if problems:
        raise RuntimeError("cannot wire the operator:\n  " + "\n  ".join(problems))
    count = 0
    for path, anchor, repl, _expected in _edits(new, prev, params or [], comment or f"{new.family} operator"):
        target = _rebase(path, root)
        raw = target.read_bytes().decode("utf-8")
        nl = "\r\n" if "\r\n" in raw else "\n"
        target.write_bytes(raw.replace(anchor.replace("\n", nl), repl.replace("\n", nl)).encode("utf-8"))
        count += 1
    return count


def main(argv: "Optional[list[str]]" = None) -> None:
    """Command line: dry run by default, ``--apply`` to write."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kind", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--flag", required=True)
    ap.add_argument("--fn", required=True)
    ap.add_argument("--module", required=True)
    ap.add_argument("--prev", default="row_stat", choices=sorted(PREVIOUS))
    ap.add_argument("--comment", default="")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    new = Operator(args.kind, args.family, args.flag, args.fn, args.module)
    if args.apply:
        print(f"{apply(new, PREVIOUS[args.prev], comment=args.comment)} edits written")
    else:
        problems = plan(new, PREVIOUS[args.prev], comment=args.comment)
        print("dry run: all anchors found" if not problems else "\n".join(problems))


if __name__ == "__main__":
    main()
