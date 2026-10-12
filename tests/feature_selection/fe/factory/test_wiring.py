"""The operator wiring tool: a dry run finds every anchor, ``apply`` writes valid edits into a copy of the tree, and a missing anchor stops it before anything is written."""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

from mlframe.feature_selection._benchmarks.fe_operator_factory import wiring

NEW = wiring.Operator("demo_kind", "demo_fam", "fe_demo_fam_enable", "apply_demo_recipe", "_demo_fam_fe")
PARAMS = [("top_k", "int = 3,"), ("min_relative_gain", "float = 0.05,")]


def _copy_tree(root: Path) -> None:
    """Copy every file the tool edits into ``root`` keeping the repository-relative layout."""
    prev = wiring.PREVIOUS["row_stat"]
    for path, *_ in wiring._edits(NEW, prev, PARAMS, "demo"):
        dest = root / path.relative_to(wiring.REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)


def test_dry_run_finds_every_anchor_in_the_current_tree():
    """The anchors of the last wired stage operator all occur exactly as often as expected."""
    assert wiring.plan(NEW, wiring.PREVIOUS["row_stat"], PARAMS, "demo") == []


def _stat_key(path):
    """Size and mtime of a file: enough to show a write happened, without reading the source back."""
    st = path.stat()
    return (st.st_size, st.st_mtime_ns)


def test_apply_writes_valid_python_with_the_new_names_into_a_copy(tmp_path):
    """On a copy of the edited files every file still parses, every registry names the new operator, and the repository itself is untouched."""
    _copy_tree(tmp_path)
    before = _stat_key(wiring.FILTERS / "mrmr/_mrmr_class.py")
    n_edits = wiring.apply(NEW, wiring.PREVIOUS["row_stat"], PARAMS, "demo", root=tmp_path)
    assert n_edits >= 20
    for path in {p for p, *_ in wiring._edits(NEW, wiring.PREVIOUS["row_stat"], PARAMS, "demo")}:
        copy = tmp_path / path.relative_to(wiring.REPO)
        text = copy.read_text(encoding="utf-8")
        ast.parse(text)
        assert "demo" in text, copy.name
    assert _stat_key(wiring.FILTERS / "mrmr/_mrmr_class.py") == before
    planned = [edit for edit in wiring._edits(NEW, wiring.PREVIOUS["row_stat"], PARAMS, "demo") if edit[0].name == "_mrmr_class.py"]
    assert any("fe_demo_fam_top_k: int = 3," in str(part) for edit in planned for part in edit[1:]), "the plan must write the new MRMR parameter"


def test_a_missing_anchor_is_reported_and_nothing_is_written(tmp_path):
    """An operator that is not wired (a made-up previous one) yields problems, and ``apply`` refuses to write."""
    ghost = wiring.Operator("ghost_kind", "ghost_fam", "fe_ghost_fam_enable", "apply_ghost", "_ghost_fe")
    problems = wiring.plan(NEW, ghost, PARAMS, "demo")
    assert problems and all("expected" in p or "missing" in p for p in problems)
    _copy_tree(tmp_path)
    snapshot = {p: _stat_key(p) for p in tmp_path.rglob("*.py")}
    with pytest.raises(RuntimeError, match="cannot wire"):
        wiring.apply(NEW, ghost, PARAMS, "demo", root=tmp_path)
    assert all(_stat_key(p) == key for p, key in snapshot.items())
