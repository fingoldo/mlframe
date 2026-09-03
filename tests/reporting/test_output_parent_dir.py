"""Every figure writer must create its output directory, not assume one exists.

``render_and_save`` always made its own directories, but thirteen other call sites wrote a figure straight to a
path -- ``savefig`` / ``write_html`` / ``write_image`` -- with nothing creating the folder. On a first run into
a fresh output tree those raise ``FileNotFoundError``, and most sit inside a best-effort ``except``, so the
chart never appears and nothing says why. A production run reported exactly that: no chart files, no
subdirectories.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from mlframe._output_paths import ensure_parent_dir

WRITERS = {"savefig", "write_html", "write_image"}
SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe"


class TestTheHelper:
    """Small enough to be obvious; the cases that matter are the degenerate ones."""

    def test_creates_a_missing_directory(self, tmp_path):
        """The whole point."""
        target = tmp_path / "charts" / "png" / "fig.png"
        assert not target.parent.exists()
        assert ensure_parent_dir(str(target)) == str(target)
        assert target.parent.is_dir()

    def test_is_idempotent(self, tmp_path):
        """Called once per write, so an existing directory must not raise."""
        target = str(tmp_path / "a" / "b.png")
        ensure_parent_dir(target)
        ensure_parent_dir(target)
        assert pathlib.Path(target).parent.is_dir()

    def test_returns_the_path_unchanged(self, tmp_path):
        """Callers wrap the destination in place, so the value has to pass straight through."""
        target = str(tmp_path / "x.png")
        assert ensure_parent_dir(target) is target

    @pytest.mark.parametrize("value", ["fig.png", "", None])
    def test_bare_name_and_empty_input_are_no_ops(self, value):
        """A filename with no directory component needs nothing created, and must not raise."""
        assert ensure_parent_dir(value) == value

    def test_an_uncreatable_directory_does_not_raise(self, tmp_path):
        """The write itself is about to fail with a far more specific error; do not mask it with a mkdir one."""
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("x", encoding="utf-8")
        assert ensure_parent_dir(str(blocker / "sub" / "fig.png"))


class TestEveryWriterIsGuarded:
    """The ratchet: a new figure writer must route through the helper or this fails."""

    def test_no_unguarded_figure_writer_remains(self):
        """Scans the package rather than trusting a list, so a newly added writer is caught."""
        offenders = []
        for path in sorted(SRC.rglob("*.py")):
            if "_benchmarks" in path.parts:
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            # Does this module create the directory itself? Read from the SYNTAX TREE, not by searching the
            # source text: a substring check would also match the word inside a comment or a docstring, and
            # asserting on source text as a stand-in for behaviour is what the meta-suite forbids.
            creates_dir = any(
                isinstance(n, ast.Call)
                and (
                    (isinstance(n.func, ast.Attribute) and n.func.attr in {"makedirs", "mkdir"})
                    or (isinstance(n.func, ast.Name) and n.func.id in {"makedirs", "ensure_parent_dir"})
                )
                for n in ast.walk(tree)
            )
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr not in WRITERS or not node.args:
                    continue
                first = node.args[0]
                guarded = isinstance(first, ast.Call) and isinstance(first.func, ast.Name) and first.func.id == "ensure_parent_dir"
                # A writer inside render_and_save's own dispatch is covered by that module's own directory
                # creation; everything else has to guard its own path.
                if not guarded and not creates_dir:
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno} ({node.func.attr})")
        assert not offenders, "figure writers with no parent-directory guard: " + ", ".join(offenders)
