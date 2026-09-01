"""An empty ``base_path`` means "do not persist this figure" -- but only the backend SELECTION honoured it.

Found as two stray files in the repository working tree after a test run: ``.html`` and ``.matplotlib.png``, both
written into the process's working directory. ``resolve_output_path`` had composed each name out of the extension
alone, because ``base_path`` was empty, and the result is dot-prefixed, so a plain ``ls`` never shows it.

``render_and_save`` computes ``will_save = bool(base_path)`` and drops save-only backends accordingly, so the
contract was already stated in the code. The save loop underneath it just never consulted it: a backend kept alive
by ``keep_handles`` or by an interactive session rendered and then wrote, empty path and all.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")

from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers.save import render_and_save
from mlframe.reporting.spec import FigureSpec, ScatterPanelSpec

# Both backends and both format families: the defect produced one artefact from each.
OUTPUT = parse_plot_output_dsl("matplotlib[png]+plotly[html]")


def _spec():
    """One trivial scatter, enough to make both backends produce a figure."""
    return FigureSpec(
        suptitle="",
        panels=((ScatterPanelSpec(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]), title="t"),),),
    )


def _listing(path):
    """Every entry in ``path``, dotfiles included -- the defect's whole point is that they hide."""
    return set(os.listdir(path))


class TestAnEmptyBasePathPersistsNothing:
    """Whatever else the call does, it must not leave a file behind."""

    def test_keeping_handles_writes_no_file(self, tmp_path, monkeypatch):
        """``keep_handles`` keeps every backend alive past the save-only filter; that is the live path."""
        monkeypatch.chdir(tmp_path)
        before = _listing(tmp_path)
        render_and_save(_spec(), OUTPUT, "", keep_handles=True)
        assert _listing(tmp_path) == before

    def test_an_interactive_session_writes_no_file(self, tmp_path, monkeypatch):
        """The other way a backend survives with nothing to save."""
        monkeypatch.chdir(tmp_path)
        before = _listing(tmp_path)
        render_and_save(_spec(), OUTPUT, "", interactive=True)
        assert _listing(tmp_path) == before

    def test_no_extension_only_dotfile_is_created(self, tmp_path, monkeypatch):
        """Names the exact artefacts that were found in the working tree."""
        monkeypatch.chdir(tmp_path)
        render_and_save(_spec(), OUTPUT, "", keep_handles=True, interactive=True)
        assert not (tmp_path / ".html").exists()
        assert not (tmp_path / ".matplotlib.png").exists()

    def test_the_handles_are_still_returned(self, tmp_path, monkeypatch):
        """Suppressing the write must not suppress the figure the caller asked for."""
        monkeypatch.chdir(tmp_path)
        handles = render_and_save(_spec(), OUTPUT, "", keep_handles=True)
        assert handles


class TestANonEmptyBasePathStillWrites:
    """The guard must not turn into a blanket suppression."""

    def test_a_real_base_path_produces_a_file(self, tmp_path, monkeypatch):
        """The ordinary path is unchanged."""
        monkeypatch.chdir(tmp_path)
        render_and_save(_spec(), OUTPUT, str(tmp_path / "chart"))
        written = [p for p in tmp_path.rglob("*") if p.is_file()]
        assert written, "a named base path must still persist the figure"
