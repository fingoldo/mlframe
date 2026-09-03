"""Some charts landed loose beside the html/ and png/ directories the rest of the run wrote into.

A production output folder held ``html/``, ``png/`` and then, at the top level, ``*_decile_table.png``,
``*_fiplot.png``, ``*_shap.png`` and ``*_report.html``. Those writers compose their own filename
(``base_path + ".png"``) instead of asking ``resolve_output_path`` where the file belongs, so the per-format
layout simply did not apply to them.

This is the second time this defect class has surfaced: the calibration plot returned a flat path while the file
had been written into ``png/``. That one was fixed where it crashed, without grepping for the rest -- so the
ratchet below scans for the pattern rather than trusting a list.
"""

from __future__ import annotations

import ast
import os
import pathlib

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlframe.reporting.renderers.save import resolve_output_path, set_format_subfolders

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe"


@pytest.fixture
def subfolders_on():
    """Turn the per-format layout on for one test and restore whatever was set before."""
    from mlframe.reporting.renderers.save import get_format_subfolders

    prior = get_format_subfolders()
    set_format_subfolders(True)
    yield
    set_format_subfolders(prior)


class TestTheWritersThatComposedTheirOwnNames:
    """Each of these produced a loose file in the production run."""

    def test_the_raw_figure_saver_writes_into_the_format_directory(self, tmp_path, subfolders_on):
        """``_save_figure`` is the path decile_table and the model-comparison charts take."""
        from mlframe.reporting.diagnostics_dispatch import _save_figure

        fig = plt.figure()
        plt.plot([1, 2, 3])
        _save_figure(fig, "matplotlib[png]", str(tmp_path / "decile_table"))
        assert (tmp_path / "png" / "decile_table.png").exists()
        assert not (tmp_path / "decile_table.png").exists()

    def test_the_shap_plot_path_is_resolved(self, tmp_path, subfolders_on):
        """SHAP writes through a ``plot_file`` the dispatcher hands it, so the dispatcher must resolve it."""
        from mlframe.reporting.diagnostics_dispatch import _png_path

        out = _png_path(str(tmp_path / "model_shap"))
        assert os.path.dirname(out).endswith("png")

    def test_the_feature_importance_plot_path_is_resolved(self, tmp_path, subfolders_on):
        """``_fiplot.png`` was concatenated onto the base path."""
        from mlframe.training.reporting._reporting import _fi_png_path

        assert os.path.dirname(_fi_png_path(str(tmp_path / "model"))).endswith("png")

    def test_the_flat_layout_is_unchanged(self, tmp_path):
        """With subfolders off, every one of these must write exactly where it always did."""
        from mlframe.reporting.renderers.save import get_format_subfolders

        prior = get_format_subfolders()
        set_format_subfolders(False)
        try:
            from mlframe.reporting.diagnostics_dispatch import _save_figure

            fig = plt.figure()
            plt.plot([1, 2, 3])
            _save_figure(fig, "matplotlib[png]", str(tmp_path / "decile_table"))
            assert (tmp_path / "decile_table.png").exists()
        finally:
            set_format_subfolders(prior)

    def test_resolve_output_path_is_what_decides(self, tmp_path, subfolders_on):
        """Pins the contract the writers now defer to, rather than each re-deriving the directory."""
        assert resolve_output_path(str(tmp_path / "x"), "matplotlib", "png", multi_output=False).endswith(os.path.join("png", "x.png"))


class TestNoWriterComposesAnExtensionAgain:
    """The ratchet: scan for the pattern instead of trusting the list above to stay complete."""

    # Files that legitimately build a name with an extension for a purpose other than writing a chart:
    # the renderers themselves (they ARE the layout), and the resolver's own module.
    ALLOWED = {
        pathlib.Path("reporting/renderers/save.py"),
        pathlib.Path("reporting/renderers/_kaleido.py"),
        pathlib.Path("reporting/renderers/matplotlib.py"),
        pathlib.Path("reporting/renderers/plotly.py"),
    }

    def test_no_new_flat_chart_path_is_composed(self):
        """``x + ".png"`` fed to a writer bypasses the per-format layout by construction."""
        offenders = []
        for path in sorted(SRC.rglob("*.py")):
            rel = path.relative_to(SRC)
            if "_benchmarks" in rel.parts or rel in self.ALLOWED:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)):
                    continue
                right = node.right
                if not (isinstance(right, ast.Constant) and isinstance(right.value, str)):
                    continue
                if not right.value.lower().endswith((".png", ".html", ".svg", ".pdf", ".jpg")):
                    continue
                offenders.append(f"{rel}:{node.lineno} (+ {right.value!r})")
        assert not offenders, "chart paths composed by hand bypass the per-format subfolder layout; route them through " "resolve_output_path: " + ", ".join(
            offenders
        )
