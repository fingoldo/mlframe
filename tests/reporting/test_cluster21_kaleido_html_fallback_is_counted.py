"""Cluster 21 regression: a static image replaced by an HTML file counts as a dropped chart (audit 2026-11-20, 21-F12)."""

from mlframe.reporting.renderers import _kaleido as kd
from mlframe.reporting.renderers import save


class _HtmlOnlyFig:
    """A figure whose HTML write works."""

    def write_html(self, path, **kwargs):
        """Write a stub HTML file."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("<html></html>")


def test_burned_kaleido_html_substitution_is_counted(tmp_path, monkeypatch):
    """With kaleido burned, a png request writes .html and the render-failure counter goes up by one (21-F12)."""
    monkeypatch.setattr(kd, "_is_kaleido_persistent_burned", lambda: True)
    save.reset_render_failure_stats()
    kd.write_image_via_kaleido(_HtmlOnlyFig(), str(tmp_path / "chart.png"), "png")
    assert (tmp_path / "chart.html").exists()
    assert save.get_render_failure_stats()["total"] == 1
    save.reset_render_failure_stats()
