"""A failed plotly JSON export leaves the previous file at that path unchanged."""

import pathlib

import plotly.graph_objects as go
import pytest

import mlframe.reporting.renderers.plotly as pr


def test_failed_json_write_keeps_the_previous_file(tmp_path, monkeypatch):
    """Simulate a write failure on the export path; the old JSON must survive."""
    out = tmp_path / "fig.json"
    out.write_text("OLD JSON", encoding="utf-8")
    real_open = open

    class _Broken:
        """A handle that fails on write, as on a full disk."""

        def __enter__(self):
            """Hand back the failing handle."""
            return self

        def __exit__(self, *exc):
            """Do not swallow the error."""
            return False

        def write(self, _text):
            """Fail the way a full disk does."""
            raise OSError("No space left on device")

    def fake_open(path, mode="r", *args, **kwargs):
        """Route writes under the export path through the failing handle, truncating like the real open."""
        if str(path).startswith(str(out)) and "w" in mode:
            real_open(path, "w").close()
            return _Broken()
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(pr, "open", fake_open, raising=False)
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    with pytest.raises(OSError):
        pr.PlotlyRenderer().save(fig, str(out), "json")
    assert pathlib.Path(out).read_text(encoding="utf-8") == "OLD JSON"
