"""Regression sensor for INV-57: the kaleido lifecycle + static-image write plumbing must live in
the ``_kaleido`` sibling, and ``plotly.py`` must stay under the 1000-LOC house limit while keeping
the public surface importable from the same place.

Pre-carve ``plotly.py`` was 1175 LOC (over the limit) and ``_kaleido.py`` did not exist, so
``from mlframe.reporting.renderers._kaleido import write_image_via_kaleido`` raised ImportError --
this sensor fails on that pre-fix state. It also EXERCISES the moved ``write_image_via_kaleido``
body (not import-only) via a fake figure so a sibling that compiled but referenced an unresolved
parent name would blow up here.
"""

from __future__ import annotations

import os


def test_inv57_public_kaleido_surface_reexported_from_plotly():
    """Inv57 public kaleido surface reexported from plotly."""
    from mlframe.reporting.renderers import _kaleido, plotly

    public = [
        "get_kaleido_oneshot_stats",
        "reset_kaleido_oneshot_stats",
        "record_kaleido_oneshot_call",
        "write_image_via_kaleido",
        "_restart_kaleido_server",
        "_ensure_kaleido_server_started",
        "_is_kaleido_persistent_burned",
        "_record_kaleido_persistent_failure",
        "_mark_kaleido_persistent_burned",
    ]
    for name in public:
        assert hasattr(_kaleido, name), f"{name} missing from _kaleido sibling"
        assert hasattr(plotly, name), f"{name} not re-exported from plotly after the carve"
    assert hasattr(plotly, "PlotlyRenderer")


def test_inv57_oneshot_stats_roundtrip_through_reexport():
    """Inv57 oneshot stats roundtrip through reexport."""
    from mlframe.reporting.renderers import _kaleido
    from mlframe.reporting.renderers.plotly import get_kaleido_oneshot_stats

    _kaleido.reset_kaleido_oneshot_stats()
    _kaleido.record_kaleido_oneshot_call(2.0)
    _kaleido.record_kaleido_oneshot_call(1.0)
    n, wall = get_kaleido_oneshot_stats()
    assert n == 2 and abs(wall - 3.0) < 1e-9
    _kaleido.reset_kaleido_oneshot_stats()
    assert get_kaleido_oneshot_stats() == (0, 0.0)


class _FakeFig:
    """Groups tests for: FakeFig."""
    def __init__(self):
        """Helper: Init  ."""
        self.image_calls = 0
        self.html_calls = 0

    def write_image(self, path, format=None):  # noqa: A002 -- must match plotly's real write_image(path, format=...) kwarg name; production calls it as format=fmt
        """Write image."""
        self.image_calls += 1
        raise RuntimeError("no kaleido in this test")

    def write_html(self, path, include_plotlyjs=None, auto_open=None):
        """Write html."""
        self.html_calls += 1
        with open(path, "w", encoding="utf-8") as f:
            f.write("<html>fallback</html>")


def test_inv57_write_image_via_kaleido_burned_path_writes_html(tmp_path):
    """When the persistent path is burned, the moved body writes interactive HTML directly and
    returns BEFORE the oneshot block (oneshot would re-enter the same dead queue)."""
    from mlframe.reporting.renderers import _kaleido

    _kaleido.reset_kaleido_oneshot_stats()
    _kaleido._mark_kaleido_persistent_burned()
    try:
        fig = _FakeFig()
        target = str(tmp_path / "burned.png")
        _kaleido.write_image_via_kaleido(fig, target, "png")
        assert os.path.exists(os.path.splitext(target)[0] + ".html"), "INV-57: burned path did not write the HTML fallback"
        assert fig.image_calls == 0, "INV-57: burned path must not re-enter write_image oneshot"
        assert _kaleido.get_kaleido_oneshot_stats()[0] == 0
    finally:
        # Clear the process-global burn flag directly so the burned state cannot leak to other
        # tests on a host without the real kaleido package (where _restart can't reset it).
        _kaleido._KALEIDO_PERSISTENT_BURNED = False
        _kaleido._KALEIDO_PERSISTENT_FAIL_COUNT = 0
        _kaleido.reset_kaleido_oneshot_stats()


def test_inv57_write_image_via_kaleido_oneshot_html_fallback(tmp_path, monkeypatch):
    """Exercise the moved oneshot->HTML fallback ladder: with the persistent server unavailable,
    the body calls ``fig.write_image`` (raises here), falls back to HTML, and records the oneshot
    call. Proves the moved code path runs end-to-end (catches a sibling-split unresolved-name
    regression that import-only sensors would miss)."""
    from mlframe.reporting.renderers import _kaleido

    _kaleido.reset_kaleido_oneshot_stats()
    # Persistent server unavailable -> deterministic oneshot branch on any host.
    monkeypatch.setattr(_kaleido, "_ensure_kaleido_server_started", lambda: False)
    monkeypatch.setattr(_kaleido, "_is_kaleido_persistent_burned", lambda: False)

    fig = _FakeFig()
    target = str(tmp_path / "chart.png")
    _kaleido.write_image_via_kaleido(fig, target, "png")

    assert fig.image_calls == 1, "INV-57: oneshot path did not call write_image"
    assert os.path.exists(os.path.splitext(target)[0] + ".html"), "INV-57: oneshot path did not write the HTML fallback after write_image raised"
    n, _ = _kaleido.get_kaleido_oneshot_stats()
    assert n == 1, f"INV-57: oneshot call not recorded by the moved write path (n={n})"
    _kaleido.reset_kaleido_oneshot_stats()


def test_inv57_plotly_module_under_house_loc_limit():
    """Inv57 plotly module under house loc limit."""
    here = os.path.dirname(__import__("mlframe.reporting.renderers.plotly", fromlist=["__file__"]).__file__)
    plotly_path = os.path.join(here, "plotly.py")
    with open(plotly_path, encoding="utf-8") as f:
        loc = sum(1 for _ in f)
    assert loc < 1000, f"INV-57: plotly.py is {loc} LOC, over the 1000-LOC house limit"
