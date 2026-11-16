"""The kaleido one-shot fallback writes an image into a directory that did not exist yet."""

import sys
import types


import mlframe.reporting.renderers._kaleido as kd


def test_fallback_writes_into_missing_directory(tmp_path, monkeypatch):
    """With the primary writer failing, the fallback creates the directory and writes the bytes."""
    target = tmp_path / "missing" / "chart.png"
    fake = types.ModuleType("kaleido")
    fake.calc_fig_sync = lambda fig, opts: b"PNGDATA"
    monkeypatch.setitem(sys.modules, "kaleido", fake)

    class _Fig:
        """Stand-in figure whose primary writer always fails."""

        def write_image(self, *a, **k):
            """Fail so the fallback path runs."""
            raise RuntimeError("primary writer unavailable")

    kd._oneshot_write_static(_Fig(), str(target), "png")
    assert target.read_bytes() == b"PNGDATA"


def test_failed_fallback_write_keeps_previous_file(tmp_path, monkeypatch):
    """A fallback write that fails mid-way leaves the previous image intact and no temp file behind."""
    target = tmp_path / "chart.png"
    target.write_bytes(b"OLD")
    fake = types.ModuleType("kaleido")
    fake.calc_fig_sync = lambda fig, opts: object()
    monkeypatch.setitem(sys.modules, "kaleido", fake)

    class _Fig:
        """Stand-in figure whose primary writer always fails."""

        def write_image(self, *a, **k):
            """Fail so the fallback path runs."""
            raise RuntimeError("primary writer unavailable")

    try:
        kd._oneshot_write_static(_Fig(), str(target), "png")
    except TypeError:
        pass
    assert target.read_bytes() == b"OLD"
    assert [p.name for p in tmp_path.iterdir()] == ["chart.png"]
