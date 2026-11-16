"""A failed write of the combined HTML report leaves the previous report file unchanged."""

import pathlib

import pytest

import mlframe.reporting.report_html as rh


def test_failed_write_keeps_the_previous_report(tmp_path, monkeypatch):
    """Simulate a disk-full write on the report path: the old file must survive."""
    out = tmp_path / "report.html"
    out.write_text("OLD REPORT", encoding="utf-8")
    real_open = open

    class _Broken:
        """A file that accepts nothing, as on a full disk."""

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
        """Route writes to the report path through the failing handle, truncating like the real open."""
        if str(path).startswith(str(out)) and "w" in mode:
            real_open(path, "w").close()
            return _Broken()
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(rh, "open", fake_open, raising=False)
    with pytest.raises(OSError):
        rh.build_combined_report([], title="t", out_path=str(out))
    assert pathlib.Path(out).read_text(encoding="utf-8") == "OLD REPORT"
