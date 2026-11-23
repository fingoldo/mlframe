"""Cluster 21 regression: the save path does not compute multi_output, and resolve_output_path works without it (audit 2026-11-20, finding 21-F24).

multi_output is documented as unused for naming. It is now optional, so the save path need not compute or pass it, while the other
callers that still pass it keep working.
"""

import inspect

from mlframe.reporting.renderers import save as sv


def test_resolve_output_path_works_without_multi_output(tmp_path):
    """Calling resolve_output_path without multi_output gives the same name as with it (21-F24)."""
    base = str(tmp_path / "chart")
    without = sv.resolve_output_path(base, "matplotlib", "png")
    with_flag = sv.resolve_output_path(base, "matplotlib", "png", multi_output=False)
    assert without == with_flag


def test_save_path_no_longer_computes_multi_output():
    """The save function neither computes multi_output nor passes it on (21-F24)."""
    text = inspect.getsource(sv)
    assert "multi_output = (len(_backends) > 1)" not in text
    assert "multi_output=multi_output" not in text
