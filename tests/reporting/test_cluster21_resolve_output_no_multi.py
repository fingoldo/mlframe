"""Cluster 21 regression: the save path does not compute multi_output, and resolve_output_path works without it (audit 2026-11-20, finding 21-F24).

multi_output is documented as unused for naming. It is now optional, so the save path need not compute or pass it, while the other
callers that still pass it keep working.
"""

import inspect

from mlframe.reporting.renderers import save as sv


def test_resolve_output_path_has_no_multi_output_argument():
    """The unused multi_output keyword is removed from resolve_output_path (21-F24)."""
    import inspect as _inspect

    assert "multi_output" not in _inspect.signature(sv.resolve_output_path).parameters
    assert sv.resolve_output_path("base", "matplotlib", "png") is not None


def test_save_path_no_longer_computes_multi_output():
    """The save function neither computes multi_output nor passes it on (21-F24)."""
    text = inspect.getsource(sv)
    assert "multi_output = (len(_backends) > 1)" not in text
    assert "multi_output=multi_output" not in text
