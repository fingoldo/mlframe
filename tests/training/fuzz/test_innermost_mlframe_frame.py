"""Failure rows name the deepest mlframe frame, not only the final exception line.

The fuzz driver's ``error_summary`` keeps only the last traceback line ("ValueError: Unexpected input dimension 11, expected 18"), which does not say where it was raised.
"""

from __future__ import annotations

from tests.training.fuzz.test_fuzz_suite import _innermost_mlframe_frame


def _raise_here() -> None:
    """Raise from a fixed line so the test can pin the reported frame."""
    raise ValueError("boom")


def test_innermost_mlframe_frame_names_the_raising_line():
    """The helper reports the file and line of the frame that raised, relative to the mlframe root."""
    try:
        _raise_here()
    except ValueError as exc:
        origin = _innermost_mlframe_frame(exc)

    assert origin.startswith("tests/training/fuzz/test_innermost_mlframe_frame.py:")
    assert origin.endswith(" in _raise_here")


def test_innermost_mlframe_frame_empty_without_mlframe_frames():
    """An exception raised outside mlframe frames yields an empty origin rather than an error."""
    exc = RuntimeError("outside")
    assert _innermost_mlframe_frame(exc) == ""
