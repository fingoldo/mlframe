"""``_code_itemsize`` picks the narrowest code width; int8 matrices are bounded by their type and need no scan of their values."""

import numpy as np

from mlframe.feature_selection.filters._family_maxt_stream import _code_itemsize


def test_int8_matrices_give_one_byte_without_depending_on_values():
    """Any number of int8 matrices (even all at 127) need one byte per code."""
    mats = [np.full((50, 4), 127, dtype=np.int8), np.zeros((50, 3), dtype=np.int8)]
    assert _code_itemsize(mats) == 1


def test_wider_dtypes_are_still_read_by_value():
    """uint8 above 127, int16 above 127 and int64 above 32767 widen the code exactly as the value scan did."""
    assert _code_itemsize([np.array([[200]], dtype=np.uint8)]) == 2
    assert _code_itemsize([np.zeros((3, 3), dtype=np.int8), np.array([[300]], dtype=np.int16)]) == 2
    assert _code_itemsize([np.array([[70000]], dtype=np.int64)]) == 8
    assert _code_itemsize([np.array([[100]], dtype=np.int64)]) == 1


def test_empty_input_and_empty_matrices_give_one_byte():
    """No matrices, or matrices without elements, keep the one-byte default."""
    assert _code_itemsize([]) == 1
    assert _code_itemsize([np.zeros((0, 4), dtype=np.int64)]) == 1
