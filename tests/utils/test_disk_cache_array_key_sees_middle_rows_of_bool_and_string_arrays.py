"""hash_array_summary distinguishes bool and fixed-width string arrays that differ only in their middle rows."""
import numpy as np

from mlframe.utils.disk_cache import hash_array_summary


def test_bool_arrays_differing_in_a_middle_row_get_different_keys():
    """Bool is not 'numeric' for np.issubdtype, so the key was only shape, dtype, head and tail."""
    a = np.zeros((200, 3), dtype=bool)
    b = a.copy()
    b[100, 0] = True
    assert hash_array_summary(a) != hash_array_summary(b)


def test_string_arrays_differing_in_a_middle_row_get_different_keys():
    """Fixed-width unicode arrays have no numeric reduction at all."""
    a = np.array(["x"] * 200)
    b = a.copy()
    b[100] = "y"
    assert hash_array_summary(a) != hash_array_summary(b)


def test_equal_arrays_still_share_a_key():
    """The key stays a pure function of the content."""
    a = np.array(["x", "y"] * 100)
    assert hash_array_summary(a) == hash_array_summary(a.copy())
