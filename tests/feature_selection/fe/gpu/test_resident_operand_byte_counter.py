"""The running byte total of the resident-operand table equals the true sum of its entries after inserts, capacity trims and a clear."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_resident_operands as RO


def _true_bytes() -> int:
    """Sum of the nbytes of every table entry."""
    return sum(int(v.nbytes) for v in RO._FE_RESIDENT_OPERANDS.values())


def test_running_byte_total_matches_table():
    """Insert, trim through raised_capacity and clear: the counter follows the table exactly."""
    RO.clear_fe_resident_operands()
    rng = np.random.default_rng(0)
    for i in range(6):
        RO.resident_operand(rng.standard_normal(1000 + i), ("t", i), dtype=np.float64)
    assert RO._RESIDENT_BYTES[0] == _true_bytes() > 0
    with RO.raised_capacity(2):
        pass
    assert RO._RESIDENT_BYTES[0] == _true_bytes()
    RO.clear_fe_resident_operands()
    assert RO._RESIDENT_BYTES[0] == 0 and not RO._FE_RESIDENT_OPERANDS
