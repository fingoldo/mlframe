"""Audit 2026-11-20 findings 25-F20 and 25-F22.

25-F22: a modular op reads with its symbol in both the live report and the recipe report (``a - b``, not ``a diff b``).
25-F20: every numba kernel in mi.py is compiled with cache=False, as the module comment requires; none is left cache=True.
"""

from numba.core.registry import CPUDispatcher

import mlframe.feature_selection.mi as mi_module
from mlframe.feature_selection.structure_discovery import _modular_op_text


def test_modular_ops_read_with_their_symbols():
    """diff is a minus, sum is a plus, prod a times, and self is the bare column."""
    assert _modular_op_text("diff", ("a", "b")) == "a - b"
    assert _modular_op_text("sum", ("a", "b", "c")) == "a + b + c"
    assert _modular_op_text("prod", ("a", "b")) == "a * b"
    assert _modular_op_text("self", ("a",)) == "a"


def test_no_numba_kernel_in_mi_is_cached():
    """Every numba kernel in mi.py has caching off: a cached kernel can keep serving machine code built from a stale callee."""
    dispatchers = [v for v in vars(mi_module).values() if isinstance(v, CPUDispatcher)]
    assert dispatchers, "mi.py is expected to define numba kernels"
    cached = [d.py_func.__name__ for d in dispatchers if type(d._cache).__name__ != "NullCache"]
    assert cached == []
