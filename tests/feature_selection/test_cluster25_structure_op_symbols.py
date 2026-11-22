"""Audit 2026-11-20 findings 25-F20 and 25-F22.

25-F22: a modular op reads with its symbol in both the live report and the recipe report (``a - b``, not ``a diff b``).
25-F20: every numba kernel in mi.py is compiled with cache=False, as the module comment requires; none is left cache=True.
"""

import re
from pathlib import Path

import mlframe.feature_selection.mi as mi_module
from mlframe.feature_selection.structure_discovery import _modular_op_text


def test_modular_ops_read_with_their_symbols():
    """diff is a minus, sum is a plus, prod a times, and self is the bare column."""
    assert _modular_op_text("diff", ("a", "b")) == "a - b"
    assert _modular_op_text("sum", ("a", "b", "c")) == "a + b + c"
    assert _modular_op_text("prod", ("a", "b")) == "a * b"
    assert _modular_op_text("self", ("a",)) == "a"


def test_no_numba_kernel_in_mi_is_cached():
    """The source of mi.py has no cache=True decorator: every kernel carries the CI-driven cache=False."""
    source = Path(mi_module.__file__).read_text(encoding="utf-8")
    assert re.search(r"cache=True", source) is None
