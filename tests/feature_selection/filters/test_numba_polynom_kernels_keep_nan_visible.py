"""No kernel of the polynomial optimizer is compiled with the full fastmath set, which assumes no NaN or inf exists."""

from __future__ import annotations

from numba.core.registry import CPUDispatcher

from mlframe.feature_selection.filters import _numba_polynom_optimizer as mod


def test_no_dispatcher_uses_full_fastmath():
    """The full ``fastmath=True`` set carries nnan / ninf; every kernel must compile with an explicit NaN-safe flag set (or none)."""
    dispatchers = {name: obj for name, obj in vars(mod).items() if isinstance(obj, CPUDispatcher) and obj.py_func.__module__ == mod.__name__}
    assert len(dispatchers) >= 10
    offenders = [name for name, d in dispatchers.items() if d.targetoptions.get("fastmath") is True]
    assert not offenders, f"compiled with full fastmath: {offenders}"
    for name, d in dispatchers.items():
        flags = d.targetoptions.get("fastmath")
        if isinstance(flags, (set, frozenset)):
            assert not {"nnan", "ninf"} & set(flags), name
