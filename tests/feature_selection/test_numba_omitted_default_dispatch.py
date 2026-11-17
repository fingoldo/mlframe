"""Calls from Python into the entropy kernels spell out every argument, so numba's dispatcher resolves them natively.

numba keeps one type code for every omitted argument: once a kernel has been called with a default left out, a call that leaves out a default of another value never matches natively and is
resolved in the interpreter each time (60 us instead of 1 us; 1800 such calls in a 100000 x 100 fit).
"""

import numpy as np
from numba import njit
from numba.core.dispatcher import Dispatcher

from mlframe.feature_selection.filters.info_theory import entropy, merge_vars


@njit
def _kernel_with_default_one(a, k=1):
    """A kernel whose omitted default (1) differs from the defaults of the entropy kernels."""
    return a.sum() + k


def test_explicit_arguments_keep_the_native_dispatch(monkeypatch):
    """After a kernel with another default has run, the fully spelled calls of ``merge_vars`` and ``entropy`` never reach the interpreter-side resolver."""
    factors = np.random.default_rng(0).integers(0, 8, (500, 6)).astype(np.int8)
    nbins = np.full(6, 8, dtype=np.int64)
    idx = np.zeros(1, dtype=np.int64)
    _kernel_with_default_one(np.ones(3))
    merge_vars(factors, idx, None, nbins, np.int32, 1, None, False)  # compiled (or loaded) before the count starts
    entropy(np.ones(4) / 4, 0.0)

    slow = []
    real = Dispatcher._compile_for_args

    def spy(self, *args, **kwargs):
        """Record the kernels that fall back to the interpreter-side resolver."""
        slow.append(self.py_func.__name__)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Dispatcher, "_compile_for_args", spy)
    for col in range(6):
        idx[0] = col
        _, freqs, _ = merge_vars(factors, idx, None, nbins, np.int32, 1, None, False)
        assert entropy(freqs, 0.0) > 0.0
    assert slow == []
