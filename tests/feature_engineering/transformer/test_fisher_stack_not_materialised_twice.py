"""The Fisher finite-difference stack must be materialised once, not twice.

`np.broadcast_to(X, (d, n, d))` is a zero-stride view, and reshaping a non-contiguous broadcast view cannot
itself be a view -- numpy has already allocated a full, writeable, independent `(d*n, d)` array by the time
`.reshape` returns. The `.copy()` that followed allocated a second one, so the transient peak was twice
what `_MAX_STACK_ELEMS` budgets: measured at 488.3 MiB against a 244.1 MiB intended stack, which at the cap
means 1.02 GB rather than 512 MB, before `predict_proba`'s own `(d*n, n_classes)` output on top.

The cap is reached on ordinary frames once `d` grows: at d=64 it binds at 15 625 rows.
"""

from __future__ import annotations

import tracemalloc

import numpy as np


def _peak_mib(fn) -> float:
    """Peak Python-allocated memory during `fn`, in MiB."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024**2


def _build_stack(X: np.ndarray, d: int, n: int) -> np.ndarray:
    """The production expression, as it now stands."""
    return np.broadcast_to(X, (d, n, d)).reshape(d * n, d)


def test_the_reshape_already_owns_its_data():
    """The premise the fix rests on: `.copy()` was duplicating something already independent."""
    n, d = 40, 6
    X = np.arange(float(n * d)).reshape(n, d)
    stack = _build_stack(X, d, n)
    assert not np.shares_memory(stack, X), "the stack aliases X, so the copy was load-bearing after all"
    assert stack.flags.writeable, "the per-block `+= eps` below needs a writeable array"


def test_the_stack_is_not_materialised_twice():
    """Peak allocation must be about one stack, not two."""
    n, d = 4000, 30
    X = np.zeros((n, d), dtype=np.float64)
    intended = n * d * d * 8 / 1024**2

    once = _peak_mib(lambda: _build_stack(X, d, n))
    twice = _peak_mib(lambda: _build_stack(X, d, n).copy())

    assert once < intended * 1.3, f"peak {once:.1f} MiB against an intended {intended:.1f} MiB stack"
    assert twice > intended * 1.7, f"the fixture no longer reproduces the doubling it was built to show ({twice:.1f} MiB)"


def test_the_contents_are_what_the_copy_produced():
    """Dropping the copy must not change a single value."""
    n, d = 25, 4
    X = np.arange(float(n * d)).reshape(n, d)
    stack = _build_stack(X, d, n)
    assert np.array_equal(stack, np.broadcast_to(X, (d, n, d)).reshape(d * n, d).copy())


def test_writing_into_one_block_leaves_x_untouched():
    """The loop adds eps per block; with an aliasing stack that would corrupt the caller's frame."""
    n, d = 25, 4
    X = np.arange(float(n * d)).reshape(n, d)
    original = X.copy()
    stack = _build_stack(X, d, n)
    for j in range(d):
        stack[j * n : (j + 1) * n, j] += 1.0
    assert np.array_equal(X, original), "the finite-difference perturbation leaked back into the input frame"


def test_each_block_carries_its_own_perturbation():
    """Guards the blocked layout itself: block j must differ from X in column j and nowhere else."""
    n, d = 12, 3
    X = np.arange(float(n * d)).reshape(n, d)
    stack = _build_stack(X, d, n)
    eps = 0.5
    for j in range(d):
        stack[j * n : (j + 1) * n, j] += eps
    for j in range(d):
        block = stack[j * n : (j + 1) * n]
        delta = block - X
        assert np.allclose(delta[:, j], eps)
        assert np.allclose(np.delete(delta, j, axis=1), 0.0)
