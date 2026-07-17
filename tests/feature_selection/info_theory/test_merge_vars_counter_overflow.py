"""Wave 9.1 loop-iter-21 regression: ``merge_vars`` per-bin counter
MUST NOT wrap on the caller's ``dtype``.

Pre-fix at ``info_theory.py:55, 65``: ``freqs`` and ``lookup_table``
were allocated with ``dtype=dtype`` (default int32, caller-overridable).
The bin counter ``freqs[newclass] += 1`` then SILENTLY wrapped when
any bin's sample count exceeded the dtype's positive range. Live
demonstration with ``dtype=int8`` and a 200-sample single-bin input::

    freqs_norm = [-0.28]   (expected [1.0])
    freqs_norm * n = [-56] (expected [200])

Same defect lurked at ``dtype=int32`` for any bin holding > 2^31
samples - a real risk at production scale (n >= 2 billion is rare but
not impossible in streaming joint-bin scenarios, and dominant-class
bins on multi-var joints concentrate samples quickly).

Effect: corrupted per-class freqs propagate through ``entropy()`` and
``conditional_mi()`` -> wrong MI scores throughout MRMR ranking. The
downstream ``freqs[freqs > 0]`` filter at info_theory.py:80 also drops
the wrapped-to-negative bins outright, silently zeroing entropy on
the affected slice.

Severity: P0 silent. Default ``dtype=int32`` is latent at typical n.
Any caller passing ``dtype=int8`` or ``int16`` (e.g. memory-constrained
batched joint computations) trips it at trivially small n.

Fix at info_theory.py:55, 65: allocate ``freqs`` and ``lookup_table``
with ``dtype=np.int64`` unconditionally. The bin counter is bounded by
``n_samples`` regardless of class-encoding dtype, so int64 is the safe
universal choice with negligible memory overhead.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_merge_vars_int8_dtype_no_wraparound():
    """200 samples in a single bin under dtype=int8 must give freq=1.0,
    not -0.28 (which is 200 mod 128 negated and normalised).
    """
    from mlframe.feature_selection.filters.info_theory import merge_vars

    n = 200
    factors = np.zeros((n, 1), dtype=np.int8)
    nbins = np.array([1], dtype=np.int64)
    classes, freqs_norm, nclasses = merge_vars(
        factors_data=factors,
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int8,
    )
    assert freqs_norm[0] > 0.99
    assert freqs_norm[0] < 1.01


@pytest.mark.parametrize("counter_dtype", [np.int8, np.int16, np.int32])
def test_merge_vars_counter_dtype_independence(counter_dtype):
    """The output ``freqs / n`` MUST equal the true relative frequency
    regardless of which dtype the caller supplied for the workspace.
    """
    from mlframe.feature_selection.filters.info_theory import merge_vars

    rng = np.random.default_rng(0)
    n = 500
    # 4 bins, 500 samples -> ~125 per bin; no overflow at any dtype.
    factors = rng.integers(0, 4, (n, 1)).astype(counter_dtype)
    nbins = np.array([4], dtype=np.int64)
    _, freqs_norm, nclasses = merge_vars(
        factors_data=factors,
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=counter_dtype,
    )
    # All freqs non-negative and sum to 1.
    assert (freqs_norm >= 0).all()
    assert abs(float(freqs_norm.sum()) - 1.0) < 1e-9


def test_merge_vars_joint_two_vars_correct():
    """Negative control: joint of two 4-bin variables on 10k samples
    must produce a valid 16-class joint summing to 1.
    """
    from mlframe.feature_selection.filters.info_theory import merge_vars

    rng = np.random.default_rng(0)
    n = 10_000
    factors = rng.integers(0, 4, (n, 2)).astype(np.int32)
    nbins = np.array([4, 4], dtype=np.int64)
    classes, freqs_norm, nclasses = merge_vars(
        factors_data=factors,
        vars_indices=np.array([0, 1], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
    )
    assert int(classes.min()) >= 0
    assert int(classes.max()) <= 15
    assert (freqs_norm >= 0).all()
    assert abs(float(freqs_norm.sum()) - 1.0) < 1e-9
