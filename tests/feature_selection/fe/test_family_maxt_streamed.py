"""The family-wise maxT gate scored one group of columns at a time takes the decisions of the single pass."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _family_maxt_gate as gate
from mlframe.feature_selection.filters import _family_maxt_stream as stream
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _streamed_gpu_mi as sg


def _family(seed: int = 0, n: int = 2500):
    """A target of 3 classes and batches of codes: two carry real signal, three are noise; widths differ, so groups cut batches in the middle."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n).astype(np.int64)
    signal = (y[:, None] + (rng.random((n, 5)) < 0.55)).astype(np.int8) % 8
    weak = ((y[:, None] == 1) & (rng.random((n, 4)) < 0.25)).astype(np.int8) + rng.integers(0, 3, (n, 4)).astype(np.int8)
    noise = [rng.integers(0, 6, (n, w)).astype(np.int8) for w in (7, 3, 11)]
    return y, [signal, noise[0], weak, noise[1], noise[2]]


def _budget(monkeypatch, columns_per_group: int, n: int):
    """Make the free host memory allow about ``columns_per_group`` one-byte columns per group."""
    monkeypatch.setattr(sg, "_host_free_bytes", lambda: int(columns_per_group * n / stream._STACK_FREE_FRACTION) + 1)


def test_the_plan_is_none_when_the_family_fits_and_cuts_batches_when_it_does_not(monkeypatch):
    """A roomy memory keeps the single pass; a tight one gives groups that together cover every column exactly once."""
    y, batches = _family()
    widths = [b.shape[1] for b in batches]
    monkeypatch.setattr(sg, "_host_free_bytes", lambda: 1 << 40)
    assert stream.plan_stream_groups(batches, widths, len(y)) is None
    _budget(monkeypatch, 6, len(y))
    groups = stream.plan_stream_groups(batches, widths, len(y))
    assert groups is not None and len(groups) > 1
    covered = sorted((i, c) for g in groups for i, first, stop in g for c in range(first, stop))
    assert covered == sorted((i, c) for i, w in enumerate(widths) for c in range(w))
    assert all(sum(stop - first for _i, first, stop in g) <= 6 for g in groups)


def test_streamed_null_maxima_equal_the_single_pass_and_leave_the_generator_in_the_same_state(monkeypatch):
    """The elementwise maximum over the groups is the family's null maximum under the same permutations, for the probe size and the early block."""
    y, batches = _family(seed=3)
    widths = [b.shape[1] for b in batches]
    _budget(monkeypatch, 9, len(y))
    groups = stream.plan_stream_groups(batches, widths, len(y))
    assert groups is not None and len(groups) > 1
    cols = gate._stack_codes_transposed(batches, widths)
    n_x = cols.max(axis=1).astype(np.int64) + 1
    n_y = int(y.max()) + 1
    null_max = stream.streamed_null_max(batches, groups, y, n_y)
    rng_stream, rng_single = np.random.default_rng(11), np.random.default_rng(11)
    for m in (gate.FAMILY_PROBE_BLOCK, 100, gate.FAMILY_EARLY_BLOCK - gate.FAMILY_PROBE_BLOCK):
        np.testing.assert_array_equal(null_max(m, rng_stream), gate.family_null_max(cols, n_x, y, n_y, m, rng_single))
    assert rng_stream.bit_generator.state == rng_single.bit_generator.state


@pytest.mark.parametrize("analytic", ["1", "0"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_streamed_flags_equal_single_pass_flags(seed, analytic, monkeypatch):
    """Same pass flag for every batch, with the analytic triage on (certified batches) and off (the permutation test decides everything)."""
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", analytic)
    monkeypatch.setattr(gate, "_family_device_flags", lambda *a, **k: None, raising=False)
    y, batches = _family(seed=seed)
    single = gate.family_pass_batches(batches, y, alpha=0.05, n_perm=199, seed=5)
    calls = []
    real = stream.family_pass_streamed
    monkeypatch.setattr(stream, "family_pass_streamed", lambda *a, **k: calls.append(1) or real(*a, **k))
    _budget(monkeypatch, 8, len(y))
    streamed = gate.family_pass_batches(batches, y, alpha=0.05, n_perm=199, seed=5)
    assert calls, "the streamed path did not run"
    assert streamed == single
    assert any(single) and not all(single), f"the fixture must mix passing and failing batches: {single}"
