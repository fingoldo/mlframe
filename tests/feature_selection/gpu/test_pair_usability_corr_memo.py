"""A large batched usability |corr| request is computed once per distinct input content, and a repeat gets an equal table without running a backend."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import batch_pair_usability_corr_gpu as mod


@pytest.fixture
def spy(monkeypatch):
    """Count the calls that reach the real backend dispatcher, clear the memo, and make every call count as large."""
    calls = []
    real = mod._dispatch_batch_pair_usability_corr_uncached

    def counting(*args, **kwargs):
        """Record the call and run the real dispatcher."""
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "_dispatch_batch_pair_usability_corr_uncached", counting)
    monkeypatch.setattr(mod, "_CORR_MEMO_MIN_WORK", 1)
    mod._CORR_MEMO.clear()
    yield calls
    mod._CORR_MEMO.clear()


def _inputs(seed=0, n=600, n_ops=12):
    """A small request: label, operand matrix, pair indices."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n).astype(np.float32)
    ops = rng.standard_normal((n_ops, n)).astype(np.float32)
    pa = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    pb = np.array([6, 7, 8, 9, 10, 11], dtype=np.int64)
    return y, ops, pa, pb


def test_repeat_with_the_same_content_does_not_reach_the_backend(spy):
    """The second request with equal input content returns an equal table and the same backend name without a backend call."""
    y, ops, pa, pb = _inputs()
    first, name1 = mod.dispatch_batch_pair_usability_corr(y, ops, pa, pb, force_backend="cpu")
    again, name2 = mod.dispatch_batch_pair_usability_corr(y.copy(), ops.copy(), pa.copy(), pb.copy(), force_backend="cpu")
    assert len(spy) == 1 and name1 == name2 == "cpu"
    np.testing.assert_array_equal(first, again)


@pytest.mark.parametrize("change", ["y", "operand", "pair", "backend"])
def test_any_changed_input_is_a_miss(spy, change):
    """Changing the label, one operand value, a pair index or the forced backend recomputes."""
    y, ops, pa, pb = _inputs()
    mod.dispatch_batch_pair_usability_corr(y, ops, pa, pb, force_backend="cpu")
    if change == "y":
        y = y.copy()
        y[3] += 1.0
    elif change == "operand":
        ops = ops.copy()
        ops[2, 5] += 1.0
    elif change == "pair":
        pb = pb.copy()
        pb[0] = 11
    mod.dispatch_batch_pair_usability_corr(y, ops, pa, pb, force_backend=None if change == "backend" else "cpu")
    assert len(spy) == 2


def test_changing_a_returned_table_does_not_change_the_stored_one(spy):
    """A caller that edits its table in place cannot corrupt what a later identical request receives."""
    y, ops, pa, pb = _inputs()
    first, _ = mod.dispatch_batch_pair_usability_corr(y, ops, pa, pb, force_backend="cpu")
    reference = first.copy()
    first[:] = -1.0
    again, _ = mod.dispatch_batch_pair_usability_corr(y, ops, pa, pb, force_backend="cpu")
    np.testing.assert_array_equal(again, reference)


def test_small_requests_are_not_memoized(spy, monkeypatch):
    """Below the work threshold every request runs the backend (no hashing, no stored table)."""
    monkeypatch.setattr(mod, "_CORR_MEMO_MIN_WORK", 10**12)
    y, ops, pa, pb = _inputs()
    mod.dispatch_batch_pair_usability_corr(y, ops, pa, pb, force_backend="cpu")
    mod.dispatch_batch_pair_usability_corr(y, ops, pa, pb, force_backend="cpu")
    assert len(spy) == 2 and not mod._CORR_MEMO


def test_memo_keeps_only_the_newest_results(spy):
    """More distinct requests than the memo holds evict the oldest, so the oldest is recomputed."""
    first_inputs = _inputs(seed=0)
    mod.dispatch_batch_pair_usability_corr(*first_inputs, force_backend="cpu")
    for seed in range(1, mod._CORR_MEMO_MAX + 1):
        mod.dispatch_batch_pair_usability_corr(*_inputs(seed=seed), force_backend="cpu")
    before = len(spy)
    mod.dispatch_batch_pair_usability_corr(*first_inputs, force_backend="cpu")
    assert len(spy) == before + 1
    assert len(mod._CORR_MEMO) <= mod._CORR_MEMO_MAX
