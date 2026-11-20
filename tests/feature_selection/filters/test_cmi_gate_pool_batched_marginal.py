"""The CMI-gate pool scores its candidates with one batched marginal-MI pass and gets the per-column values (structural launch reduction, 2026-11-20)."""

from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._mrmr_fe_step import _step_score as ss


def _pool(n=6000, k=12, seed=0):
    """A pool of ``k`` continuous candidate columns, some related to the 3-class target, and the target."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n).astype(np.int64)
    cols = np.stack([rng.normal(size=n) + 0.4 * j * (y == 1) for j in range(k)], axis=1).astype(np.float32)
    return cols, y


def _score(monkeypatch, batched):
    """``st._cmi_cands`` of one pool, with the batched marginal MI on or off."""
    cols, y = _pool()
    names = [f"c{j}" for j in range(cols.shape[1])]
    self = SimpleNamespace(quantization_nbins=10)
    st = SimpleNamespace(_gate_resident=True, _cmi_cands={})
    if not batched:
        monkeypatch.setattr(ss, "_batched_marginal_mi", lambda *a, **k: None)
    ss._score_candidates_one_by_one(self, st, {"k": (True, cols, names, 0, [])}, 1, y)
    return st._cmi_cands


def test_batched_marginal_mi_equals_the_per_column_values(monkeypatch):
    """Every candidate gets the marginal MI of the per-column path to 1e-12, and the host values are the pool columns."""
    batched = _score(monkeypatch, True)
    monkeypatch.undo()
    reference = _score(monkeypatch, False)
    assert batched.keys() == reference.keys() and len(batched) == 12
    for name in batched:
        assert batched[name][1] == pytest.approx(reference[name][1], abs=1e-12)
        np.testing.assert_array_equal(batched[name][0], reference[name][0])


def test_batched_path_makes_one_marginal_call(monkeypatch):
    """The batched route calls ``batched_cmi_gpu`` once for the pool, not once per candidate."""
    import mlframe.feature_selection.filters._fe_batched_mi_blocks as blocks

    calls = []
    real = blocks.batched_cmi_gpu

    def counting(*a, **k):
        """Count the batched marginal calls."""
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(blocks, "batched_cmi_gpu", counting)
    _score(monkeypatch, True)
    assert len(calls) == 1
