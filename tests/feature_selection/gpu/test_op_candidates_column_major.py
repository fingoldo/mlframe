"""The best-existing-op candidate block built column-major is the transpose of the row-major one, and its max MI is unchanged."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _resident_candidate_mi as rc


def _cols(m, n, seed=0):
    """m resident operand columns, with a zero and a negative stretch so the safe-divide and abs branches run."""
    rng = np.random.default_rng(seed)
    arrs = [rng.normal(size=n) for _ in range(m)]
    arrs[0][::13] = 0.0
    return [cp.asarray(a) for a in arrs]


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_cm_block_is_the_transpose(m):
    """Same columns, same order, same bits."""
    cols = _cols(m, 20_001, seed=m)
    rm = rc._build_best_existing_op_candidates_gpu(cols, cp)
    cm = rc._build_best_existing_op_candidates_gpu_cm(cols, cp)
    assert cm.shape == (rm.shape[1], rm.shape[0]) and cm.flags.c_contiguous
    np.testing.assert_array_equal(cp.asnumpy(cm), cp.asnumpy(rm).T)


@pytest.mark.parametrize("m", [2, 3, 5])
def test_max_mi_is_unchanged(m, monkeypatch):
    """best_existing_op_mi_resident returns the same number through the column-major block."""
    rng = np.random.default_rng(10 + m)
    n = 30_000
    arrs = {f"c{i}": rng.normal(size=n) for i in range(m)}
    y = ((arrs["c0"] * arrs["c1"] + 0.3 * rng.normal(size=n)) > 0).astype(np.int64)
    names = list(arrs)
    got = rc.best_existing_op_mi_resident(arrs, names, y, 10)
    assert got is not None
    cols = [cp.asarray(arrs[c]) for c in names]
    rm = rc._build_best_existing_op_candidates_gpu(cols, cp)
    from mlframe.feature_selection.filters._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

    want = float(np.max(_plugin_mi_classif_batch_cuda_resident(rm, cp.asarray(y), 10, y_min=0, n_classes=int(y.max()) + 1, relax_binning=True)))
    assert got == want


def _grid_specs(n, seed):
    """One mask combo and one select combo over three operand columns."""
    rng = np.random.default_rng(seed)
    a, b, c = (rng.normal(size=n) for _ in range(3))
    taus = np.linspace(-1.0, 1.0, 9)
    return [("mask", ("a", "c"), (c, a), taus), ("select", ("a", "b", "c"), (c, a, b), taus)]


def test_gate_grid_mi_is_identical_with_and_without_rank_binning_layouts():
    """The tau-grid MI vector through the column-major block equals the row-major block's, column for column."""
    n = 20_000
    specs = _grid_specs(n, 1)
    y = (np.random.default_rng(2).normal(size=n) > 0).astype(np.int64)
    got = rc.gate_grid_mi_resident(specs, y, 10)
    assert got is not None and got.shape == (18,)
    # the same blocks, built row-major by hand and scored through the unchanged row-major entry
    from mlframe.feature_selection.filters._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

    blocks = []
    for mode, _ctup, cols, taus in specs:
        t = cp.asarray(taus)
        if mode == "mask":
            cv, av = (cp.asarray(x) for x in cols)
            blocks.append((cv[:, None] > t[None, :]).astype(cp.float64) * av[:, None])
        else:
            cv, av, bv = (cp.asarray(x) for x in cols)
            blocks.append(cp.where(cv[:, None] > t[None, :], av[:, None], bv[:, None]))
    mat = cp.ascontiguousarray(cp.concatenate(blocks, axis=1))
    want = _plugin_mi_classif_batch_cuda_resident(mat, cp.asarray(y), 10, y_min=0, n_classes=2, relax_binning=True)
    np.testing.assert_array_equal(got, want)


def test_float32_store_equals_casting_the_float64_block():
    """The float32 build rounds each value at the store: bit for bit the cast of the float64 block."""
    cols = _cols(4, 25_000, seed=21)
    f64 = rc._build_best_existing_op_candidates_gpu_cm(cols, cp)
    f32 = rc._build_best_existing_op_candidates_gpu_cm(cols, cp, out_dtype=cp.float32)
    assert f32.dtype == cp.float32 and f32.shape == f64.shape
    np.testing.assert_array_equal(cp.asnumpy(f32), cp.asnumpy(f64.astype(cp.float32)))


def test_relaxed_candidate_dtype_follows_the_criterion_dtype(monkeypatch):
    """float32 with the relaxed criterion dtype on (default), float64 with MLFRAME_CRIT_DTYPE_RELAXED=0."""
    from mlframe.feature_selection.filters._hermite_fe_mi import relaxed_candidate_dtype

    monkeypatch.delenv("MLFRAME_CRIT_DTYPE_RELAXED", raising=False)
    assert relaxed_candidate_dtype(cp) == cp.float32
    monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", "0")
    assert relaxed_candidate_dtype(cp) == cp.float64
