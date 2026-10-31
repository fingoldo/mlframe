"""The masked scatter moments on the device keep float64 accuracy on a large-offset, small-scale column."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def test_masked_scatter_moments_match_numpy_on_large_offset_column():
    """Per-cell count, mean and centred second moment over the masked-in rows agree with numpy to double precision."""
    try:
        cp.zeros(1)
    except Exception as exc:  # no usable device
        pytest.skip(f"no CUDA device: {exc}")
    from mlframe.feature_selection.filters._binned_numeric_agg_resident import _per_cell_moments_stable_masked_scatter_gpu

    rng = np.random.default_rng(3)
    n_cells, per_cell = 5, 500
    codes = np.repeat(np.arange(n_cells), per_cell)
    values = 8.5e3 + 0.06 * rng.standard_normal(codes.shape[0])
    keep = (rng.random(codes.shape[0]) > 0.3).astype(np.float64)
    cnt, mean, cm2, _cm3, _cm4 = _per_cell_moments_stable_masked_scatter_gpu(cp, cp.asarray(codes, dtype=cp.int64), cp.asarray(values), cp.asarray(keep), n_cells)
    for c in range(n_cells):
        rows = (codes == c) & (keep > 0)
        ref_mean = values[rows].mean()
        assert float(cnt[c]) == rows.sum()
        assert abs(float(mean[c]) - ref_mean) < 1e-9
        assert abs(float(cm2[c]) - ((values[rows] - ref_mean) ** 2).sum()) < 1e-9
