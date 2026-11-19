"""The batched device quantile coder returns, column for column, the codes of the single-column device coder (GPU tiny-cupy item, 2026-11-18)."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no device: nothing to compare on
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._mi_greedy_cmi_fe_binning import _qbin_float_dtype, _sync_free_qbin_codes, _sync_free_qbin_codes_matrix


def _columns(rng: np.random.Generator, n: int) -> np.ndarray:
    """Continuous, tied, constant, two-valued, low-cardinality and heavy-tail columns in one matrix."""
    cols = [
        rng.normal(size=n),
        np.round(rng.normal(size=n), 1),
        np.full(n, 3.25),
        rng.integers(0, 2, n).astype(np.float64),
        rng.integers(0, 4, n).astype(np.float64),
        rng.standard_cauchy(size=n),
    ]
    return np.column_stack(cols)


@pytest.mark.parametrize("nbins", [4, 10])
@pytest.mark.parametrize("n", [257, 5000])
def test_matrix_codes_equal_the_single_column_codes(nbins: int, n: int):
    """Each column of the batched result is identical to the single-column coder on that column."""
    rng = np.random.default_rng(7)
    X = _columns(rng, n).astype(_qbin_float_dtype())  # the criterion dtype, as in production
    Xd = cp.asarray(X)
    batched = cp.asnumpy(_sync_free_qbin_codes_matrix(cp, Xd, nbins))
    for c in range(X.shape[1]):
        single = cp.asnumpy(_sync_free_qbin_codes(cp, cp.asarray(X[:, c]), nbins))
        np.testing.assert_array_equal(batched[:, c], single, err_msg=f"column {c}")


@pytest.mark.parametrize("stride", [1, 3])
def test_scoring_helper_codes_equal_the_resident_per_column_binner(stride: int):
    """The batched codes the candidate scorer uses are the resident codes of ``_quantile_bin_gpu_resident`` column for column, with the same row stride."""
    from types import SimpleNamespace

    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin_gpu_resident
    from mlframe.feature_selection.filters._mrmr_fe_step._step_score import _batched_resident_codes

    rng = np.random.default_rng(11)
    X = _columns(rng, 3000).astype(_qbin_float_dtype())
    self_ = SimpleNamespace(quantization_nbins=10)
    batched = cp.asnumpy(_batched_resident_codes(self_, X, X.shape[1], stride))
    for c in range(X.shape[1]):
        vals = X[::stride, c]
        single = cp.asnumpy(_quantile_bin_gpu_resident(vals, 10))
        np.testing.assert_array_equal(batched[:, c], single, err_msg=f"column {c}")
