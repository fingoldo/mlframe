"""The ALS skinny-matrix kernels reproduce the cupy matmul products they replace."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def test_matvec_and_weighted_gram_match_cupy_products():
    """Both kernels equal the cupy matmul products for tiny, ordinary and ragged-grid sizes."""
    from mlframe.feature_selection.filters.hermite_fe._als_kernels_gpu import design_matvec, weighted_gram

    rng = np.random.default_rng(0)
    for n, d in ((1, 3), (1000, 6), (50_001, 5)):
        B = cp.asarray(rng.standard_normal((n, d)))
        w = cp.asarray(rng.standard_normal(n))
        y = cp.asarray(rng.standard_normal(n))
        c = cp.asarray(rng.standard_normal(d))
        np.testing.assert_allclose(design_matvec(cp, B, c).get(), (B @ c).get(), rtol=1e-12, atol=1e-12)
        for ww in (None, w):
            A = B if ww is None else B * ww[:, None]
            ata, atb = weighted_gram(cp, B, ww, y)
            np.testing.assert_allclose(ata.get(), (A.T @ A).get(), rtol=1e-10, atol=1e-9)
            np.testing.assert_allclose(atb.get(), (A.T @ y).get(), rtol=1e-10, atol=1e-9)
