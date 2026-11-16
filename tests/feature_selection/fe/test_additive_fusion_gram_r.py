"""The Gram-matrix form of the additive-fusion separability fits matches the lstsq fits, and the host / device proposals admit the same fusions."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_additive_fusion as fus


def _lstsq_r(cols, y):
    """Multiple R of the OLS fit of y on cols with an intercept."""
    d = np.column_stack([*cols, np.ones(len(y))])
    beta = np.linalg.lstsq(d, y, rcond=None)[0]
    return abs(np.corrcoef(d @ beta, y)[0, 1])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gram_r_matches_lstsq(seed):
    """The four fits of a pair from one Gram matrix equal the lstsq fits."""
    r = np.random.default_rng(seed)
    n = 5000
    a, b = r.standard_normal(n), r.standard_normal(n) ** 2
    y = 0.7 * a + 0.3 * b + r.standard_normal(n)
    fused = a + b
    got = fus._pair_multiple_r(a, b, fused, y)
    want = (_lstsq_r([a], y), _lstsq_r([b], y), _lstsq_r([a, b], y), _lstsq_r([fused], y))
    assert np.allclose(got, want, atol=1e-6)


def test_gram_r_degenerate_columns_give_zero():
    """Constant columns and a constant target give 0.0; a collinear pair block still gives the single R."""
    n = 100
    y = np.arange(n, dtype=float)
    assert fus._pair_multiple_r(np.zeros(n), np.ones(n), np.zeros(n), y)[:2] == (0.0, 0.0)
    assert fus._pair_multiple_r(y, y, y, np.ones(n)) == (0.0, 0.0, 0.0, 0.0)
    got = fus._pair_multiple_r(y, 2 * y, y, y + 1e-3 * np.sin(y))  # collinear pair block
    assert got[2] > 0.99


def test_host_and_device_admit_the_same_fusions():
    """The sweep fixture admits the identical set on both paths."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _fe_additive_fusion_ktc as ktc

    kw, fs = ktc._fusion_inputs({"n_samples": 20000})
    assert np.array_equal(ktc._fusion_host(kw, fs), ktc._fusion_device(kw, fs))
