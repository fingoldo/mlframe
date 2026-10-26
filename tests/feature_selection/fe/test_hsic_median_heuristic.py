"""The compiled median-heuristic bandwidth equals the dense upper-triangle definition."""
import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_hsic_fe import median_heuristic_sigma


def _dense(z: np.ndarray) -> float:
    """The definition: median of |z_i - z_j| over i < j."""
    diff = np.abs(z[:, None] - z[None, :])
    med = float(np.median(diff[np.triu_indices(z.shape[0], k=1)]))
    return 1.0 if med <= 0.0 else med


@pytest.mark.parametrize("n", [2, 3, 10, 499, 500])
def test_equals_dense_definition(n):
    """Even and odd pair counts, with ties."""
    rng = np.random.default_rng(n)
    z = np.round(rng.normal(size=n), 1)
    assert median_heuristic_sigma(z) == pytest.approx(_dense(np.asarray(z, dtype=np.float64)), rel=1e-6)


def test_constant_and_tiny_inputs_fall_back_to_one():
    """A constant column and a single value give the neutral 1.0."""
    assert median_heuristic_sigma(np.zeros(50)) == 1.0
    assert median_heuristic_sigma(np.array([3.0])) == 1.0
