"""quantile_normal_y maps a heavily tied value to the middle of its tie block, and the round trip still recovers it."""
import numpy as np
from scipy.special import ndtri

from mlframe.training.composite.transforms.unary import quantile_normal_y_fit, quantile_normal_y_forward, quantile_normal_y_inverse


def test_mass_point_maps_to_the_block_mid_rank():
    """90% zeros: the zeros occupy ranks 0..0.9, so their mid-rank is 0.45 (T about -0.13); the last-duplicate position 0.90 gave T about +1.28."""
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(900), rng.random(100) + 1.0])
    params = quantile_normal_y_fit(y)
    t0 = float(quantile_normal_y_forward(np.array([0.0]), params)[0])
    assert abs(t0 - float(ndtri(0.45))) < 0.05, t0
    assert float(quantile_normal_y_inverse(np.array([t0]), params)[0]) == 0.0
