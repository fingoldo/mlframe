"""The adaptive Fourier detector finds the high-frequency tone of a square wave on a few hundred rows and invents none on noise."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_generate import generate_extra_basis_features


def _square_wave(n, seed):
    """A normal column and the parity of floor(3 x): a square wave of about eleven cycles over the column's range."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    return pd.DataFrame({"x": x}), ((np.floor(x * 3).astype(int)) % 2).astype(int)


def test_square_wave_tone_is_detected_on_560_rows():
    """560 rows (the 70% train split of 800) give an adaptive column that correlates strongly with the parity target."""
    X, y = _square_wave(560, 0)
    cols = generate_extra_basis_features(X, y=y, extra_bases=("fourier",), fourier_adaptive=True)[0]
    best = max(abs(np.corrcoef(cols[c], y)[0, 1]) for c in cols.columns)
    assert best > 0.6


def test_pure_noise_gets_no_adaptive_tone():
    """On 40 noise columns of 560 rows no frequency beyond the fixed grid is added."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((560, 40)), columns=[f"x{i}" for i in range(40)])
    y = (rng.random(560) > 0.5).astype(int)
    cols = generate_extra_basis_features(X, y=y, extra_bases=("fourier",), fourier_adaptive=True)[0]
    assert not [c for c in cols.columns if "." in c]
