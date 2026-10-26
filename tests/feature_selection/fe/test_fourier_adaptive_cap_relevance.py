"""The adaptive-Fourier column cap keeps the most relevant columns, not the first ones by position."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_generate import generate_extra_basis_features


def test_periodic_column_beyond_cap_still_gets_adaptive_frequency():
    """A sine at an off-grid frequency in column 130 of 160 (cap 100) gets the same adaptive column as with no cap."""
    rng = np.random.default_rng(0)
    n, p = 6000, 160
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)])
    y = (np.sin(9.3 * X["x130"]) + 0.3 * rng.standard_normal(n) > 0).astype(int).to_numpy()
    kw = dict(extra_bases=("fourier",), fourier_adaptive=True, fourier_chirp=True)
    capped = generate_extra_basis_features(X, y=y, max_adaptive_cols=100, **kw)[0]
    free = generate_extra_basis_features(X, y=y, max_adaptive_cols=None, **kw)[0]
    own = lambda df: sorted(c for c in df.columns if "x130" in c)
    assert own(capped) == own(free)
    assert len(own(capped)) > 8
