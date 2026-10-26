"""The wavelet column cap keeps the most relevant columns, not the first ones by position."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._wavelet_basis_fe_recipes import generate_wavelet_features


def test_localized_column_beyond_cap_still_gets_legs():
    """A localized-bump signal in column 130 of 160 (cap 100) gets the same wavelet legs as with no cap."""
    rng = np.random.default_rng(0)
    n, p = 6000, 160
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)])
    y = ((np.abs(X["x130"]) < 0.4) + 0.3 * rng.standard_normal(n) > 0.5).astype(int).to_numpy()
    capped = generate_wavelet_features(X, y=y, max_cols=100)[0]
    free = generate_wavelet_features(X, y=y, max_cols=None)[0]
    own = lambda df: sorted(c for c in df.columns if c.startswith("x130_"))
    assert own(capped) == own(free) != []
