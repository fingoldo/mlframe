"""``validate_pair_fe_cv`` indexes the target positionally even when it is a Series with a shuffled index."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import composition


class _Res:
    """Stub of a HermiteResult."""

    mi = 1.0
    bin_func_name = "add"

    def transform(self, a, b):
        """Sum of the two inputs."""
        return a + b


def test_series_target_with_shuffled_index_matches_ndarray_target(monkeypatch):
    """Label-based ``y[tr]`` picked other rows than the fold's; the result must equal the ndarray run."""
    import mlframe.feature_selection.filters.hermite_fe as hf

    monkeypatch.setattr(hf, "optimise_hermite_pair", lambda *a, **k: _Res())
    rng = np.random.default_rng(0)
    n = 300
    xa = rng.standard_normal(n)
    xb = rng.standard_normal(n)
    y = (xa + xb > 0).astype(int)
    perm = rng.permutation(n)
    ys = pd.Series(y, index=perm)
    a = composition.validate_pair_fe_cv(xa, xb, y, n_splits=3)
    b = composition.validate_pair_fe_cv(xa, xb, ys, n_splits=3)
    assert a["oos_per_fold"] == b["oos_per_fold"]
