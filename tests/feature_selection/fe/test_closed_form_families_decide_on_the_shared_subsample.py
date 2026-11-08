"""Modular, numeric_decompose, parity, hinge and wavelet decide on the fit's shared row subsample and replay their winners at full n."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _periodic_fe, _numeric_decompose_fe, _parity_fe
from mlframe.feature_selection.filters._mrmr_fit_impl._helpers import fe_decide_on_subsample

_FAMILIES = [
    ("modular", _periodic_fe, "hybrid_modular_fe_with_recipes", {"cols": None, "periods": (7, 12), "top_k": 3, "seed": 0}),
    ("numeric_decompose", _numeric_decompose_fe, "hybrid_numeric_decompose_fe_with_recipes", {"cols": None, "top_k": 3, "n_boot": 3, "seed": 0}),
    ("parity", _parity_fe, "hybrid_parity_fe_with_recipes", {"cols": None}),
]


@pytest.mark.parametrize("name,module,func,kwargs", _FAMILIES, ids=[f[0] for f in _FAMILIES])
def test_the_family_sees_only_the_subsample_and_returns_full_length_columns(name, module, func, kwargs):
    """The family is called on the drawn rows only, and whatever it appends comes back at the full length with its recipe."""
    rng = np.random.default_rng(0)
    n, m = 6000, 1500
    X = pd.DataFrame({"a": rng.integers(0, 40, n).astype(float), "b": rng.standard_normal(n), "c": rng.standard_normal(n)})
    y = (X["a"].to_numpy() % 7 == 0).astype(int)
    seen_rows: list = []
    real = getattr(module, func)

    def spy(X_in, y_in, **kw):
        """Record how many rows the family was given, then run the real family."""
        seen_rows.append(len(X_in))
        return real(X_in, y_in, **kw)

    idx = np.sort(rng.choice(n, m, replace=False)).astype(np.int64)
    out = fe_decide_on_subsample(spy, X, y, shared_subsample_idx=idx, recipes_at=2, **kwargs)
    assert seen_rows == [m]
    X_aug, appended, recipes = out[0], out[1], out[2]
    assert len(X_aug) == n
    assert set(appended) <= set(X_aug.columns)
    assert {r.name for r in recipes} >= set(appended)
