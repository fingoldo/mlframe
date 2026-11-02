"""A fitted MRMR no longer keeps the permutation-null closure, which pinned a float64 copy of the training frame (tracker 2026-11-02, PKL7-2)."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import MRMR


def test_fitted_mrmr_holds_no_null_closure():
    """After fit the honest-null closure attribute is None, so the frame copy it captured can be freed."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 5)), columns=list("abcde"))
    y = (X["a"] + 0.1 * rng.normal(size=400) > 0).astype(int)
    sel = MRMR(verbose=0, n_workers=1, quantization_nbins=5, full_npermutations=0, baseline_npermutations=2)
    sel.fit(X, y)
    assert getattr(sel, "_honest_null_codes_fn", None) is None
