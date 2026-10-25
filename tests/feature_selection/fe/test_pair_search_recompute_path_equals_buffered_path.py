"""The pair search gives the same engineered features when the shared candidate buffer does not fit the RAM budget as when it does.

Without the buffer the candidates are materialised in bounded blocks, binned and scored in batches. That path only exists to save memory, so it must select what the buffered path selects.
"""

import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import feature_engineering as fe_mod
from mlframe.feature_selection.filters.mrmr import MRMR


def _frame(n=2500, seed=0):
    """y = a**2/b + log(c)*sin(d) over uniform positive columns plus two noise columns."""
    rng = np.random.default_rng(seed)
    cols = {name: rng.uniform(0.3, 2.0, n) for name in "abcdef"}
    y = cols["a"] ** 2 / cols["b"] + np.log(cols["c"]) * np.sin(cols["d"]) + 0.05 * rng.standard_normal(n)
    return pd.DataFrame(cols), y


def _fit(hoist):
    """Names selected by a small FE fit, with the shared-buffer decision forced to ``hoist``."""
    df, y = _frame()
    real = fe_mod._can_hoist_shared_buffer
    fe_mod._can_hoist_shared_buffer = (lambda buf, **k: (True, buf, 1 << 40)) if hoist else (lambda buf, **k: (False, buf, 1 << 20))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fs = MRMR(verbose=0, n_jobs=1, random_state=3, fe_max_steps=1).fit(df, y)
    finally:
        fe_mod._can_hoist_shared_buffer = real
    return [str(n) for n in fs.get_feature_names_out()], [str(f) for f in (fs._engineered_features_ or [])]


def test_recompute_path_selects_what_the_buffered_path_selects():
    """Same selected names and engineered features whether or not the candidate buffer is hoisted."""
    buffered = _fit(hoist=True)
    recomputed = _fit(hoist=False)
    assert recomputed == buffered
    assert buffered[1], "setup: the fit produced no engineered feature, so the comparison is vacuous"
