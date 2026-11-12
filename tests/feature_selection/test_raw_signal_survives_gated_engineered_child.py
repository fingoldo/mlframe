"""A dominant raw feature must survive selection when the engineered child that would subsume it is gated out.

The raw-redundancy sweep judged ``f0`` subsumed by an engineered candidate, but that candidate was then rejected by the
prevalence gate, leaving nothing to carry ``f0``'s signal. ``f0`` alone determines ``y`` here, so the support must keep it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import MRMR


def test_raw_signal_survives_when_engineered_child_is_gated_out():
    """The dominant raw feature stays in support when the engineered children that would subsume it are gated out."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.int32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])

    sel = MRMR(full_npermutations=5, baseline_npermutations=3, n_jobs=1, verbose=0, random_state=11).fit(df, y)

    assert 0 in set(np.asarray(sel.support_).tolist()), f"the dominant raw feature f0 was dropped: support_={sel.support_}"
