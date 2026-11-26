"""Cluster 12 regression: the rich-operator bench drops its unused fit statistics and unreachable branch (audit 2026-11-20, finding 12-F87).

``sd_a``/``sd_b`` were computed and never read, and the ``ratio_eps`` operator is not in any operator list, so its branch was unreachable. The
remaining operators and the train-fit parameters are unchanged.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection._benchmarks.fs_hybrid import fe_richops_bench as rb


def test_live_operator_still_applies():
    """Control: the live ratio operator still produces its column (12-F87)."""
    X = pd.DataFrame({"a": [2.0, 4.0], "b": [1.0, 3.0]})
    out = rb._apply_op("ratio", X, [("a", "b")], {("a", "b"): {}})
    np.testing.assert_allclose(out["ratio_0"].to_numpy(), [2.0 / 2.0, 4.0 / 4.0])
