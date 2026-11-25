"""Cluster 12 regression: the rich-operator bench drops its unused fit statistics and unreachable branch (audit 2026-11-20, finding 12-F87).

``sd_a``/``sd_b`` were computed and never read, and the ``ratio_eps`` operator is not in any operator list, so its branch was unreachable. The
remaining operators and the train-fit parameters are unchanged.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.feature_selection._benchmarks.fs_hybrid import fe_richops_bench as rb


def test_dead_fit_fields_and_branch_removed():
    """The unused sd fields and the ratio_eps branch are gone from the source (12-F87)."""
    src = Path(rb.__file__).resolve().read_text(encoding="utf-8")
    assert "sd_a=" not in src
    assert '"ratio_eps"' not in src


def test_live_operator_still_applies():
    """Control: the live ratio operator still produces its column (12-F87)."""
    X = pd.DataFrame({"a": [2.0, 4.0], "b": [1.0, 3.0]})
    out = rb._apply_op("ratio", X, [("a", "b")], {("a", "b"): {}})
    np.testing.assert_allclose(out["ratio_0"].to_numpy(), [2.0 / 2.0, 4.0 / 4.0])
