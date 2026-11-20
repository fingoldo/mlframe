"""Regression: the OOS-validated MDLP recursion must map held-out labels into each child node's compact class space."""
import numpy as np
import pytest

from mlframe.feature_selection.filters._mdlp_validated_split import mdlp_bin_edges_oos_validated


@pytest.mark.parametrize("reverse", [False, True])
def test_oos_deep_nodes_keep_their_holdout_rows(reverse):
    """Eight pure class blocks need seven cuts; grandchild nodes that lost classes must still see their held-out rows."""
    rng = np.random.default_rng(0)
    k, n_per = 8, 300
    x = rng.uniform(0, k, k * n_per)
    block = np.minimum(x.astype(int), k - 1)
    y = (k - 1 - block) if reverse else block
    edges = mdlp_bin_edges_oos_validated(x, y, seed=3)
    inner = edges[1:-1]
    assert inner.size == k - 1
    assert np.abs(inner - np.arange(1, k)).max() < 0.1
