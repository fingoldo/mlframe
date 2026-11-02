"""Negative bin codes (pandas -1 for a missing category) must not leak into a neighbouring feature's counts (tracker 2026-11-02 wave 11, NJ11-1)."""

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import _setup_su_kernel_inputs


def test_minus_one_codes_give_the_same_entropies_as_the_shifted_column():
    """Shifting a column by a constant does not change its entropy; the -1 rows used to land in the previous feature's last bin."""
    rng = np.random.default_rng(0)
    n_features, n = 60, 400
    cols = [rng.integers(0, 6, n).astype(np.int32) for _ in range(n_features)]
    cols[7] = cols[7].copy()
    cols[7][::9] = -1
    shifted = [c.copy() for c in cols]
    shifted[7] = shifted[7] + 1
    got = _setup_su_kernel_inputs(cols, None)
    ref = _setup_su_kernel_inputs(shifted, None)
    assert got is not None and ref is not None
    np.testing.assert_allclose(got[4], ref[4])
    assert (got[0].min() >= 0) and (got[5] == ref[5]).all()
