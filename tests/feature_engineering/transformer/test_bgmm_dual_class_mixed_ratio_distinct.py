"""Regression test: ``compute_bgmm_dual_class_features`` ``bdc_mixed_ratio_k*`` columns must be distinct from ``bdc_loggap_virtneg_k*``.

The bug (fixed): ``mixed_ratio`` was computed with the SAME expression as ``log_gap_virtneg`` (``log(neg_d_virtual) - log(pos_d)``); the source comment even admitted
"same as log_gap_virtneg". The 4 advertised ``bdc_mixed_ratio_k*`` features were exact duplicates of the 4 ``bdc_loggap_virtneg_k*`` features, carrying zero extra
signal. The fix computes the intended distinct mixed-side quantity (real-neg vs virtual-neg distance, isolating the synthetic-augmentation effect on the neg manifold).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer import compute_bgmm_dual_class_features


pytestmark = pytest.mark.fast


def test_bgmm_mixed_ratio_not_duplicate_of_loggap_virtneg():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n, d = 240, 5
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    Xq = rng.standard_normal((60, d)).astype(np.float32)

    df = compute_bgmm_dual_class_features(X, y, Xq, None, seed=0, task="binary")

    mixed = np.column_stack([df[f"bdc_mixed_ratio_k{k}"].to_numpy() for k in (1, 3, 5, 10)])
    virtneg = np.column_stack([df[f"bdc_loggap_virtneg_k{k}"].to_numpy() for k in (1, 3, 5, 10)])

    # The two families must NOT be identical (they were exact duplicates pre-fix).
    assert not np.allclose(mixed, virtneg, atol=1e-6), (
        "bdc_mixed_ratio_k* are identical to bdc_loggap_virtneg_k*; the mixed-side ratio must compute a distinct quantity, "
        "not duplicate the virtual-neg log-gap."
    )
