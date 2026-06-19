"""Pin for the chunked engineered-MI scoring in score_features_by_mi_uplift (loop iter 2026-06-19).

Materialising the full engineered matrix as one float64 array OOM'd at scale (measured (16000,20000) float64
= 2.38 GiB). MI is per-column, so scoring in column BLOCKS must be BIT-IDENTICAL to the all-at-once call --
this test pins that on a frame WIDER than the 1024-col chunk boundary so the chunked path is exercised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe import score_features_by_mi_uplift
from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_mi_backends import _mi_classif_batch


def test_chunked_engineered_mi_bit_identical_to_full():
    rng = np.random.default_rng(0)
    n, n_eng = 3000, 1500          # > 1024 -> exercises the chunked path
    y = (rng.random(n) < 0.5).astype(np.int64)
    raw = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    # engineered columns named "<source>__k" so the uplift baseline lookup works
    eng = pd.DataFrame(
        {f"a__{k}": rng.standard_normal(n) for k in range(n_eng)}
    )

    # reference: score the FULL engineered matrix in one call (the pre-fix path)
    ref = _mi_classif_batch(eng.to_numpy(dtype=np.float64), y, nbins=10)
    ref_map = {name: float(ref[j]) for j, name in enumerate(eng.columns)}

    out = score_features_by_mi_uplift(raw, eng, y, nbins=10)
    got_map = dict(zip(out["engineered_col"], out["engineered_mi"]))

    assert set(got_map) == set(ref_map)
    for name, ref_mi in ref_map.items():
        assert got_map[name] == ref_mi, f"chunked MI != full MI for {name}: {got_map[name]} vs {ref_mi}"


def test_pair_cross_chunked_engineered_mi_bit_identical_to_full():
    """Same chunked-MI fix on the (wider) pair-cross-basis scorer: bit-identical to the full-matrix call."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import score_pair_cross_basis_by_mi_uplift

    rng = np.random.default_rng(1)
    n, n_eng = 3000, 1300          # > 1024 -> chunked path
    y = (rng.random(n) < 0.5).astype(np.int64)
    raw = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    # pair-cross engineered names "{i}*{j}__k" so the baseline lookup parses
    eng = pd.DataFrame({f"a*b__{k}": rng.standard_normal(n) for k in range(n_eng)})

    ref = _mi_classif_batch(eng.to_numpy(dtype=np.float64), y, nbins=10)
    ref_map = {name: float(ref[j]) for j, name in enumerate(eng.columns)}

    out = score_pair_cross_basis_by_mi_uplift(raw, eng, y, nbins=10)
    got_map = dict(zip(out["engineered_col"], out["engineered_mi"]))
    assert set(got_map) == set(ref_map)
    for name, ref_mi in ref_map.items():
        assert got_map[name] == ref_mi, f"pair chunked MI != full MI for {name}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s", "--no-cov"])
