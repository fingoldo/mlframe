"""Identity regression for the auto-scorer batched separable path (CPX12).

The column-separable scorers (plug-in MI, copula MI) are now batched across all
columns in one kernel call inside _compute_per_scorer_rank_table. This pins
bit-identity vs the prior per-column _score_plug_in / _score_copula path, and
that the full rank table is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._orth_auto_scorer_fe import (
    _score_copula,
    _score_plug_in,
)
from mlframe.feature_selection.filters._orthogonal_copula_mi_fe import _copula_mi_batch
from mlframe.feature_selection.filters._orthogonal_scorer_auto_fe import (
    _compute_per_scorer_rank_table,
)
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch


@pytest.mark.parametrize("seed", [0, 2, 5])
def test_batch_matches_per_column(seed):
    rng = np.random.default_rng(seed)
    n, p = 1200, 30
    X = rng.standard_normal((n, p))
    y = rng.integers(0, 4, n)

    mi_pc = np.array([_score_plug_in(X[:, j], y, nbins=10) for j in range(p)])
    cop_pc = np.array([_score_copula(X[:, j], y, n_bins=20) for j in range(p)])
    mi_b = np.asarray(_mi_classif_batch(X, y.astype(np.int64), nbins=10))
    cop_b = np.asarray(_copula_mi_batch(X, y, n_bins=20))
    assert np.array_equal(mi_pc, mi_b)
    assert np.array_equal(cop_pc, cop_b)


def test_rank_table_plug_in_copula_unchanged():
    rng = np.random.default_rng(0)
    n = 1000
    raw = pd.DataFrame({f"r{j}": rng.standard_normal(n) for j in range(4)})
    eng = pd.DataFrame({
        f"r{j}__sq": (raw[f"r{j}"] ** 2).to_numpy() for j in range(4)
    })
    y = (raw["r0"] > 0).astype(np.int64).to_numpy()

    table, baseline, ranks = _compute_per_scorer_rank_table(
        raw, eng, y, scorers=["plug_in", "copula"], random_state=0,
        nbins=10, n_neighbors=3, copula_n_bins=20, dcor_n_sample=500,
    )
    # Reference: score each engineered column individually.
    for col in eng.columns:
        xv = eng[col].to_numpy(dtype=np.float64)
        ref_mi = _score_plug_in(xv, y, nbins=10)
        ref_cop = _score_copula(xv, y, n_bins=20)
        rowmask = table["engineered_col"] == col
        assert float(table.loc[rowmask, "score_plug_in"].iloc[0]) == ref_mi
        assert float(table.loc[rowmask, "score_copula"].iloc[0]) == ref_cop
