"""The cat-pair interaction-information scorer (one parallel pass over all pairs) returns the frame of the per-pair ``np.unique`` + ``_plug_in_mi`` reference."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._adaptive_nbins import _plug_in_mi
from mlframe.feature_selection.filters._cat_pair_fe import _bin_target, score_cat_pairs_by_interaction_information
from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str


def _reference(X, y, columns):
    """The scorer as it was written before the batch kernel: per pair, ``np.unique`` of the Horner key and ``_plug_in_mi`` of the dense codes."""
    y_bin = _bin_target(y, n_bins=10)
    codes = {c: np.unique(_column_to_str(X[c]), return_inverse=True)[1].astype(np.int64) for c in columns}
    radix = {c: int(codes[c].max()) + 1 for c in columns}
    marginal = {c: float(_plug_in_mi(codes[c], y_bin)) for c in columns}
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            joint = np.unique(codes[a] * radix[b] + codes[b], return_inverse=True)[1]
            mi_joint = float(_plug_in_mi(joint, y_bin))
            rows.append({"cat_i": a, "cat_j": b, "mi_joint": mi_joint, "mi_i": marginal[a], "mi_j": marginal[b], "ii": mi_joint - marginal[a] - marginal[b]})
    return pd.DataFrame(rows).sort_values("ii", ascending=False, kind="mergesort").reset_index(drop=True)


def test_scores_equal_the_per_pair_reference():
    """Eight categorical columns of mixed cardinality and a target tied to one pair: every pair row (joint MI, marginals, II) is identical, the tied pair first."""
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({f"c{i}": rng.integers(0, 3 + 4 * i, n).astype(str) for i in range(8)})
    y = ((X["c1"].astype(int) + X["c2"].astype(int)) % 3 == 0).astype(int).to_numpy()
    columns = list(X.columns)
    got = score_cat_pairs_by_interaction_information(X, y, columns)
    want = _reference(X, y, columns)
    pd.testing.assert_frame_equal(got[["cat_i", "cat_j", "mi_joint", "mi_i", "mi_j", "ii"]], want[["cat_i", "cat_j", "mi_joint", "mi_i", "mi_j", "ii"]])
    assert got.iloc[0][["cat_i", "cat_j"]].tolist() == ["c1", "c2"]


def test_explicit_pair_list_and_continuous_target():
    """An explicit pair list and a continuous (quantile-binned) target take the same path."""
    rng = np.random.default_rng(1)
    n = 2000
    X = pd.DataFrame({f"c{i}": rng.integers(0, 6 + i, n).astype(str) for i in range(5)})
    y = rng.normal(size=n) + (X["c0"].astype(int) % 2)
    got = score_cat_pairs_by_interaction_information(X, y.to_numpy(), list(X.columns), pairs=[("c0", "c3"), ("c1", "c2")])
    assert len(got) == 2 and set(map(tuple, got[["cat_i", "cat_j"]].to_numpy())) == {("c0", "c3"), ("c1", "c2")}
    assert np.isfinite(got["ii"]).all()
