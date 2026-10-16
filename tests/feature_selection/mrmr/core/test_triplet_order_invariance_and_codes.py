"""Triplet stage invariants that do not need a full MRMR fit.

* Rare binary levels keep their own code (quantile binning used to merge them, audit D4/G2).
* The enumerated triple ranking matches a plain-Python reference exactly, including its order.
* The CMI gate gives the same verdict for a candidate whatever its position among the engineered columns (audit adversarial: shared RNG stream).
"""

import itertools

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _orthogonal_triplet_fe as T


def test_quantile_codes_keep_rare_binary_level():
    """Quantile codes keep a rare binary level as its own code."""
    values = np.concatenate([np.zeros(950), np.ones(50)])
    codes = T._quantile_codes(values, nbins=4)
    assert len(np.unique(codes)) == 2, "a rare level must not share a code with the common one"
    assert codes[0] != codes[-1]


def _mi_reference(code, y):
    """Reference mutual information for the triplet order test."""
    k = int(code.max()) + 1
    joint = np.bincount(code * 2 + y, minlength=2 * k).reshape(k, 2).astype(np.float64)
    p = joint / joint.sum()
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


def test_enumeration_matches_python_reference_order():
    """The triplet enumeration order matches the Python reference."""
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.integers(0, 2, size=(n, 3))
    y = (x[:, 0] ^ x[:, 1] ^ x[:, 2]).astype(np.int64)
    cols = {"x1": x[:, 0].astype(float), "x2": x[:, 1].astype(float), "x3": x[:, 2].astype(float)}
    for k in range(9):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    names = sorted(X.columns)
    scored = []
    for a, b, c in itertools.combinations(names, 3):
        codes = [T._quantile_codes(X[v].to_numpy(dtype=np.float64), 4) for v in (a, b, c)]
        code = (codes[0] * 5 + codes[1]) * 5 + codes[2]
        _, inv = np.unique(code, return_inverse=True)
        scored.append((-_mi_reference(inv.astype(np.int64), y), (a, b, c)))
    scored.sort()
    reference = [tr for _, tr in scored[:5]]
    assert T._enumerate_triplet_candidates(X, list(X.columns), y, top_m=5) == reference
    assert reference[0] == ("x1", "x2", "x3")


def test_gate_verdict_does_not_depend_on_candidate_position():
    """The gate verdict does not depend on a candidate's position."""
    rng = np.random.default_rng(7)
    n = 3000
    a = rng.integers(0, 2, n).astype(float)
    b = rng.integers(0, 2, n).astype(float)
    c = rng.integers(0, 2, n).astype(float)
    y = (a * b * c).astype(np.int64)
    raw = pd.DataFrame({"A": a, "B": b, "C": c, "N": rng.standard_normal(n)})
    eng = pd.DataFrame({"A*B*C__T1_T1_T1": 2 * a - 1, "A*B*N__T1_T1_T1": (2 * a - 1) * (2 * b - 1) * rng.standard_normal(n)})
    forward = T.score_triplet_cross_basis_by_cmi(raw, eng, y, nbins=10).set_index("engineered_col")
    reverse = T.score_triplet_cross_basis_by_cmi(raw, eng[eng.columns[::-1]], y, nbins=10).set_index("engineered_col")
    pd.testing.assert_series_equal(forward["cmi_gain"], reverse.loc[forward.index, "cmi_gain"])
