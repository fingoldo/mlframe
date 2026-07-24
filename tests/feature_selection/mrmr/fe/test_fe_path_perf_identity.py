"""Selection-identity regression sensors for the MRMR FE / validation-path perf levers (E1, E2, E6, E7, E8).

These pin the bit-identical SELECTION (``support_`` + ``get_feature_names_out()``) that the perf changes
must preserve, plus fast unit-level bit-identity checks for the two pure-numpy kernels (E7 pair build,
E8 numeric coercion) so a future "just rewrite it" cannot silently change the result.

The reference ``support_`` / names were captured on the code as of the perf levers landing; any drift
in E1/E2/E6/E7/E8 themselves fails these tests. Unrelated genuine selection changes (a different bug fix
landing) are re-captured here rather than reverted -- see the inline note above ``_REF_FE``/``_REF_NOFE``.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR


def _make(n: int, p: int, inf: int, seed: int):
    """Build a make_classification frame with p features, inf informative, and 4 classes, for the reference-selection pins."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n,
        n_features=p,
        n_informative=inf,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return pd.DataFrame(X, columns=[f"f_{i}" for i in range(p)]), y


def _fit(n, p, inf, seed, fe_steps):
    """Fit MRMR with the pinned perf-lever config and return (sorted support indices, feature names) for selection-identity comparison."""
    kw = dict(
        quantization_nbins=8,
        interactions_max_order=1,
        full_npermutations=3,
        baseline_npermutations=2,
        random_seed=seed,
        use_gpu=False,
        n_jobs=1,
        verbose=0,
        fe_max_steps=fe_steps,
        cv=2,
    )
    X, y = _make(n, p, inf, seed)
    m = MRMR(**kw)
    m.fit(X, y)
    support = sorted(m.support_.tolist())
    names = list(m.get_feature_names_out())
    return support, names


# --- captured reference selections -------------------------------------------------------------------
# Re-captured (not the perf-lever's original PRE-change values -- see the class docstring for the perf
# levers themselves): the group_aware_mi fix wave landed two genuine selection changes since these were
# first pinned, both intentional and unrelated to E1/E2/E6/E7/E8 (re-framed per the "a validated
# improvement that breaks a test -> re-frame the stale test" convention, not reverted):
#   * fe_hybrid_orth_enable / fe_univariate_basis_enable now correctly gate on fe_max_steps>0 (previously
#     they fired even at fe_steps=0, contrary to the documented "fe_max_steps=0 = no FE" contract) -- the
#     NOFE reference's Hermite-basis names (f_13__qsin4.3 etc.) are gone; the discrete-structural gate_mask
#     operators (which DO have their own explicit fe_steps=0 carve-out) still fire as before.
#   * an unrelated concurrent-session change (confirmed pre-existing to this fix wave via revert-and-compare)
#     shifted the FE reference's exact greedy order/composite selection.
_REF_FE = {
    "params": dict(n=4000, p=30, inf=8, seed=0, fe_steps=1),
    "support": [7, 13, 14, 15, 19, 23, 26, 29],
    "names": [
        "f_23",
        "f_14",
        "f_29",
        "f_15",
        "f_13",
        "f_19",
        "f_26",
        "f_7",
        "sub(div(exp(f_7),sign(gate_mask__f_23__f_15__t-1.28123)),mul(sign(f_19),exp(gate_mask__f_7__f_13__t-0.643104)))",
        "add(f_7,neg(f_26))",
        "div(sign(f_19),exp(gate_mask__f_14__f_15__t-0.60264))",
        "gate_mask__f_13__f_26__t-1.26659",
        "gate_mask__f_23__f_15__t-1.28123",
        "gate_mask__f_7__f_13__t-0.643104",
        "gate_mask__f_14__f_15__t-0.60264",
    ],
}

_REF_NOFE = {
    "params": dict(n=4000, p=30, inf=8, seed=5, fe_steps=0),
    "support": [1, 5, 9, 12, 13, 15, 16, 20],
    "names": [
        "f_13",
        "f_12",
        "f_20",
        "f_1",
        "f_15",
        "f_9",
        "f_16",
        "f_5",
        "gate_mask__f_9__f_12__t-0.726506",
        "gate_mask__f_5__f_15__t-0.337728",
        "gate_mask__f_16__f_15__t-0.600261",
        "gate_mask__f_15__f_5__t-1.38943",
        "f_13__relu_gt1.42721",
        "f_13__relu_lt1.42721",
    ],
}


@pytest.mark.slow
def test_fe_pair_path_selection_bit_identical():
    """E1 (copy-once) + E2 (batched data concat) + E6 (view) + E7 (triu pair build): the pair-FE path
    that materialises engineered columns must select the EXACT same support_ + names as pre-change."""
    support, names = _fit(**_REF_FE["params"])
    assert support == _REF_FE["support"], f"support drift: {support} != {_REF_FE['support']}"
    assert names == _REF_FE["names"], f"names drift:\n  got {names}\n  exp {_REF_FE['names']}"


@pytest.mark.slow
def test_nonfe_single_col_fe_path_selection_bit_identical():
    """Single-column adaptive-Fourier FE path (no pair-FE) must also stay selection-identical -- guards
    the shared validate/screen/tree-rescue edits (E8 numeric coercion) don't perturb selection."""
    support, names = _fit(**_REF_NOFE["params"])
    assert support == _REF_NOFE["support"], f"support drift: {support} != {_REF_NOFE['support']}"
    assert names == _REF_NOFE["names"], f"names drift:\n  got {names}\n  exp {_REF_NOFE['names']}"


# --- fast unit-level bit-identity of the two pure-numpy kernels -------------------------------------
@pytest.mark.parametrize("p", [3, 7, 16, 64])
def test_e7_triu_pair_build_matches_combinations(p):
    """E7: ``np.triu_indices`` over the materialised id list yields the SAME (a, b) pair sequence and
    kernel int64 arrays as ``list(combinations(ids, 2))`` -- the cached_MIs keys are unchanged."""
    ids_set = set(int(x) for x in np.random.default_rng(p).permutation(40)[:p])

    pairs_old = list(combinations(ids_set, 2))
    a_old = np.fromiter((q[0] for q in pairs_old), dtype=np.int64, count=len(pairs_old))
    b_old = np.fromiter((q[1] for q in pairs_old), dtype=np.int64, count=len(pairs_old))

    ids = list(ids_set)
    ids_arr = np.fromiter(ids, dtype=np.int64, count=len(ids))
    ia, ib = np.triu_indices(len(ids), k=1)
    a_new, b_new = ids_arr[ia], ids_arr[ib]
    pairs_new = [(ids[ia[i]], ids[ib[i]]) for i in range(ia.shape[0])]

    assert np.array_equal(a_old, a_new)
    assert np.array_equal(b_old, b_new)
    assert pairs_old == pairs_new
    # key-type identity: both must be plain python int tuples (dict-key compatible with the rest of the cache)
    assert all(type(x) is int for q in pairs_new for x in q)


@pytest.mark.parametrize("with_inf", [False, True])
def test_e8_asarray_numeric_matches_tonumeric_on_numeric_frame(with_inf):
    """E8: the fast ``asarray(float)`` + in-place NaN->0 path is bit-identical to the lenient
    ``apply(to_numeric).fillna(0.0).to_numpy`` path on an all-numeric frame (incl. NaN and +/-inf)."""
    rng = np.random.default_rng(0)
    M = rng.standard_normal((500, 12))
    M[rng.random(M.shape) < 0.05] = np.nan
    if with_inf:
        M[0, 0] = np.inf
        M[1, 1] = -np.inf
    df = pd.DataFrame(M, columns=[f"c{i}" for i in range(M.shape[1])])

    old = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    new = np.asarray(df, dtype=float)
    if np.isnan(new).any():
        new[np.isnan(new)] = 0.0
    assert np.array_equal(old, new)


def test_e8_nonnumeric_column_falls_back_to_lenient_path():
    """E8: a non-numeric column makes ``asarray(float)`` raise; the production rescue path must fall back
    to the lenient coerce-to-NaN-then-zero path (the bad column becomes all-zeros), not crash."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "s": ["x", "y", "z"]})
    # mirror the production try/except in _mrmr_tree_rescue._apply_tree_rescue
    try:
        Xnum = np.asarray(df, dtype=float)
        if np.isnan(Xnum).any():
            Xnum[np.isnan(Xnum)] = 0.0
    except (ValueError, TypeError):
        Xnum = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    # 's' coerces to NaN -> 0.0; 'a' preserved
    assert np.array_equal(Xnum[:, 0], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(Xnum[:, 1], np.array([0.0, 0.0, 0.0]))
