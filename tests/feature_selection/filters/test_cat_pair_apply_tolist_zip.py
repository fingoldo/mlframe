"""Identity pin for the .tolist()+zip per-row replay of apply_cat_pair_cross.

The transform-time replay keeps the per-row ``mapping.get((si, sj))`` dict lookup (a prior
factorize-fold dedup was bench-rejected at 0.9x), but iterates via ``zip(cats_i.tolist(),
cats_j.tolist())`` instead of indexed ``cats_i[r]`` / ``cats_j[r]`` -- removing the per-row numpy
__getitem__ boxing (~1.17-1.22x, bench_cat_pair_cross_replay_dedup). This pins it bit-identical to
the original indexed per-row loop across raw + target encodings, including unseen-pair fallbacks
(sentinel for raw, global_mean for target). ``_cat_pair_fe`` is a general (non-MRMR) FE filter, so
the test lives here, not under tests/.../mrmr/.
"""

import numpy as np
import pandas as pd
import pytest


def _indexed_per_row_loop(X_test, cat_i, cat_j, mapping, *, encoding="raw", te_lookup=None, global_mean=0.0):
    """The PRE-optimization indexed per-row loop (cats_i[r] / cats_j[r])."""
    from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str

    cats_i = np.asarray(_column_to_str(X_test[cat_i]))
    cats_j = np.asarray(_column_to_str(X_test[cat_j]))
    n = len(cats_i)
    sentinel = len(mapping)
    lookup = te_lookup or {}

    def _value_for_pair(si, sj):
        """Value for pair."""
        if encoding == "target":
            code = mapping.get((si, sj))
            return global_mean if code is None else float(lookup.get(code, global_mean))
        return float(mapping.get((si, sj), sentinel))

    if n == 0:
        return np.empty(0, dtype=np.float64)
    return np.array([_value_for_pair(cats_i[r], cats_j[r]) for r in range(n)], dtype=np.float64)


@pytest.mark.parametrize("ki,kj", [(3, 3), (8, 5), (25, 4)])
@pytest.mark.parametrize("encoding", ["raw", "target"])
def test_apply_cat_pair_bit_identical_to_indexed_loop(ki, kj, encoding):
    """Apply cat pair bit identical to indexed loop."""
    from mlframe.feature_selection.filters._cat_pair_fe import apply_cat_pair_cross, _encode_pair
    from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str

    rng = np.random.default_rng(7)
    n_fit, n_test = 400, 300
    # Fit on a subset of the grid so some test pairs are UNSEEN (exercise fallbacks).
    df_fit = pd.DataFrame(
        {
            "ci": [f"a{v}" for v in rng.integers(0, max(ki - 1, 1), size=n_fit)],
            "cj": [f"b{v}" for v in rng.integers(0, max(kj - 1, 1), size=n_fit)],
        }
    )
    df_test = pd.DataFrame(
        {
            "ci": [f"a{v}" for v in rng.integers(0, ki, size=n_test)],
            "cj": [f"b{v}" for v in rng.integers(0, kj, size=n_test)],
        }
    )

    codes, mapping = _encode_pair(_column_to_str(df_fit["ci"]), _column_to_str(df_fit["cj"]))
    te_lookup = None
    global_mean = 0.0
    if encoding == "target":
        ymean = rng.normal(size=int(codes.max()) + 1)
        te_lookup = {int(c): float(ymean[c]) for c in range(len(ymean))}
        global_mean = -0.123

    expected = _indexed_per_row_loop(
        df_test,
        "ci",
        "cj",
        mapping,
        encoding=encoding,
        te_lookup=te_lookup,
        global_mean=global_mean,
    )
    got = apply_cat_pair_cross(
        df_test,
        "ci",
        "cj",
        mapping,
        encoding=encoding,
        te_lookup=te_lookup,
        global_mean=global_mean,
    )
    assert np.array_equal(expected, got), "tolist+zip replay diverged from indexed per-row loop"


def test_apply_cat_pair_empty():
    """Apply cat pair empty."""
    from mlframe.feature_selection.filters._cat_pair_fe import apply_cat_pair_cross

    out = apply_cat_pair_cross(pd.DataFrame({"ci": [], "cj": []}), "ci", "cj", {}, encoding="raw")
    assert out.shape == (0,)
    assert out.dtype == np.float64
