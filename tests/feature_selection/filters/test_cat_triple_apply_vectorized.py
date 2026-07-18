"""Identity pin for the factorize-gather vectorized replay of apply_cat_triple_cross.

The transform-time replay was changed from a per-row ``mapping.get((a, b, c))`` Python loop to a
vectorized factorize-gather; this pins it bit-identical to the prior loop across raw + target
encodings, including unseen-triple fallbacks (sentinel for raw, global_mean for target).
``_cat_triple_fe`` is a general (non-MRMR) FE filter, so the test lives here, not under tests/.../mrmr/.
"""

import numpy as np
import pandas as pd
import pytest


def _per_row_loop(X_test, cat_a, cat_b, cat_c, mapping, *, encoding="raw", te_lookup=None, global_mean=0.0):
    """Per row loop."""
    from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str

    cats_a = _column_to_str(X_test[cat_a])
    cats_b = _column_to_str(X_test[cat_b])
    cats_c = _column_to_str(X_test[cat_c])
    n = len(cats_a)
    sentinel = len(mapping)
    if encoding == "target":
        lookup = te_lookup or {}
        out = np.empty(n, dtype=np.float64)
        for r in range(n):
            code = mapping.get((cats_a[r], cats_b[r], cats_c[r]))
            out[r] = global_mean if code is None else float(lookup.get(code, global_mean))
        return out
    out = np.empty(n, dtype=np.float64)
    for r in range(n):
        code = mapping.get((cats_a[r], cats_b[r], cats_c[r]), sentinel)
        out[r] = float(code)
    return out


@pytest.mark.parametrize("card", [3, 8, 25])
@pytest.mark.parametrize("encoding", ["raw", "target"])
def test_vectorized_replay_bit_identical_to_per_row_loop(card, encoding):
    """Vectorized replay bit identical to per row loop."""
    from mlframe.feature_selection.filters._cat_triple_fe import apply_cat_triple_cross, _encode_triple
    from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str

    rng = np.random.default_rng(card)
    n = 4000
    fit = pd.DataFrame(
        {
            "a": rng.integers(0, card, n).astype(str),
            "b": rng.integers(0, card, n).astype(str),
            "c": rng.integers(0, card, n).astype(str),
        }
    )
    _, mapping = _encode_triple(
        np.asarray(_column_to_str(fit["a"])),
        np.asarray(_column_to_str(fit["b"])),
        np.asarray(_column_to_str(fit["c"])),
    )
    # Test set deliberately spans unseen categories (card+3) to exercise the
    # sentinel (raw) / global_mean (target) unseen-triple fallback.
    test = pd.DataFrame(
        {
            "a": rng.integers(0, card + 3, n).astype(str),
            "b": rng.integers(0, card + 3, n).astype(str),
            "c": rng.integers(0, card + 3, n).astype(str),
        }
    )
    kw = {}
    if encoding == "target":
        te_lookup = {code: float(rng.normal()) for code in set(mapping.values())}
        kw = {"encoding": "target", "te_lookup": te_lookup, "global_mean": 0.5}

    expected = _per_row_loop(test, "a", "b", "c", mapping, **kw)
    got = apply_cat_triple_cross(test, "a", "b", "c", mapping, **kw)
    np.testing.assert_array_equal(got, expected)


def test_empty_frame_returns_empty_float():
    """Empty frame returns empty float."""
    from mlframe.feature_selection.filters._cat_triple_fe import apply_cat_triple_cross

    empty = pd.DataFrame({"a": [], "b": [], "c": []})
    out = apply_cat_triple_cross(empty, "a", "b", "c", {("x", "y", "z"): 0})
    assert out.shape == (0,)
    assert out.dtype == np.float64
