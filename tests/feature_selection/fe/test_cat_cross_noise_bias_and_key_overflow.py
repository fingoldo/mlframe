"""Bias-corrected gate of the cat pair/triple crosses, the int64-safe triple keys and the copy-free column append."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._cat_pair_fe import hybrid_cat_pair_fe, ii_plugin_bias
from mlframe.feature_selection.filters._cat_triple_fe import _encode_triple, _join_codes, _triple_cell_keys, hybrid_cat_triple_fe


def test_ii_plugin_bias_matches_miller_madow_closed_form():
    """The bias is prod(card - 1) * (k_y - 1) / (2 n) and vanishes for a constant factor."""
    assert ii_plugin_bias((10, 10), 2, 1000) == 81 / 2000
    assert ii_plugin_bias((10, 1), 2, 1000) == 0.0


def test_pair_scores_carry_a_bias_corrected_column_centred_on_zero_for_noise():
    """On pure noise the plug-in II averages the Miller-Madow bias and the corrected column averages near zero."""
    rng = np.random.default_rng(1)
    n = 2000
    X = pd.DataFrame({f"c{i}": rng.integers(0, 6, n).astype(str) for i in range(8)})
    y = rng.integers(0, 2, n)
    scores = hybrid_cat_pair_fe(X, y)[3]
    assert scores["ii"].mean() > 0.004
    assert abs(scores["ii_mm"].mean()) < 0.25 * scores["ii"].mean()


def test_pair_gate_keeps_planted_xor_pair():
    """The corrected gate still emits a real XOR pair."""
    rng = np.random.default_rng(2)
    n = 3000
    X = pd.DataFrame({f"c{i}": rng.integers(0, 4, n).astype(str) for i in range(4)})
    y = ((X["c0"].astype(int) + X["c1"].astype(int)) % 2).to_numpy()
    assert "cross_c0_c1" in hybrid_cat_pair_fe(X, y)[1]


def test_triple_gate_rejects_noise_triples():
    """Pure-noise 10-level columns give a plug-in II3 near 0.3 nats of bias; none may be emitted."""
    rng = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame({f"c{i}": rng.integers(0, 10, n).astype(str) for i in range(6)})
    y = rng.integers(0, 2, n)
    assert hybrid_cat_triple_fe(X, y)[1] == []


def test_triple_gate_keeps_parity_triple_and_shares_column_buffers():
    """A planted three-way parity is emitted and the augmented frame shares the untouched column buffers with the input."""
    rng = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame({f"c{i}": rng.integers(0, 2, n) for i in range(5)})
    y = (X["c0"] ^ X["c1"] ^ X["c2"]).to_numpy()
    X_aug, appended, _, _ = hybrid_cat_triple_fe(X, y)
    assert "cross3_c0_c1_c2" in appended
    assert np.shares_memory(X["c3"].to_numpy(), X_aug["c3"].to_numpy())


def test_join_codes_does_not_wrap_int64():
    """Rows with distinct code triples keep distinct joint codes even when the packed key would exceed 2**64."""
    big = 2**22
    a = np.array([0, 2**20, big - 1])
    b = np.array([0, 0, big - 1])
    c = np.array([0, 0, big - 1])
    codes = _join_codes(a, b, c)
    assert len(set(codes.tolist())) == 3


def test_encode_triple_does_not_wrap_int64():
    """A value triple is never merged with another one when the three cardinalities multiply past int64."""
    big = 2**22
    uniq = np.arange(big)
    inv_a = np.array([0, 2**20, big - 1])
    inv_b = np.array([0, 0, big - 1])
    inv_c = np.array([0, 0, big - 1])
    codes, mapping = _encode_triple(precomputed_a=(uniq, inv_a), precomputed_b=(uniq, inv_b), precomputed_c=(uniq, inv_c))
    assert len(set(codes.tolist())) == 3 and len(mapping) == 3
    assert mapping[(2**20, 0, 0)] == int(codes[1])


def test_triple_cell_keys_are_order_preserving_when_packed_in_two_stages():
    """The two-stage packing keeps the lexicographic order of the (a, b, c) triples."""
    big = 2**22
    inv_a = np.array([5, 1, 1, 0])
    inv_b = np.array([0, 7, 3, big - 1])
    inv_c = np.array([2, 2, 9, big - 1])
    keys = _triple_cell_keys(inv_a, inv_b, inv_c, big, big)
    order = np.lexsort((inv_c, inv_b, inv_a))
    assert np.array_equal(np.argsort(keys, kind="stable"), order)
