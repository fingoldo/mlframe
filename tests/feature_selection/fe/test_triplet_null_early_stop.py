"""The early-stopping within-stratum null of the triplet scorer keeps every decision of the full null."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _orthogonal_triplet_fe as T
from mlframe.feature_selection.filters._cmi_strata_njit import permutation_seed, permutation_sources, permutation_sources_chunk
from mlframe.feature_selection.filters._numba_utils import stratified_cmi_null


def test_chunks_are_the_same_data_as_one_block():
    """Slices of the null drawn separately equal the rows of the single draw."""
    s = np.random.default_rng(0).integers(0, 5, 2000)
    seed = permutation_seed(np.random.default_rng([4, 9]))
    whole = permutation_sources(s, 70, np.random.default_rng([4, 9]))
    parts = np.vstack([permutation_sources_chunk(s, 0, 32, seed), permutation_sources_chunk(s, 32, 32, seed), permutation_sources_chunk(s, 64, 6, seed)])
    np.testing.assert_array_equal(whole, parts)


def _codes(rng, n=3000):
    """A binary target, a strong x, a pure-noise x and a 3-level stratum, all as dense codes."""
    s = rng.integers(0, 3, n)
    y = (rng.random(n) < 0.4 + 0.1 * s).astype(np.int64)
    strong = np.clip((y * 2 + rng.integers(0, 3, n)) // 2, 0, 3).astype(np.int64)
    noise = rng.integers(0, 4, n).astype(np.int64)
    return strong, noise, y, s.astype(np.int64)


def test_quantile_is_exact_without_a_stop_and_a_bound_with_one():
    """Strong candidate (never stops) gets numpy's quantile of the full null; a noise candidate stops with a value at or above its cmi, so its gain is not positive, as with the full null."""
    rng = np.random.default_rng(5)
    strong, noise, y, s = _codes(rng)
    n_perm, quantile = 400, 1.0 - 0.01 / 20
    for x, expect_stop in ((strong, False), (noise, True)):
        cmi = T._stratified_cmi(x, y, s)
        seed = permutation_seed(np.random.default_rng([1, 2]))
        full = stratified_cmi_null(x, y, s, permutation_sources_chunk(s, 0, n_perm, seed), 30)
        full_q = float(np.quantile(full, quantile))
        got = T._null_quantile_or_bound(x, y, s, s, cmi, n_perm, quantile, seed)
        if expect_stop:
            assert got >= cmi
            assert (cmi - got) <= 0.0 and (cmi - full_q) <= 1e-12  # the full null gives a non-positive gain too
        else:
            assert got == full_q
            assert cmi - full_q > 0.0


def test_winners_match_the_full_null():
    """On planted triplets plus noise the candidates with a positive gain are the same as with the full null for every candidate."""
    rng = np.random.default_rng(11)
    n = 4000
    raw = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(6)})
    y = ((raw["x0"] * raw["x1"] * raw["x2"]) > 0).astype(int).to_numpy()
    eng = pd.DataFrame({
        "x0*x1*x2__tri": raw["x0"] * raw["x1"] * raw["x2"],
        "x3*x4*x5__tri": raw["x3"] * raw["x4"] * raw["x5"],
        "x0*x3*x4__tri": raw["x0"] * raw["x3"] * raw["x4"],
        "x1*x2*x5__tri": raw["x1"] * raw["x2"] * raw["x5"],
    })
    got = T.score_triplet_cross_basis_by_cmi(raw, eng, y)
    assert set(got.loc[got["cmi_gain"] > 0, "engineered_col"]) == {"x0*x1*x2__tri"}
    assert (got.loc[got["engineered_col"] != "x0*x1*x2__tri", "cmi_gain"] <= 0).all()


def test_a_precomputed_strata_layout_gives_the_same_permutations():
    """``permutation_sources_chunk`` with the strata layout computed once equals the call that sorts the strata itself, for every chunk."""
    import numpy as np

    from mlframe.feature_selection.filters._cmi_strata_njit import permutation_sources_chunk, strata_layout

    s = np.random.default_rng(1).integers(0, 7, 4000)
    layout = strata_layout(s)
    for first in (0, 32, 64):
        np.testing.assert_array_equal(permutation_sources_chunk(s, first, 32, 99), permutation_sources_chunk(s, first, 32, 99, layout))
