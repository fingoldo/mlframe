"""The honest null draws its seeded permutations once per spec, not once per column batch, and the labels are the ones an uncached draw gives."""
import numpy as np

from mlframe.feature_selection.filters._honest_mdlp_null import _PERM_LABELS_CACHE_MAX_BYTES, _permuted_labels_cached
from mlframe.feature_selection.filters._mdlp_honest_bycol_njit import permuted_labels_seeded


class _CountingRng:
    """A seeded generator that counts the seeds it is asked for (a draw takes one seed and generates its shuffles from it)."""

    def __init__(self, seed):
        """Wrap a default generator."""
        self.rng = np.random.default_rng(seed)
        self.draws = 0

    def integers(self, low, high):
        """One seed."""
        self.draws += 1
        return self.rng.integers(low, high)


def _reference(y_raw, y_codes, seed, count):
    """The labels of the draw a generator seeded ``seed`` makes: its first seed expands to ``count`` shuffles."""
    return permuted_labels_seeded(y_raw, y_codes, count, int(np.random.default_rng(seed).integers(0, 2**31 - 1)))


def _labels(n=500):
    """Raw and coded labels of a three-class target."""
    y = np.random.default_rng(1).integers(0, 3, n).astype(np.int64)
    return y, y.copy()


def test_second_call_with_the_same_key_draws_nothing_and_matches_uncached():
    """Same seed, rows and permutation count: the labels equal the uncached ones and the second call costs no shuffle."""
    y_raw, y_codes = _labels()
    n, count, seed = len(y_raw), 40, 7
    spec: dict = {}
    first_rng = _CountingRng(seed)
    first = _permuted_labels_cached(spec, y_raw, y_codes, first_rng, n, count, seed)
    second_rng = _CountingRng(seed)
    second = _permuted_labels_cached(spec, y_raw, y_codes, second_rng, n, count, seed)
    reference = _reference(y_raw, y_codes, seed, count)
    assert first_rng.draws == 1 and second_rng.draws == 0
    for got, again, want in zip(first, second, reference):
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(again, want)


def test_a_different_seed_or_label_vector_is_a_different_entry():
    """The key holds the seed and the label content, so neither is served another draw's matrices."""
    y_raw, y_codes = _labels()
    n, count = len(y_raw), 20
    spec: dict = {}
    a = _permuted_labels_cached(spec, y_raw, y_codes, _CountingRng(1), n, count, 1)
    b = _permuted_labels_cached(spec, y_raw, y_codes, _CountingRng(2), n, count, 2)
    assert not np.array_equal(a[0], b[0])
    flipped = (y_raw + 1) % 3
    c_rng = _CountingRng(2)
    _permuted_labels_cached(spec, flipped, flipped.copy(), c_rng, n, count, 2)
    assert c_rng.draws == 1


def test_a_draw_above_the_byte_cap_is_not_kept_and_says_so(caplog):
    """Beyond the cap the matrices are regenerated each call, nothing is stored on the spec, and the skipped cache is logged."""
    y_raw, y_codes = _labels(200)
    count = _PERM_LABELS_CACHE_MAX_BYTES // (2 * len(y_raw)) + 1
    spec: dict = {}
    rng = _CountingRng(3)
    with caplog.at_level("INFO", logger="mlframe.feature_selection.filters._honest_mdlp_null"):
        _permuted_labels_cached(spec, y_raw, y_codes, rng, len(y_raw), count, 3)
    assert not spec.get("_perm_labels")
    assert any("exceed" in r.getMessage() and "cache cap" in r.getMessage() for r in caplog.records)


def test_a_shorter_count_is_served_from_a_longer_draw_and_a_longer_one_redraws():
    """The first k permutations of a seed's stream are the same whatever the total, so a smaller count slices the cached draw and a larger one extends it."""
    y_raw, y_codes = _labels()
    n, seed = len(y_raw), 5
    spec: dict = {}
    _permuted_labels_cached(spec, y_raw, y_codes, _CountingRng(seed), n, 60, seed)
    short_rng = _CountingRng(seed)
    short = _permuted_labels_cached(spec, y_raw, y_codes, short_rng, n, 25, seed)
    want = _reference(y_raw, y_codes, seed, 25)
    assert short_rng.draws == 0
    np.testing.assert_array_equal(short[0], want[0])
    np.testing.assert_array_equal(short[1], want[1])
    long_rng = _CountingRng(seed)
    longer = _permuted_labels_cached(spec, y_raw, y_codes, long_rng, n, 90, seed)
    assert long_rng.draws == 1 and longer[0].shape[0] == 90


def test_the_seeded_shuffles_are_permutations_and_depend_on_the_seed_alone():
    """Every row of a draw holds the labels of the target in another order (same multiset), the rows differ from each other, and the same seed gives the same matrices."""
    y_raw, y_codes = _labels()
    a = permuted_labels_seeded(y_raw, y_codes, 12, 99)
    b = permuted_labels_seeded(y_raw, y_codes, 12, 99)
    np.testing.assert_array_equal(a[0], b[0])
    for row in a[0]:
        np.testing.assert_array_equal(np.sort(row), np.sort(y_raw))
    assert len({row.tobytes() for row in a[0]}) == 12
    assert not np.array_equal(a[0], permuted_labels_seeded(y_raw, y_codes, 12, 100)[0])
