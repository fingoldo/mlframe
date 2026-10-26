"""The parallel within-stratum permutation draws do not depend on the number of threads."""
import numba
import numpy as np

from mlframe.feature_selection.filters._cmi_strata_njit import permutation_sources


def test_draws_are_identical_for_one_and_many_threads():
    """Draw p is seeded from its own index, so the schedule cannot change a row."""
    s = np.random.default_rng(0).integers(0, 7, 5000)
    many = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        one = permutation_sources(s, 64, np.random.default_rng([3, 7]))
        numba.set_num_threads(many)
        par = permutation_sources(s, 64, np.random.default_rng([3, 7]))
    finally:
        numba.set_num_threads(many)
    np.testing.assert_array_equal(one, par)
