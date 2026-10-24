"""The compiled stratified CMI and its within-stratum permutation null equal the scikit-learn / per-stratum ``rng.permutation`` forms they replace."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._cmi_strata_njit import permutation_sources, stratified_cmi, stratified_cmi_null
from mlframe.feature_selection.filters._orthogonal_triplet_fe import _dense_codes, _stratified_cmi


def _reference_cmi(x, y, s):
    """Reference cmi."""
    from sklearn.metrics import mutual_info_score

    n = y.shape[0]
    total = 0.0
    for sv in np.unique(s):
        mask = s == sv
        m = int(mask.sum())
        if m < 30:
            continue
        total += (m / n) * mutual_info_score(x[mask], y[mask])
    return total


def _reference_permutation(x, s, rng):
    """Reference permutation."""
    out = x.copy()
    for sv in np.unique(s):
        idx = np.flatnonzero(s == sv)
        out[idx] = x[rng.permutation(idx)]
    return out


def _data(seed, n=3000, nbins=10):
    """Data."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 6, n)
    y = rng.integers(0, 4, n)
    x = np.clip(y + s + rng.integers(0, 4, n), 0, nbins)  # includes the NaN-style top code
    return x.astype(np.int64), y.astype(np.int64), s.astype(np.int64)


@pytest.mark.parametrize("seed", range(5))
def test_observed_cmi_matches_sklearn(seed):
    """Observed cmi matches sklearn."""
    x, y, s = _data(seed)
    assert _stratified_cmi(x, y, s) == pytest.approx(_reference_cmi(x, y, s), abs=1e-12)
    assert _stratified_cmi(x, y, np.zeros_like(s)) == pytest.approx(_reference_cmi(x, y, np.zeros_like(s)), abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_permutation_sources_are_within_stratum_permutations(seed):
    """Every draw permutes the positions of each stratum among themselves and leaves the strata in place."""
    _x, _y, s = _data(seed)
    src = permutation_sources(s, 40, np.random.default_rng([seed, 7]))
    for row in src:
        for v in np.unique(s):
            idx = np.flatnonzero(s == v)
            np.testing.assert_array_equal(np.sort(row[idx]), idx)


@pytest.mark.parametrize("seed", range(3))
def test_null_distribution_matches_the_per_stratum_permutation(seed):
    """Rebaselined: the new generator stream gives different draws, so the null is compared in distribution, not draw by draw.

    The null mean and spread over many draws agree with the per-stratum ``rng.permutation`` reference within sampling error."""
    x, y, s = _data(seed)
    n_perm = 400
    rng = np.random.default_rng([seed, 7])
    ref = np.array([_reference_cmi(_reference_permutation(x, s, rng), y, s) for _ in range(n_perm)])
    x_d, y_d, s_d = _dense_codes(x, y, s)
    got = stratified_cmi_null(x_d, y_d, s_d, permutation_sources(s, n_perm, np.random.default_rng([seed, 7])), 30)
    se = np.sqrt(ref.var() / n_perm + got.var() / n_perm)
    assert abs(got.mean() - ref.mean()) <= 5 * se + 1e-9
    assert abs(got.std() - ref.std()) <= 0.25 * ref.std() + 1e-9


def test_small_strata_are_skipped():
    """Small strata are skipped."""
    x = np.array([0, 1] * 10, dtype=np.int64)
    y = np.array([0, 1] * 10, dtype=np.int64)
    s = np.zeros(20, dtype=np.int64)
    assert float(stratified_cmi(x, y, s, 30)) == 0.0
