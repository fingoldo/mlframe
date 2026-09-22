"""The batched artifact histograms are exactly what the per-column ``np.bincount`` produced.

The artifact builder computes ``H(X_j)`` for every retained feature and did it one column at a time: a strided gather out of the shared binned
matrix plus an ``np.bincount``, per feature, from Python, in a module with nothing compiled in it at all. The counts are now computed for
every column in one parallel pass. They are integers, so "the same histogram" is a literal claim and is asserted as one; the entropy's float
reduction was deliberately left in the caller so no reported value shifts in its last bits.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_artifact_entropy import column_histograms


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_batched_histograms_equal_bincount(seed):
    """Every requested column's counts match ``np.bincount`` on that column exactly."""
    rng = np.random.default_rng(seed)
    n, k, width = 900, 6, 12
    data = rng.integers(0, width, size=(n, k)).astype(np.int64)
    cols = [0, 2, 3, 5]
    got = column_histograms(data, cols, width)
    for j, c in enumerate(cols):
        want = np.bincount(data[:, c], minlength=width)
        assert np.array_equal(got[j], want), f"column {c}: {got[j]} vs {want}"


def test_a_constant_column_lands_entirely_in_one_bin():
    """A column with no variation puts every row in its single bin and nothing anywhere else."""
    data = np.zeros((300, 2), dtype=np.int64)
    data[:, 1] = 3
    got = column_histograms(data, [0, 1], 8)
    assert got[0, 0] == 300 and got[0, 1:].sum() == 0
    assert got[1, 3] == 300 and got[1].sum() == 300


def test_bins_no_row_reaches_stay_at_zero():
    """A width wider than the codes present must leave the unused bins empty, as ``minlength`` does."""
    data = np.full((50, 1), 2, dtype=np.int64)
    got = column_histograms(data, [0], 9)
    assert got.shape == (1, 9)
    assert got[0, 2] == 50 and got[0].sum() == 50


def test_requesting_no_columns_returns_an_empty_block():
    """Nothing to histogram is an empty result, not an error, so the caller's guard stays simple."""
    assert column_histograms(np.zeros((10, 3), dtype=np.int64), [], 5).shape == (0, 5)


def test_the_entropies_a_fit_reports_match_a_bincount_recomputation():
    """End to end: the H(X) implied by every exported SU and MI is the entropy recomputed from ``np.bincount`` on the exported bins."""
    import pandas as pd

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    # Three informative columns of decreasing strength, so the identity below is checked on more than one feature, plus one pure-noise
    # column that legitimately reports SU=0 and carries no equation.
    y = ((a + 0.7 * b + 0.3 * c + 0.25 * rng.normal(size=n)) > 0).astype(np.int64)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": rng.normal(size=n)})
    MRMR._FIT_CACHE.clear()
    est = MRMR(random_state=0, verbose=0, fe_max_steps=0, full_npermutations=3, baseline_npermutations=2, retain_artifacts=True).fit(X, y)

    # ``export_artifacts()`` is the surface: a fit asked to retain artifacts that cannot produce them is a broken contract, not a
    # configuration to step around, and the capture path swallows its own failure into a warning.
    artifacts = est.export_artifacts()
    names = artifacts["feature_names"]
    su = np.asarray(artifacts["su_to_target"], dtype=np.float64)
    mi = np.asarray(artifacts["mi_to_target"], dtype=np.float64)
    bins, nbins = artifacts["bins"], artifacts["nbins_per_feature"]
    assert set(bins) == set(names), f"exported bins {sorted(bins)} do not cover the exported features {sorted(names)}"

    y_counts = np.bincount(y, minlength=2).astype(np.float64)
    y_p = y_counts / y_counts.sum()
    h_y = float(-np.sum(y_p[y_p > 0] * np.log(y_p[y_p > 0])))

    # SU = 2*I(X,y) / (H(X) + H(y)), so the reported pair pins H(X) exactly. Recomputing it from the exported bins closes the loop on the
    # batched histogram pass: if those counts ever stopped matching a per-column bincount, this equality is where it would show.
    checked = 0
    for i, name in enumerate(names):
        if not (np.isfinite(mi[i]) and su[i] > 0.0):
            continue
        counts = np.bincount(np.asarray(bins[name]), minlength=int(nbins[name])).astype(np.float64)
        p = counts / counts.sum()
        h_x_recomputed = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
        h_x_implied = 2.0 * mi[i] / su[i] - h_y
        assert h_x_recomputed == pytest.approx(h_x_implied, abs=1e-9), (
            f"{name}: H(X) recomputed from the exported bins is {h_x_recomputed:.9f}, but the reported SU={su[i]:.9f} and "
            f"MI={mi[i]:.9f} imply {h_x_implied:.9f}"
        )
        checked += 1
    assert checked >= 3, f"only {checked} feature(s) carried a usable SU, so this reconciliation checked almost nothing"


def test_every_reported_su_is_a_valid_similarity():
    """SU is a normalised quantity: outside [0, 1] means the cached MI and the marginal entropies disagree."""
    import pandas as pd

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 1200
    a = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = (a > 0).astype(np.int64)
    MRMR._FIT_CACHE.clear()
    est = MRMR(random_state=0, verbose=0, fe_max_steps=0, full_npermutations=3, baseline_npermutations=2, retain_artifacts=True).fit(X, y)
    su = np.asarray(est.export_artifacts()["su_to_target"], dtype=np.float64)
    assert su.shape[0] == X.shape[1]
    finite = su[np.isfinite(su)]
    assert finite.size, "every reported SU was NaN, so this test is not looking at anything"
    assert np.all((finite >= -1e-9) & (finite <= 1.0 + 1e-9)), f"SU outside [0, 1]: {finite}"
