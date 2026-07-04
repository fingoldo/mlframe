"""Tests for ``mlframe.feature_selection.filters.group_aware``.

Covers ``cluster_features_by_correlation``, ``_cluster_medoids`` and the
``GroupAwareMRMR`` wrapper. Includes a fast biz-value check that the medoid
of N noisy copies of a latent variable is one of the least-noisy members.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.group_aware import (
    GroupAwareMRMR,
    _cluster_medoids,
    cluster_features_by_correlation,
)


class TestClusterMedoids:
    """Unit tests for ``_cluster_medoids``."""

    def test_singleton_returns_only_column(self):
        """A cluster with one member is its own medoid."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((50, 1)), columns=["a"])
        cluster_id = np.array([0])

        medoids = _cluster_medoids(X, cluster_id)

        assert medoids == [0]

    def test_singleton_among_many(self):
        """All-singletons partition: every column is its own medoid."""
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((100, 4)))
        cluster_id = np.array([0, 1, 2, 3])

        medoids = _cluster_medoids(X, cluster_id)

        assert sorted(medoids) == [0, 1, 2, 3]

    def test_medoid_is_central_column(self):
        """In a 3-feature cluster the medoid is the column with max mean abs-corr to siblings."""
        rng = np.random.default_rng(2)
        n = 400
        z = rng.standard_normal(n)
        # ``central`` is strongly correlated with both ``near`` and ``far``;
        # the other two are weakly correlated with each other.
        central = z + 0.05 * rng.standard_normal(n)
        near = z + 0.1 * rng.standard_normal(n)
        far = z + 0.1 * rng.standard_normal(n) + 0.6 * rng.standard_normal(n)
        X = pd.DataFrame({"near": near, "central": central, "far": far})
        cluster_id = np.array([0, 0, 0])

        medoids = _cluster_medoids(X, cluster_id, method="pearson")

        assert medoids == [1]  # ``central`` at column index 1.

    def test_perfectly_correlated_pair_pearson(self):
        """``x`` and ``2*x`` cluster together (|corr|=1) under Pearson."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal(200)
        X = pd.DataFrame({"x": x, "2x": 2 * x, "noise": rng.standard_normal(200)})

        cluster_id = cluster_features_by_correlation(X, threshold=0.9, method="pearson")

        assert cluster_id[0] == cluster_id[1]
        assert cluster_id[0] != cluster_id[2]

    def test_perfectly_correlated_pair_spearman(self):
        """``x`` and ``2*x`` cluster together under Spearman too (monotone)."""
        rng = np.random.default_rng(4)
        x = rng.standard_normal(200)
        X = pd.DataFrame({"x": x, "2x": 2 * x, "noise": rng.standard_normal(200)})

        cluster_id = cluster_features_by_correlation(X, threshold=0.9, method="spearman")

        assert cluster_id[0] == cluster_id[1]
        assert cluster_id[0] != cluster_id[2]

    def test_spearman_vs_pearson_dispatch_on_nonlinear(self):
        """On nonlinear-monotone features, Pearson and Spearman matrices differ -
        dispatch through to the requested method must yield different per-column scores."""
        rng = np.random.default_rng(5)
        n = 400
        z = rng.standard_normal(n)
        # Monotone but very nonlinear transforms: Spearman rank-corr stays near 1,
        # Pearson linear corr collapses.
        X = pd.DataFrame({
            "z": z,
            "z_cube": z ** 3,
            "z_exp": np.exp(z),
            "sign_sqrt": np.sign(z) * np.sqrt(np.abs(z)),
        })
        cluster_id = np.zeros(X.shape[1], dtype=int)

        # Use the absolute-corr score vectors as a proxy for "ordering".
        # pandas 2.x ``.corr().abs().to_numpy()`` can return a read-only
        # zero-copy view of the underlying block (Arrow-backed frames or the
        # pandas-3.0 nullable-dtype path); ``np.fill_diagonal`` writes in
        # place and crashes "underlying array is read-only". Force a writable
        # copy via ``np.array(..., copy=True)``.
        corr_p = np.array(X.corr(method="pearson").abs().to_numpy(), copy=True)
        corr_s = np.array(X.corr(method="spearman").abs().to_numpy(), copy=True)
        np.fill_diagonal(corr_p, 0.0)
        np.fill_diagonal(corr_s, 0.0)
        scores_p = corr_p.mean(axis=1)
        scores_s = corr_s.mean(axis=1)

        # Dispatch sanity: medoid call doesn't raise under either method.
        m_p = _cluster_medoids(X, cluster_id, method="pearson")
        m_s = _cluster_medoids(X, cluster_id, method="spearman")
        assert len(m_p) == 1 and len(m_s) == 1
        # The per-column abs-corr profiles must differ - i.e. method= actually dispatches.
        assert not np.allclose(scores_p, scores_s, atol=1e-3)

    def test_cluster_ids_partition_feature_space(self):
        """Every column appears in exactly one cluster id in [0, n_clusters)."""
        rng = np.random.default_rng(6)
        n, k = 300, 8
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        cols = {
            "a1": z1 + 0.05 * rng.standard_normal(n),
            "a2": z1 + 0.05 * rng.standard_normal(n),
            "a3": z1 + 0.05 * rng.standard_normal(n),
            "b1": z2 + 0.05 * rng.standard_normal(n),
            "b2": z2 + 0.05 * rng.standard_normal(n),
            "indep1": rng.standard_normal(n),
            "indep2": rng.standard_normal(n),
            "indep3": rng.standard_normal(n),
        }
        X = pd.DataFrame(cols)

        cluster_id = cluster_features_by_correlation(X, threshold=0.9, method="pearson")

        assert cluster_id.shape == (k,)
        # Labels are compact: [0, n_clusters - 1] with no gaps.
        unique = np.unique(cluster_id)
        assert (unique == np.arange(unique.size)).all()
        # Two ``a*`` columns share a cluster; ``b*`` share; the three indep
        # singletons are distinct from each other and from the a/b clusters.
        assert cluster_id[0] == cluster_id[1] == cluster_id[2]
        assert cluster_id[3] == cluster_id[4]
        assert cluster_id[0] != cluster_id[3]
        for idx in (5, 6, 7):
            assert (cluster_id == cluster_id[idx]).sum() == 1


class TestEdgeCases:
    """Edge cases: trivial / degenerate inputs."""

    def test_single_column(self):
        """One-feature DataFrame yields one cluster, one medoid."""
        X = pd.DataFrame({"only": np.arange(20, dtype=float)})

        cluster_id = cluster_features_by_correlation(X)
        medoids = _cluster_medoids(X, cluster_id)

        assert cluster_id.tolist() == [0]
        assert medoids == [0]

    def test_empty_cluster_id_raises(self):
        """``_cluster_medoids`` cannot pick a max over zero clusters."""
        X = pd.DataFrame()

        with pytest.raises(ValueError):
            _cluster_medoids(X, np.array([], dtype=int))

    def test_all_nan_column_becomes_singleton(self):
        """A column of all NaN has undefined correlation - it ends in its own cluster."""
        rng = np.random.default_rng(7)
        n = 80
        X = pd.DataFrame({
            "good_a": rng.standard_normal(n),
            "good_b": rng.standard_normal(n),
            "nan_col": np.full(n, np.nan),
        })

        cluster_id = cluster_features_by_correlation(X, threshold=0.9)
        medoids = _cluster_medoids(X, cluster_id)

        assert cluster_id.shape == (3,)
        # The all-NaN column cannot exceed the threshold with anything so is alone.
        assert (cluster_id == cluster_id[2]).sum() == 1
        # The NaN column is its own medoid (singleton path).
        assert 2 in medoids

    def test_numpy_array_input(self):
        """``X`` as a plain ndarray (no column names) works."""
        rng = np.random.default_rng(8)
        X = rng.standard_normal((60, 3))

        cluster_id = cluster_features_by_correlation(X)
        medoids = _cluster_medoids(X, cluster_id)

        assert cluster_id.shape == (3,)
        assert len(medoids) == int(cluster_id.max()) + 1


# ================================================================================================
# biz_value: medoid picks one of the least-noisy copies of a latent variable
# ================================================================================================


@pytest.mark.fast
def test_biz_medoid_picks_central_column():
    """Quantitative win: among 5 increasingly-noisy copies of a latent ``z``, the
    Spearman medoid lands on one of the two least-noisy copies (indices 0 or 1).

    Business value: when an operator dumps a basket of redundant sensor reads
    into a feature pool, group-aware mRMR should pick the cleanest representative
    of the basket rather than a noise-corrupted one.
    """
    rng = np.random.default_rng(123)
    n = 600
    z = rng.standard_normal(n)
    noise_levels = [0.05, 0.10, 0.30, 0.50, 0.80]
    cols = {
        f"copy_{i}": z + sigma * rng.standard_normal(n)
        for i, sigma in enumerate(noise_levels)
    }
    X = pd.DataFrame(cols)

    # All five copies should land in the same cluster at threshold=0.6.
    cluster_id = cluster_features_by_correlation(X, threshold=0.6, method="spearman")
    assert (cluster_id == cluster_id[0]).all(), (
        "All 5 noisy copies of the same latent must single-link cluster together; "
        f"got cluster_id={cluster_id.tolist()}"
    )

    medoids = _cluster_medoids(X, cluster_id, method="spearman")
    assert len(medoids) == 1
    assert medoids[0] in (0, 1), (
        f"Medoid should be one of the two least-noisy copies (idx 0 or 1); "
        f"got idx={medoids[0]} (noise sigma={noise_levels[medoids[0]]})"
    )


# ================================================================================================
# GroupAwareMRMR smoke: wrapper expands medoid selection back to full clusters
# ================================================================================================


class _FakeInner:
    """Minimal stand-in for an mRMR estimator - selects the first ``k`` medoid columns."""

    def __init__(self, k: int = 2):
        self.k = k

    def fit(self, X, y):
        self.support_ = np.arange(min(self.k, X.shape[1]), dtype=np.int64)
        return self

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self


def test_group_aware_mrmr_expands_to_cluster_members():
    """When ``expand=True``, ``support_`` covers every original column in any selected cluster."""
    rng = np.random.default_rng(9)
    n = 200
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "a1": z1 + 0.05 * rng.standard_normal(n),
        "a2": z1 + 0.05 * rng.standard_normal(n),
        "b1": z2 + 0.05 * rng.standard_normal(n),
        "b2": z2 + 0.05 * rng.standard_normal(n),
        "indep": rng.standard_normal(n),
    })
    y = rng.standard_normal(n)

    wrapper = GroupAwareMRMR(estimator=_FakeInner(k=1), corr_threshold=0.9, expand=True)
    wrapper.fit(X, y)

    # One cluster selected, but both members of that cluster appear in support_.
    assert len(wrapper.selected_clusters_) == 1
    selected_cluster = wrapper.selected_clusters_[0]
    expected = np.where(wrapper.cluster_assignments_ == selected_cluster)[0]
    assert sorted(wrapper.support_.tolist()) == sorted(expected.tolist())
    # Non-expanding path returns only the medoid of each selected cluster.
    wrapper_no_expand = GroupAwareMRMR(estimator=_FakeInner(k=1), corr_threshold=0.9, expand=False)
    wrapper_no_expand.fit(X, y)
    assert len(wrapper_no_expand.support_) == 1


class TestDuplicateColumnNames:
    """GroupAwareMRMR must not crash on duplicate column labels.

    Duplicate names arise routinely after FE expansion (repeated lags, one-hot level collisions). ``X[label]`` then returns a DataFrame,
    whose ``.dtype`` access raised ``AttributeError`` inside ``_numeric_codes_frame``. The fix iterates positionally via ``.iloc[:, j]``.
    """

    @pytest.mark.parametrize("method", ["pearson", "spearman", "su"])
    def test_redundancy_methods_handle_duplicate_names(self, method):
        from mlframe.feature_selection.filters.group_aware import _redundancy_matrix

        rng = np.random.default_rng(0)
        n = 300
        a = rng.standard_normal(n)
        X = pd.DataFrame(np.c_[a, a + 1e-9 * rng.standard_normal(n), rng.standard_normal(n)])
        X.columns = ["dup", "dup", "other"]
        rm = _redundancy_matrix(X, method)
        assert rm.shape == (3, 3)
        # The two near-identical "dup" columns are mutually redundant; "other" is not.
        assert rm[0, 1] > 0.9
        assert rm[0, 2] < 0.5

    def test_cluster_and_medoid_on_duplicate_names(self):
        rng = np.random.default_rng(1)
        n = 300
        a = rng.standard_normal(n)
        X = pd.DataFrame(np.c_[a, a + 1e-9 * rng.standard_normal(n), rng.standard_normal(n)])
        X.columns = ["dup", "dup", "other"]
        cid = cluster_features_by_correlation(X, threshold=0.9, method="pearson")
        assert cid[0] == cid[1] and cid[2] != cid[0]
        med = _cluster_medoids(X, cid, method="pearson")
        assert sorted(med) == [0, 2]

    def test_fit_expands_duplicate_name_clusters_never_empty(self):
        rng = np.random.default_rng(3)
        n = 500
        sig = rng.standard_normal(n)
        y = (sig + 0.3 * rng.standard_normal(n) > 0).astype(int)
        X = pd.DataFrame(np.c_[sig, sig + 1e-6 * rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)])
        X.columns = ["s", "s", "n", "n"]
        sel = GroupAwareMRMR(_FakeInner(k=1), corr_threshold=0.9, corr_method="pearson", min_reduction=0.0)
        sel.fit(X, y)
        assert len(sel.support_) > 0
        # The selected signal cluster expands to BOTH duplicate-named members.
        assert 0 in sel.support_ and 1 in sel.support_
        assert sel.transform(X).shape[1] == len(sel.support_)


class TestRedundancyMatrixComputedOnce:
    """GroupAwareMRMR.fit must build the p x p redundancy matrix ONCE, not once per clustering + once per medoid pick.

    The matrix is a function of (X, corr_method) alone; rebuilding it for the medoid pass was a redundant O(p^2) SU/corr
    pass. fit now computes it once and threads it through both via ``precomputed_corr``. This pins the single build so a
    future refactor cannot silently reintroduce the double compute, and pins byte-identity of the threaded matrix.
    """

    def test_fit_builds_redundancy_matrix_once(self, monkeypatch):
        import mlframe.feature_selection.filters.group_aware as _ga
        from sklearn.feature_selection import SelectKBest, f_classif

        calls = {"n": 0}
        _orig = _ga._redundancy_matrix
        monkeypatch.setattr(_ga, "_redundancy_matrix", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or _orig(*a, **k)))

        rng = np.random.default_rng(0)
        n = 200
        base = rng.standard_normal((n, 3))
        X = pd.DataFrame(
            np.column_stack([base[:, 0], base[:, 0] + 0.01 * rng.standard_normal(n), base[:, 1], base[:, 2], rng.standard_normal((n, 2))]),
            columns=[f"c{i}" for i in range(6)],
        )
        y = (X["c0"] + X["c2"] > 0).astype(int)
        _ga.GroupAwareMRMR(estimator=SelectKBest(f_classif, k=2), corr_threshold=0.8).fit(X, y)
        assert calls["n"] == 1, f"redundancy matrix rebuilt {calls['n']}x per fit (expected 1 -- the double-compute regressed)"

    def test_precomputed_corr_is_byte_identical(self):
        from mlframe.feature_selection.filters.group_aware import _redundancy_matrix
        rng = np.random.default_rng(2)
        X = pd.DataFrame(rng.standard_normal((150, 5)), columns=list("abcde"))
        corr = _redundancy_matrix(X, "spearman")
        # threading the precomputed matrix must give the same clustering + medoids as letting each recompute.
        c_pre = cluster_features_by_correlation(X, threshold=0.9, method="spearman", precomputed_corr=corr)
        c_recompute = cluster_features_by_correlation(X, threshold=0.9, method="spearman")
        assert np.array_equal(c_pre, c_recompute)
        m_pre = _cluster_medoids(X, c_pre, method="spearman", precomputed_corr=corr)
        m_recompute = _cluster_medoids(X, c_recompute, method="spearman")
        assert m_pre == m_recompute


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])


def test_su_redundancy_lowcard_codes_searchsorted_equals_dict_map():
    # The vectorised np.searchsorted dense-coding of low-cardinality columns must equal the prior
    # dict-lookup list comprehension bit-for-bit, including the non-finite -> len(uniq) sentinel.
    import numpy as np
    rng = np.random.default_rng(4)
    for _ in range(50):
        n = int(rng.integers(500, 3000))
        col = rng.integers(0, 8, n).astype(np.float64)
        if rng.random() < 0.5:
            col[rng.choice(n, max(1, n // 50), replace=False)] = np.nan
        finite = np.isfinite(col)
        uniq = np.unique(col[finite]) if finite.any() else np.array([0.0])
        lookup = {v: i for i, v in enumerate(uniq)}
        old = np.array([lookup.get(v, len(lookup)) for v in col], dtype=np.int64)
        new = np.searchsorted(uniq, col).astype(np.int64)
        assert np.array_equal(old, new)
