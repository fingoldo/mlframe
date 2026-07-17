"""Regression guards for the stability-selection fixes (audit 2026-06-03):

- hierarchy-stability-9: frequencies must divide by the number of SUCCESSFUL
  subsamples (effective B), not the nominal count, when selector_fn raises on
  some draws -- otherwise every frequency is silently deflated by
  (n_success / nominal) and the Faletto-Bien / Shah-Samworth bounds (which are
  parameterised by the subsample count) no longer hold. The effective and
  failed counts must be surfaced in the returned info dict.
- hierarchy-stability-8: complementary_pairs_stability's two halves must form a
  TRUE partition of the sample (idx[:half] and idx[half:]). The old
  idx[half:2*half] dropped the middle row for odd n, breaking the
  complementary-pair structure the bound assumes.

These functions are standalone (no MRMR import), so the tests are fast.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._stability_cluster import (
    cluster_stability_selection,
    complementary_pairs_stability,
)


def _data(n, p, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


class TestEffectiveBootstrapDenominator:
    def test_cluster_freq_divides_by_successful_runs(self):
        X, y = _data(60, 3)
        calls = {"i": 0}

        def sel(Xs, ys):
            calls["i"] += 1
            if calls["i"] % 2 == 0:  # fail every 2nd bootstrap
                raise ValueError("injected selector failure")
            return np.array([0], dtype=np.int64)  # always pick feature 0

        # corr_threshold > 1 => no clustering edges => each feature its own cluster.
        chosen, feat_freq, info = cluster_stability_selection(
            X,
            y,
            sel,
            n_bootstrap=10,
            pi_threshold=0.5,
            corr_threshold=2.0,
            rng_seed=0,
        )
        assert info["n_effective"] == 5, info
        assert info["n_failed"] == 5, info
        # Feature 0 was selected in EVERY successful run -> frequency 1.0,
        # NOT 0.5 (which the nominal-denominator bug would have produced).
        assert abs(float(feat_freq[0]) - 1.0) < 1e-9, feat_freq.tolist()

    def test_complementary_freq_divides_by_successful_pairs(self):
        X, y = _data(60, 3)
        calls = {"i": 0}

        def sel(Xs, ys):
            # Raise on the second half of the first 5 pairs (each pair calls sel
            # twice: sel_b then sel_bc). Failing sel_bc keeps two calls/pair, so
            # exactly 5 pairs fail and 5 succeed -- but we assert the robust
            # invariant below rather than the exact split, since the key proof
            # is that the divisor is n_effective (freq[0]==1.0), not n_pairs.
            calls["i"] += 1
            if calls["i"] <= 10 and calls["i"] % 2 == 0:
                raise ValueError("injected selector failure")
            return np.array([0], dtype=np.int64)

        chosen, comp_freq, info = complementary_pairs_stability(
            X,
            y,
            sel,
            n_pairs=10,
            pi_threshold=0.5,
            rng_seed=0,
        )
        # Robust invariants (independent of the exact failure split):
        assert info["n_effective"] + info["n_failed"] == 10, info
        assert info["n_failed"] >= 1, info
        # Feature 0 is complementary-selected in EVERY successful pair, so its
        # frequency is 1.0 IFF the divisor is n_effective. With the nominal-B
        # bug it would be n_effective/10 < 1.0.
        assert abs(float(comp_freq[0]) - 1.0) < 1e-9, (comp_freq.tolist(), info)


class TestComplementaryPairsTruePartition:
    def test_odd_n_halves_partition_the_sample(self):
        n, p = 7, 4  # odd n: the middle row must NOT be dropped
        X, y = _data(n, p)
        sizes = []

        def sel(Xs, ys):
            sizes.append(int(Xs.shape[0]))
            return np.array([0], dtype=np.int64)

        complementary_pairs_stability(X, y, sel, n_pairs=1, rng_seed=0)
        assert len(sizes) == 2, sizes
        # True partition: the two halves cover ALL n rows (3 + 4 = 7).
        # The pre-fix idx[half:2*half] gave 3 + 3 = 6, dropping the middle row.
        assert sizes[0] + sizes[1] == n, f"complementary halves must partition all {n} rows; got sizes={sizes}"
