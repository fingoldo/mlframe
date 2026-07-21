"""Regression test for MEM2: conditional permutation importance allocates one working buffer,
mutate-restored across (j x repeat), instead of a fresh full-matrix copy each iteration."""

import numpy as np

from mlframe.feature_selection.wrappers._helpers_importance import _conditional_permutation_importance


class _AllColModel:
    """Score depends on every column so conditional permutation importance is non-degenerate.
    Also records id() of each X it scores to detect whether a fresh buffer is allocated per repeat."""

    def __init__(self):
        self.w = None
        self.seen_ids = []

    def fit(self, X, y):
        """Performs 2 setup steps, then returns self unchanged."""
        X = np.asarray(X)
        # Least-squares weights so score reflects each feature's contribution.
        self.w, *_ = np.linalg.lstsq(X, np.asarray(y), rcond=None)
        return self

    def score(self, X, y):
        """Test spy/callback: records each invocation (Xa = np.asarray(X); self.seen_ids.append(id(X)); pred = Xa @ self.w)."""
        Xa = np.asarray(X)
        self.seen_ids.append(id(X))
        pred = Xa @ self.w
        ss_res = float(np.sum((np.asarray(y) - pred) ** 2))
        ss_tot = float(np.sum((np.asarray(y) - np.mean(y)) ** 2)) or 1.0
        return 1.0 - ss_res / ss_tot


def _make_data():
    """Make data."""
    rng = np.random.default_rng(0)
    n, p = 200, 4
    X = rng.random((n, p))
    # Feature 1 is the dominant signal; 0 and 1 mildly correlated to exercise the conditioning tree.
    X[:, 0] = 0.5 * X[:, 1] + 0.5 * X[:, 0]
    y = X[:, 1] * 3.0 + X[:, 2] * 0.5 + rng.normal(0, 0.05, n)
    return X, y, n, p


def test_single_working_buffer_reused_across_repeats():
    """With copy-once-before-loop, every score() on the ndarray path receives the SAME buffer
    object; the pre-fix per-repeat ``X_arr.copy()`` produced a distinct id each call."""
    X, y, _n, p = _make_data()
    model = _AllColModel().fit(X, y)
    imp = _conditional_permutation_importance(model, X, y, n_repeats=3, random_state=0)

    assert imp.shape == (p,)
    # score() is called once on the original X (baseline) plus once per (j x repeat) on the
    # working buffer. Post-fix that working buffer is a SINGLE reused object => exactly 2
    # distinct ids (baseline X + the one reused X_perm). Pre-fix each repeat allocated a fresh
    # ``X_arr.copy()`` that lived simultaneously => 1 + n_repeats*p distinct ids.
    distinct = len(set(model.seen_ids))
    assert distinct == 2, f"expected baseline + one reused working buffer (2 ids), saw {distinct}"
    # Guard the regression direction explicitly: more buffers than baseline+1 means per-iteration copy.
    assert distinct < 1 + 3 * p


def test_importances_signal_feature_dominates_and_finite():
    """Semantics preserved by the mutate-restore: the dominant signal feature scores highest."""
    X, y, _n, _p = _make_data()
    model = _AllColModel().fit(X, y)
    imp = _conditional_permutation_importance(model, X, y, n_repeats=4, random_state=7)
    assert np.all(np.isfinite(imp))
    assert int(np.argmax(imp)) == 1


def test_buffer_restored_clean_between_features():
    """After the run the algorithm must not have leaked permutations: a second identical run
    yields the same importances (buffer was restored, not progressively corrupted)."""
    X, y, _n, _p = _make_data()
    m1 = _AllColModel().fit(X, y)
    m2 = _AllColModel().fit(X, y)
    imp1 = _conditional_permutation_importance(m1, X, y, n_repeats=3, random_state=11)
    imp2 = _conditional_permutation_importance(m2, X, y, n_repeats=3, random_state=11)
    np.testing.assert_allclose(imp1, imp2)
