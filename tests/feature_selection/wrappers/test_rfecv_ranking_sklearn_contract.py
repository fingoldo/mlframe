"""Regression: `RFECV.ranking_` must satisfy sklearn's contract, not hand out the consensus NAME list.

The class advertises the sklearn `RFECV` surface, and `ranking_ == 1` is how a caller asks which features
survived. It used to be assigned straight from `_rank_features_by_importance`, which returns an ordered list
of feature NAMES -- so `np.asarray(sel.ranking_, dtype=float)` raised `could not convert string to float:
'f453'`, and any caller following the documented idiom got nonsense instead of an error. The name order is
still load-bearing internally (`support_` is derived from it by membership), so it stays available under
`consensus_ranking_`.

Found by the Phase 0 benchmark harness: 8 of 20 seeds on the `arcene` bed died on exactly that cast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers import RFECV


@pytest.fixture(scope="module")
def fitted_rfecv() -> RFECV:
    """A small fitted RFECV: enough columns for the vote-based path, small enough to stay quick."""
    rng = np.random.default_rng(0)
    n, p = 300, 24
    x = rng.normal(size=(n, p))
    y = (x[:, 0] + 0.8 * x[:, 1] - 0.6 * x[:, 2] + 0.4 * rng.normal(size=n) > 0).astype(np.int64)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(p)])
    selector = RFECV(LogisticRegression(max_iter=300), cv=3, verbose=0)
    selector.fit(frame, pd.Series(y))
    return selector


def test_ranking_is_a_numeric_vector_aligned_to_the_features(fitted_rfecv: RFECV) -> None:
    """The pre-fix value was a list of name strings, so this cast is the exact failure that was shipping."""
    ranking = getattr(fitted_rfecv, "ranking_", None)
    assert ranking is not None, "ranking_ must be set after fit; sklearn callers rely on it existing"
    arr = np.asarray(ranking)
    assert np.issubdtype(arr.dtype, np.number), f"ranking_ must be numeric, got dtype {arr.dtype}"
    assert arr.shape == (len(fitted_rfecv.feature_names_in_),), f"ranking_ must align to the features, got {arr.shape}"
    np.asarray(ranking, dtype=np.float64)


def test_rank_one_selects_exactly_the_supported_features(fitted_rfecv: RFECV) -> None:
    """`ranking_ == 1` is the canonical sklearn idiom and must agree with `support_` exactly."""
    by_rank = np.asarray(fitted_rfecv.ranking_) == 1
    np.testing.assert_array_equal(by_rank, np.asarray(fitted_rfecv.support_, dtype=bool))


def test_dropped_features_rank_from_two_upward(fitted_rfecv: RFECV) -> None:
    """Rank 1 is reserved for survivors; anything dropped must be strictly worse, never tied at 1."""
    ranking = np.asarray(fitted_rfecv.ranking_)
    dropped = ranking[~np.asarray(fitted_rfecv.support_, dtype=bool)]
    if dropped.size:
        assert dropped.min() >= 2, "a dropped feature must not share rank 1 with the survivors"


def test_consensus_ranking_keeps_the_name_order(fitted_rfecv: RFECV) -> None:
    """The native name order is still reachable, under a name that says what it is."""
    consensus = getattr(fitted_rfecv, "consensus_ranking_", None)
    if consensus is None:
        pytest.skip("this fit took a branch that does not build a consensus ranking")
    assert isinstance(consensus, list)
    known = set(str(f) for f in fitted_rfecv.feature_names_in_)
    assert set(str(c) for c in consensus) <= known, "consensus_ranking_ must only name input features"
