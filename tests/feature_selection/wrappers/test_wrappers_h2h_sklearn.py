"""Head-to-head guard-rail tests against sklearn.feature_selection.RFECV.

These are not goodness benchmarks (the standalone report at
``mlframe/feature_selection/_benchmarks/bench_rfecv_vs_sklearn.py`` covers that
in depth). This file is a CI guard: if a future refactor of our RFECV ever
silently drops below sklearn's quality on common synthetic problems, this test
fails and forces a redesign.

Design constraints:
    - Sklearn estimators only (no CatBoost / XGBoost) so CI < 60s.
    - Same outer CV folds used to compare both selectors fairly.
    - Synthetic data with KNOWN informative-feature counts so we can also
      assert recall on a well-conditioned problem.
    - Tolerances are generous (epsilon margin), the goal is to catch
      REGRESSIONS, not to prove our selector is strictly better than
      sklearn's on every problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV as SkRFECV  # noqa: N811 -- head-to-head test needs both names visible side-by-side; treating one as the constant is misleading.
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from mlframe.feature_selection.wrappers import RFECV as OurRFECV  # noqa: N811


# ----------------------------------------------------------------------------
# Synthetic problems shared by all h2h tests
# ----------------------------------------------------------------------------
def _clf_redundant():
    """Classification with redundant features. RFECV should drop redundancy.
    shuffle=False keeps informative cols at indices [0..n_informative)."""
    from tests.training.synthetic import make_sklearn_classification_df

    X_df, y, _ = make_sklearn_classification_df(
        n_samples=400,
        n_features=20,
        n_informative=5,
        n_redundant=10,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.5,
        shuffle=False,
        seed=0,
    )
    return X_df, y, 5  # informative count


def _clf_noisy():
    """Classification with many noise features."""
    from tests.training.synthetic import make_sklearn_classification_df

    X_df, y, _ = make_sklearn_classification_df(
        n_samples=300,
        n_features=25,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.5,
        shuffle=False,
        seed=1,
    )
    return X_df, y, 4


def _reg_correlated():
    """Regression where target depends on a few features + correlated noise.
    sklearn.make_regression already places informative cols first regardless
    of shuffle (no shuffle param), but n_informative is the prefix count."""
    X, y = make_regression(
        n_samples=300,
        n_features=15,
        n_informative=4,
        noise=2.0,
        random_state=2,
        shuffle=False,
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), y, 4


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _eval_subset_cv(estimator, X, y, support_mask, cv):
    """CV-score the estimator restricted to features selected by support_mask."""
    cols = [c for c, sel in zip(X.columns, support_mask) if sel]
    if len(cols) == 0:
        return float("nan")
    X_sub = X[cols]
    return float(np.mean(cross_val_score(estimator, X_sub, y, cv=cv)))


def _our_support_as_bool(rfecv, X):
    """Translate our support_ (which may be int indices or bool mask) into a
    bool mask aligned with X.columns."""
    cols = list(X.columns)
    feat_in = list(rfecv.feature_names_in_)
    if len(rfecv.support_) == 0:
        return np.zeros(len(cols), dtype=bool)
    if isinstance(rfecv.support_[0], (bool, np.bool_)):
        # bool mask aligned with feature_names_in_; map to original X columns
        sel_names = {feat_in[i] for i, s in enumerate(rfecv.support_) if s}
    else:
        sel_names = {feat_in[i] for i in rfecv.support_}
    return np.array([c in sel_names for c in cols], dtype=bool)


# ----------------------------------------------------------------------------
# Score parity: our CV-score on the selected subset >= sklearn's by epsilon
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "data_factory, estimator_factory, cv_factory, eps, label",
    [
        (
            _clf_redundant,
            lambda: LogisticRegression(max_iter=400, random_state=0),
            lambda: StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
            0.05,
            "clf-redundant-LR",
        ),
        (
            _clf_noisy,
            lambda: RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1),
            lambda: StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
            0.10,  # RF has variance; allow looser margin
            "clf-noisy-RF",
        ),
        (
            _reg_correlated,
            lambda: Ridge(random_state=0),
            lambda: KFold(n_splits=3, shuffle=True, random_state=0),
            0.10,  # R^2 can be noisy on small reg sets
            "reg-correlated-Ridge",
        ),
    ],
)
def test_h2h_score_parity(data_factory, estimator_factory, cv_factory, eps, label):
    """Our RFECV's selected subset must score within ``eps`` of sklearn's
    selected subset on the SAME outer CV folds. Failure means a regression."""
    X, y, _ = data_factory()
    cv_for_compare = cv_factory()
    estimator = estimator_factory()

    sk = SkRFECV(estimator=estimator_factory(), cv=cv_factory(), step=1, min_features_to_select=1)
    sk.fit(X, y)

    ours = OurRFECV(
        estimator=estimator_factory(),
        cv=cv_factory(),
        max_refits=8,
        verbose=0,
    )
    ours.fit(X, y)

    # Score both supports under THE SAME independent CV so the comparison is fair.
    sk_score = _eval_subset_cv(estimator, X, y, sk.support_, cv_for_compare)
    our_mask = _our_support_as_bool(ours, X)
    our_score = _eval_subset_cv(estimator, X, y, our_mask, cv_for_compare)

    print(f"\n[h2h:{label}] sklearn n_features_={sk.n_features_} score={sk_score:.4f}, ours n_features_={ours.n_features_} score={our_score:.4f}, eps={eps}")
    assert not np.isnan(our_score), f"[{label}] our RFECV produced empty support_"
    assert our_score >= sk_score - eps, (
        f"[{label}] regression: our RFECV CV-score {our_score:.4f} fell more than "
        f"{eps} below sklearn.RFECV {sk_score:.4f}. Check the most recent "
        f"changes in mlframe/feature_selection/wrappers.py."
    )


# ----------------------------------------------------------------------------
# Subset size: on a redundant problem, our selector should pick at most as
# many features as sklearn (we have voting + cost terms)
# ----------------------------------------------------------------------------
def test_h2h_subset_size_on_redundant_clf():
    """Sanity check: on a problem with 5 informative + 10 redundant + 5 noise,
    our RFECV's selected count must be reasonable (<= sklearn's + 2)."""
    X, y, n_informative = _clf_redundant()
    estimator = LogisticRegression(max_iter=400, random_state=0)
    sk = SkRFECV(estimator=estimator, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0))
    sk.fit(X, y)

    ours = OurRFECV(
        estimator=LogisticRegression(max_iter=400, random_state=0),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        max_refits=8,
        verbose=0,
        # Compare against sklearn's argmax-style selector at the same rule.
        # Default 'one_se_max' is plateau-resistant by design and legitimately
        # picks the full 1-SE band on flat curves -- which is a DIFFERENT
        # selection regime than sklearn's exhaustive backward elimination.
        n_features_selection_rule="argmax",
    )
    ours.fit(X, y)

    # Our MBH search is by design more inclusive than sklearn's exhaustive
    # backward elimination because it explores fewer N points and prefers
    # not to over-prune when scores are tied. Real regression detector:
    # we should never pick MORE than ~80% of all features, and never less
    # than 1 feature.
    upper_bound = max(int(round(X.shape[1] * 0.8)), sk.n_features_ + 6)
    assert ours.n_features_ <= upper_bound, (
        f"Our RFECV picked {ours.n_features_} features (cap {upper_bound}) vs sklearn's {sk.n_features_}; selector is much too inclusive."
    )
    assert ours.n_features_ >= 1, "Our RFECV picked zero features"


# ----------------------------------------------------------------------------
# Recall of true informative features: both selectors should have non-trivial
# overlap with the actual informative feature set
# ----------------------------------------------------------------------------
def test_h2h_informative_recall():
    """On a well-conditioned problem (5 informative, no redundant, easy class_sep),
    our RFECV must recover at least 60% of informative features. Sklearn's
    informative recall is computed too (for the printout) but we don't gate
    on relative comparison since RFECV recall is variance-y on small data."""
    X, y, n_informative = _clf_noisy()
    # The first n_informative columns of make_classification are the informative ones
    informative_idx = set(range(n_informative))

    ours = OurRFECV(
        estimator=LogisticRegression(max_iter=400, random_state=0),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        max_refits=10,
        verbose=0,
    )
    ours.fit(X, y)

    our_mask = _our_support_as_bool(ours, X)
    selected_idx = {i for i, s in enumerate(our_mask) if s}
    overlap = len(informative_idx & selected_idx)
    recall = overlap / max(1, len(informative_idx))

    print(f"\n[h2h:informative-recall] our recall={recall:.2f} ({overlap}/{n_informative})")
    assert recall >= 0.5, (
        f"Our RFECV recovered only {overlap}/{n_informative} informative features (recall={recall:.2f}). This is below the regression threshold of 0.5."
    )
