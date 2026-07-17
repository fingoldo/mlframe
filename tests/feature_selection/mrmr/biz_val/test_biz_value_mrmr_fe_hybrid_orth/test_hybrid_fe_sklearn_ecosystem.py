"""Consolidated from test_biz_value_mrmr_layer28.py.

Layer 28 biz_value: PUSH HYBRID FE THROUGH SKLEARN ECOSYSTEM EDGE CASES.

Layers 21-23 pinned standalone + auto-wired hybrid orthogonal-polynomial
FE; Layer 14 pinned MRMR's generic sklearn-integration surface (clone /
pickle / Pipeline / GridSearchCV / set_output / fit_transform / CV
support stability) WITHOUT hybrid FE engaged. Layer 28 is the
intersection: every Layer-14 contract must continue to hold when the
``fe_hybrid_orth_*`` knobs are turned on. Modal production tabular
pipelines combine MRMR (with FE) inside ``Pipeline`` + ``GridSearchCV``
+ ``StandardScaler`` + downstream estimator, and the FE flip must NOT
break that surface.

Contracts pinned
----------------
A. **GridSearchCV over FE hyperparameters**
   ``GridSearchCV(MRMR, {'fe_hybrid_orth_enable': [False, True],
   'fe_hybrid_orth_basis': ['hermite', 'chebyshev']}, cv=3).fit(X, y)``
   on a quadratic-signal target completes WITHOUT crashing on any
   candidate, ``cv_results_`` has 2x2=4 rows, and the search surfaces a
   config that SOLVES the quadratic (best CV >= 0.95) far above the legacy
   linear baseline. (The winner is NOT pinned on
   ``fe_hybrid_orth_enable=True``: the univariate He_2 recovery is
   default-on via ``fe_univariate_basis_enable``, so with
   ``fe_hybrid_orth_pair_enable=False`` the master switch is inert and the
   score is driven by the BASIS axis -- see the test's docstring.)

B. **Pipeline with non-trivial downstream**
   ``Pipeline([(MRMR(fe_hybrid_orth_enable=True), StandardScaler(),
   LogisticRegression())])`` end-to-end on a quadratic-signal binary
   target: predict accuracy on held-out data clears 0.85. FE columns
   must flow through scaler + classifier without dtype / shape errors.

C. **Cross-validation with hybrid FE**
   ``cross_val_score`` on the same Pipeline returns 5 finite scores;
   each fold's MRMR fit produces a non-empty ``hybrid_orth_features_``;
   the recipe must be fit fresh per fold (no y-leakage from holdout
   into fit-time scoring).

D. **Deep clone**
   ``clone(clone(clone(m)))`` preserves all hybrid params bit-exactly
   and produces an unfitted estimator. ``get_params`` round-trips
   identically across three clone hops.

E. **set_output("pandas") with hybrid FE**
   With ``fe_hybrid_orth_enable=True``, ``transform(X)`` returns a
   DataFrame whose columns include the engineered names; the column
   list matches ``get_feature_names_out()``; engineered names are
   reachable downstream.

F. **Regression target with hybrid + scaler**
   Same Pipeline shape but with continuous y and ``LinearRegression``:
   holdout R^2 on a quadratic regression target clears 0.85.

NEVER xfail. If sklearn integration has rough edges, fix them on the
spot.
"""

from __future__ import annotations

import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


SEEDS = (1, 13, 42)


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr


def _build_quadratic_binary(seed: int, n: int = 1500):
    """``y = sign(x1^2 - 1)`` binary target. He_2(x1) is the signal.

    Raw linear LogReg cannot solve it (E[y | x1] is symmetric in x1, so
    any linear logit beats only by chance). With hybrid FE the He_2
    column is engineered and a linear classifier on engineered + raw
    columns solves the task.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
            "noise_c": rng.standard_normal(n),
        }
    )
    y = ((x1 * x1 - 1.0) + 0.05 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_quadratic_regression(seed: int, n: int = 1500):
    """Continuous ``y = x1^2 - 1 + small noise``. Linear regression on
    raw x1 cannot solve it (cov(x1, x1^2) = 0 for standard normal x1);
    with engineered He_2(x1) a linear regressor recovers the signal.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
            "noise_c": rng.standard_normal(n),
        }
    )
    y_arr = (x1 * x1 - 1.0) + 0.1 * rng.standard_normal(n)
    return X, pd.Series(y_arr, name="y")


# ---------------------------------------------------------------------------
# A. GridSearchCV over FE hyperparameters
# ---------------------------------------------------------------------------


class TestGridSearchCVOverFEParams:
    """GridSearchCV over fe_hybrid_orth_* hyperparameters completes cleanly and surfaces a solving config."""

    def test_gridsearch_4_candidates_complete(self):
        """2x2 grid over (fe_hybrid_orth_enable, fe_hybrid_orth_basis)
        completes without crashing on any candidate; cv_results_ has 4
        rows. Pipeline = MRMR -> LogReg (no scaler needed for the grid
        contract: this scenario is about the search itself, not the
        downstream accuracy).
        """
        X, y = _build_quadratic_binary(seed=1, n=1200)
        pipe = Pipeline(
            [
                (
                    "mrmr",
                    _make_mrmr(
                        fe_hybrid_orth_enable=True,
                        fe_hybrid_orth_pair_enable=False,
                        fe_hybrid_orth_top_k=3,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=300)),
            ]
        )
        grid = GridSearchCV(
            pipe,
            param_grid={
                "mrmr__fe_hybrid_orth_enable": [False, True],
                "mrmr__fe_hybrid_orth_basis": ["hermite", "chebyshev"],
            },
            cv=3,
            n_jobs=1,
            refit=True,
        )
        grid.fit(X, y)
        # 2 x 2 = 4 candidates evaluated.
        assert len(grid.cv_results_["params"]) == 4, (
            f"expected 4 grid candidates (2 x 2), got {len(grid.cv_results_['params'])}; params={grid.cv_results_['params']}"
        )
        # Every candidate produced a finite mean_test_score.
        assert np.all(np.isfinite(grid.cv_results_["mean_test_score"])), f"non-finite mean_test_score among candidates: {grid.cv_results_['mean_test_score']}"

    def test_gridsearch_finds_config_that_solves_quadratic(self):
        """GridSearch over the FE knobs MUST surface a config that solves the
        quadratic via an engineered basis column, far above the legacy linear
        baseline (raw LogReg on x1/x2 is symmetric in x1, so it only guesses).

        Contract note (why this is NOT pinned on ``fe_hybrid_orth_enable=True``):
        the univariate orthogonal-basis FE that recovers the He_2 column is
        DEFAULT-ON (``fe_univariate_basis_enable=True``, landed 2026-06-02),
        independent of the heavy ``fe_hybrid_orth_enable`` master switch. That
        master switch additionally gates ONLY the pair-CROSS-basis stage
        (``_h_pair_enable = fe_hybrid_orth_pair_enable AND fe_hybrid_orth_enable``,
        ``_mrmr_fit_impl.py``). This pipeline sets ``fe_hybrid_orth_pair_enable=
        False``, so the master switch toggles nothing here: enable=False and
        enable=True produce BIT-IDENTICAL CV scores (measured 0.9827==0.9827 for
        hermite, 0.9087==0.9087 for chebyshev). The grid's score variance is
        driven entirely by the BASIS axis -- hermite's degree-2 column is exactly
        x1**2-1 (corr 1.00 with x1**2 -> CV 0.983) while chebyshev's saturated
        ``p2cos1`` is a weaker quadratic proxy (corr -0.93 -> CV 0.909). Pinning
        the winner to enable=True was the stale premise; the honest contract is
        that SOME engineered config clears the legacy baseline by a wide margin.
        """
        X, y = _build_quadratic_binary(seed=13, n=1500)
        pipe = Pipeline(
            [
                (
                    "mrmr",
                    _make_mrmr(
                        fe_hybrid_orth_enable=True,
                        fe_hybrid_orth_pair_enable=False,
                        fe_hybrid_orth_top_k=3,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=300)),
            ]
        )
        grid = GridSearchCV(
            pipe,
            param_grid={
                "mrmr__fe_hybrid_orth_enable": [False, True],
                "mrmr__fe_hybrid_orth_basis": ["hermite", "chebyshev"],
            },
            cv=3,
            n_jobs=1,
            refit=True,
        )
        grid.fit(X, y)
        # Legacy linear baseline: univariate-basis FE OFF -> only raw x1/x2 are
        # available, and E[y|x1] is symmetric in x1, so a linear logit cannot
        # separate the classes (measured CV ~0.5-0.7, train acc 0.695).
        legacy = _make_mrmr(
            fe_univariate_basis_enable=False,
            fe_univariate_fourier_enable=False,
            fe_hybrid_orth_enable=False,
        )
        legacy_pipe = Pipeline([("mrmr", legacy), ("clf", LogisticRegression(max_iter=300))])
        legacy_score = float(cross_val_score(legacy_pipe, X, y, cv=3).mean())
        best_score = float(grid.best_score_)
        # The engineered path solves the quadratic; measured best 0.983.
        assert best_score >= 0.95, (
            f"GridSearch should find a config that solves the quadratic via an "
            f"engineered basis column (measured best ~0.983). Got "
            f"best_score={best_score:.4f}, best_params={grid.best_params_}, "
            f"mean_test_scores={grid.cv_results_['mean_test_score']}"
        )
        # And it must crush the legacy linear baseline (no symmetric-in-x1 fix).
        assert best_score >= legacy_score + 0.15, (
            f"Engineered FE must clear the legacy linear baseline by a wide margin on the quadratic; best={best_score:.4f}, legacy={legacy_score:.4f}"
        )

    def test_gridsearch_best_estimator_serializes(self):
        """Refit + pickle of the best estimator round-trips cleanly.
        Production model-registry path: GridSearch -> pickle the best.
        """
        import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

        X, y = _build_quadratic_binary(seed=42, n=1200)
        pipe = Pipeline(
            [
                (
                    "mrmr",
                    _make_mrmr(
                        fe_hybrid_orth_enable=True,
                        fe_hybrid_orth_pair_enable=False,
                        fe_hybrid_orth_top_k=3,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=300)),
            ]
        )
        grid = GridSearchCV(
            pipe,
            param_grid={
                "mrmr__fe_hybrid_orth_enable": [False, True],
                "mrmr__fe_hybrid_orth_basis": ["hermite", "chebyshev"],
            },
            cv=3,
            n_jobs=1,
            refit=True,
        )
        grid.fit(X, y)
        blob = pickle.dumps(grid.best_estimator_)
        reloaded = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        # Predict on the training data with both originals and reloaded:
        # exact match.
        preds_orig = grid.best_estimator_.predict(X)
        preds_reloaded = reloaded.predict(X)
        np.testing.assert_array_equal(preds_orig, preds_reloaded)


# ---------------------------------------------------------------------------
# B. Pipeline with non-trivial downstream (MRMR + StandardScaler + LogReg)
# ---------------------------------------------------------------------------


class TestPipelineWithScalerAndClassifier:
    """MRMR + StandardScaler + LogisticRegression with hybrid FE solves a quadratic-signal target end-to-end."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_pipeline_predict_quadratic_accuracy(self, seed):
        """The full Pipeline clears 0.85 holdout accuracy on the quadratic-signal binary target."""
        X, y = _build_quadratic_binary(seed=seed, n=2000)
        n_train = 1500
        Xtr, ytr = X.iloc[:n_train], y.iloc[:n_train]
        Xte, yte = X.iloc[n_train:], y.iloc[n_train:]
        pipe = Pipeline(
            [
                (
                    "mrmr",
                    _make_mrmr(
                        fe_hybrid_orth_enable=True,
                        fe_hybrid_orth_pair_enable=False,
                        fe_hybrid_orth_basis="hermite",
                        fe_hybrid_orth_top_k=5,
                    ),
                ),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        )
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        acc = (preds == yte.to_numpy()).mean()
        assert acc >= 0.85, (
            f"seed={seed}: Pipeline test accuracy {acc:.3f} below 0.85 "
            f"on a quadratic-signal target with hybrid FE engaged. The "
            f"engineered He_2 column should let a linear classifier "
            f"solve it; verify MRMR -> scaler -> LogReg data flow."
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_pipeline_predict_proba_finite(self, seed):
        """predict_proba on held-out data returns finite, valid
        probabilities (no NaN / inf from FE replay).
        """
        X, y = _build_quadratic_binary(seed=seed, n=1800)
        n_train = 1300
        Xtr, ytr = X.iloc[:n_train], y.iloc[:n_train]
        Xte = X.iloc[n_train:]
        pipe = Pipeline(
            [
                (
                    "mrmr",
                    _make_mrmr(
                        fe_hybrid_orth_enable=True,
                        fe_hybrid_orth_pair_enable=False,
                        fe_hybrid_orth_top_k=5,
                    ),
                ),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        )
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)
        assert proba.shape == (Xte.shape[0], 2), proba.shape
        assert np.all(np.isfinite(proba)), f"seed={seed}: non-finite probabilities in predict_proba: {proba[~np.isfinite(proba)][:5]}"
        # Rows sum to 1.
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# C. Cross-validation with hybrid FE: no leakage, finite scores
# ---------------------------------------------------------------------------


class TestCrossValWithHybridFE:
    """Cross-validation with hybrid FE returns finite scores and fits fresh recipes per fold without y-leakage."""

    def test_cross_val_score_5_finite_scores(self):
        """cross_val_score returns 5 finite scores with mean accuracy >= 0.80 on the quadratic-signal target."""
        X, y = _build_quadratic_binary(seed=1, n=1800)
        pipe = Pipeline(
            [
                (
                    "mrmr",
                    _make_mrmr(
                        fe_hybrid_orth_enable=True,
                        fe_hybrid_orth_pair_enable=False,
                        fe_hybrid_orth_top_k=5,
                    ),
                ),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        )
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, n_jobs=1)
        assert scores.shape == (5,), scores.shape
        assert np.all(np.isfinite(scores)), f"non-finite CV scores: {scores}"
        # The pipeline solves the quadratic task on every fold -- mean
        # accuracy must be solid even if some folds are noisier.
        assert scores.mean() >= 0.80, f"cross_val_score mean {scores.mean():.3f} below 0.80 on quadratic signal with hybrid FE; per-fold: {scores}"

    def test_each_fold_fits_hybrid_recipes_fresh(self):
        """Manual 5-fold loop: every fold's MRMR fit produces a non-
        empty hybrid_orth_features_. Confirms the recipe builder runs
        ON THE TRAINING SLICE of every fold (no shortcut that re-uses
        a leaky fit from outside the fold). y-leakage from the
        holdout slice into the recipe is structurally impossible
        because recipes are pure functions of X, but we still pin
        that the FE pipeline engaged on each fold's training data.
        """
        X, y = _build_quadratic_binary(seed=7, n=1800)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold_idx, (tr_idx, _te_idx) in enumerate(kf.split(X)):
            m = _make_mrmr(
                fe_hybrid_orth_enable=True,
                fe_hybrid_orth_pair_enable=False,
                fe_hybrid_orth_top_k=5,
            )
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            assert len(m.hybrid_orth_features_) > 0, (
                f"fold={fold_idx}: hybrid_orth_features_ empty on a quadratic signal; FE pipeline failed to engage on this fold's training slice"
            )

    def test_no_y_leakage_across_folds(self):
        """Permuting the target inside the holdout slice MUST NOT
        change the recipe built on the training slice. Per Layer 23
        we know transform() is y-independent; here we extend the
        contract to fit time: refit MRMR on the same training slice
        AFTER scrambling y outside the training slice and confirm
        identical hybrid_orth_features_. (Trivial structurally - MRMR
        fits on (Xtr, ytr) only - but pins the contract surface.)
        """
        X, y = _build_quadratic_binary(seed=21, n=1500)
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        tr_idx, te_idx = next(kf.split(X))
        Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
        m1 = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_top_k=5,
        )
        m1.fit(Xtr, ytr)
        feats_baseline = list(m1.hybrid_orth_features_)

        # Permute holdout y, refit on the unchanged training slice.
        rng = np.random.default_rng(123)
        y_holdout_scrambled = y.copy()
        y_holdout_scrambled.iloc[te_idx] = rng.permutation(y_holdout_scrambled.iloc[te_idx].to_numpy())
        m2 = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_top_k=5,
        )
        m2.fit(Xtr, y_holdout_scrambled.iloc[tr_idx])
        # Training slice y is unchanged so the recipe must match.
        assert list(m2.hybrid_orth_features_) == feats_baseline, (
            f"holdout-y scramble leaked into training fit: baseline={feats_baseline} vs scrambled-holdout={m2.hybrid_orth_features_}"
        )


# ---------------------------------------------------------------------------
# D. Deep clone preserves params + remains unfitted
# ---------------------------------------------------------------------------


class TestDeepClonePreserves:
    """clone(clone(clone(m))) preserves all hybrid params bit-exactly and stays unfitted."""

    def test_three_deep_clone_preserves_params_and_unfitted(self):
        """Three clone hops preserve get_params() identically and produce an unfitted estimator."""
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_degrees=(2, 4),
            fe_hybrid_orth_basis="legendre",
            fe_hybrid_orth_top_k=7,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_pair_max_degree=3,
        )
        m1 = clone(m)
        m2 = clone(m1)
        m3 = clone(m2)
        # All four objects share the same constructor params.
        assert m.get_params(deep=True) == m1.get_params(deep=True)
        assert m1.get_params(deep=True) == m2.get_params(deep=True)
        assert m2.get_params(deep=True) == m3.get_params(deep=True)
        # The deepest clone is unfitted.
        assert not hasattr(m3, "support_")
        assert not hasattr(m3, "hybrid_orth_features_")
        assert not hasattr(m3, "feature_names_in_")
        # Specific hybrid params survived.
        assert m3.fe_hybrid_orth_enable is True
        assert tuple(m3.fe_hybrid_orth_degrees) == (2, 4)
        assert m3.fe_hybrid_orth_basis == "legendre"
        assert m3.fe_hybrid_orth_top_k == 7
        assert m3.fe_hybrid_orth_pair_enable is False
        assert m3.fe_hybrid_orth_pair_max_degree == 3

    def test_three_deep_clone_after_fit_still_unfitted(self):
        """Even after fitting the source, the deeply cloned descendants
        remain unfitted. Layer 14's single-clone contract extended.
        """
        X, y = _build_quadratic_binary(seed=1, n=800)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_top_k=3,
        )
        m.fit(X, y)
        assert hasattr(m, "support_")
        # Now triple-clone; descendants must shed fit state.
        m1 = clone(clone(clone(m)))
        assert not hasattr(m1, "support_"), "triple-clone of fitted MRMR must produce an unfitted clone"
        # And the surviving constructor params still match.
        assert m.get_params(deep=True) == m1.get_params(deep=True)


# ---------------------------------------------------------------------------
# E. set_output('pandas') with hybrid FE engaged
# ---------------------------------------------------------------------------


@cache
def _seed1_n1200_pandas_hybrid_fit():
    """Cached ``(X, y, m, out)`` for the seed=1, n=1200 set_output(pandas)
    hybrid fit + transform. Shared between
    test_transform_returns_dataframe_when_hybrid_enabled and
    test_engineered_column_values_finite, which both fit the identical
    config on identical data. Nothing downstream mutates X/y/m/out in place.
    """
    X, y = _build_quadratic_binary(seed=1, n=1200)
    m = _make_mrmr(
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=False,
        fe_hybrid_orth_basis="hermite",
        fe_hybrid_orth_top_k=5,
    ).set_output(transform="pandas")
    m.fit(X, y)
    out = m.transform(X)
    return X, y, m, out


class TestSetOutputPandasWithHybrid:
    """set_output(transform='pandas') with hybrid FE returns a DataFrame whose columns match get_feature_names_out()."""

    def test_transform_returns_dataframe_when_hybrid_enabled(self):
        """With hybrid FE engaged, set_output(pandas) makes transform() return a DataFrame."""
        _X, _y, m, out = _seed1_n1200_pandas_hybrid_fit()
        # Hybrid stage engaged.
        assert len(m.hybrid_orth_features_) > 0, m.hybrid_orth_features_
        # transform on a DataFrame returns DataFrame.
        assert isinstance(out, pd.DataFrame), f"set_output(pandas) + hybrid FE: transform should return DataFrame, got {type(out).__name__}"

    def test_feature_names_out_includes_engineered_names(self):
        """Every engineered hybrid_orth_features_ name appears in get_feature_names_out()."""
        X, y = _build_quadratic_binary(seed=13, n=1200)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=5,
        ).set_output(transform="pandas")
        m.fit(X, y)
        names_out = list(m.get_feature_names_out())
        # Every engineered name appears in feature_names_out.
        for eng_name in m.hybrid_orth_features_:
            assert eng_name in names_out, f"engineered name {eng_name!r} not in get_feature_names_out()={names_out}"

    def test_transform_columns_match_feature_names_out(self):
        """transform()'s DataFrame column order matches get_feature_names_out() exactly."""
        X, y = _build_quadratic_binary(seed=42, n=1200)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=5,
        ).set_output(transform="pandas")
        m.fit(X, y)
        out = m.transform(X)
        assert list(out.columns) == list(m.get_feature_names_out()), (
            f"transform DataFrame columns disagree with get_feature_names_out():\n  columns={list(out.columns)}\n  names_out={list(m.get_feature_names_out())}"
        )

    def test_engineered_column_values_finite(self):
        """Every engineered hybrid_orth_features_ column value in transform() output is finite."""
        _X, _y, m, out = _seed1_n1200_pandas_hybrid_fit()
        for eng_name in m.hybrid_orth_features_:
            if eng_name in out.columns:
                col = out[eng_name].to_numpy()
                assert np.all(np.isfinite(col)), (
                    f"engineered column {eng_name!r} has non-finite values; first bad rows: {np.flatnonzero(~np.isfinite(col))[:5]}"
                )


# ---------------------------------------------------------------------------
# F. Regression target with hybrid FE + StandardScaler + LinearRegression
# ---------------------------------------------------------------------------


class TestRegressionWithHybridFE:
    """MRMR + StandardScaler + LinearRegression with hybrid FE solves a quadratic-signal regression target."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_regression_pipeline_r2_on_quadratic(self, seed):
        """Continuous y = x1^2 - 1 + noise. Raw x1 gives R^2 ~ 0 for a
        linear regressor (cov(x1, x1^2) = 0 for x1 ~ N(0, 1)). With
        hybrid FE the He_2 column makes the linear fit recover the
        signal.
        """
        X, y = _build_quadratic_regression(seed=seed, n=2000)
        n_train = 1500
        Xtr, ytr = X.iloc[:n_train], y.iloc[:n_train]
        Xte, yte = X.iloc[n_train:], y.iloc[n_train:]
        pipe = Pipeline(
            [
                (
                    "mrmr",
                    _make_mrmr(
                        fe_hybrid_orth_enable=True,
                        fe_hybrid_orth_pair_enable=False,
                        fe_hybrid_orth_basis="hermite",
                        fe_hybrid_orth_top_k=5,
                    ),
                ),
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ]
        )
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        r2 = r2_score(yte.to_numpy(), preds)
        assert r2 >= 0.85, (
            f"seed={seed}: regression Pipeline R^2 {r2:.3f} below 0.85 "
            f"on quadratic target; hybrid FE should engineer He_2(x1) "
            f"and let LinearRegression solve it. "
            f"hybrid_orth_features_={pipe.named_steps['mrmr'].hybrid_orth_features_}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_regression_pipeline_predict_finite(self, seed):
        """Predict on held-out data returns finite values (no NaN /
        inf bubbling from engineered cols)."""
        X, y = _build_quadratic_regression(seed=seed, n=1500)
        n_train = 1100
        Xtr, ytr = X.iloc[:n_train], y.iloc[:n_train]
        Xte = X.iloc[n_train:]
        pipe = Pipeline(
            [
                (
                    "mrmr",
                    _make_mrmr(
                        fe_hybrid_orth_enable=True,
                        fe_hybrid_orth_pair_enable=False,
                        fe_hybrid_orth_top_k=5,
                    ),
                ),
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ]
        )
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        assert preds.shape == (Xte.shape[0],), preds.shape
        assert np.all(np.isfinite(preds)), f"seed={seed}: non-finite predictions: {preds[~np.isfinite(preds)][:5]}"
