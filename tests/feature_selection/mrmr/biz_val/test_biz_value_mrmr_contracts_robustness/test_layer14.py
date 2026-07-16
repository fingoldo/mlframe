"""Consolidated from test_biz_value_mrmr_layer14.py.

Layer 14 biz_value MRMR contracts: sklearn PIPELINE / CLONE / PICKLE /
GridSearchCV / set_output / fit_transform / CV-stability.

WHY THIS LAYER
--------------
Every production tabular pipeline composes feature selection with a
downstream estimator inside ``sklearn.pipeline.Pipeline`` and tunes
hyperparameters via ``GridSearchCV`` / ``RandomizedSearchCV``. The
ecosystem assumes the selector honours the BaseEstimator + Transformer
+ SelectorMixin contracts BIT-EXACTLY:

* ``clone(est)`` must return a fresh, unfitted estimator carrying the
  same constructor parameters but NO fitted attributes and NO data
  references.
* ``pickle.dumps/loads`` must round-trip the fitted state so the same
  ``transform`` output reappears on the loaded copy.
* ``Pipeline([...]).fit(X, y).predict(X_test)`` must work end-to-end
  without the selector raising or producing degenerate (0-column)
  intermediate matrices.
* ``GridSearchCV(pipe, {selector__param: [a, b]}).fit(X, y)`` must
  pick one of the candidates without crashing on either.
* ``set_output(transform='pandas')`` must produce a DataFrame even
  when ``transform()`` receives ndarray.
* ``fit_transform(X, y)`` must match ``fit(X, y).transform(X)``
  feature-by-feature.
* Cross-validation over 5 folds must yield STABLE support
  intersection: the strong signal columns appear in every fold.

A regression in any of those contracts breaks every production model
that touches MRMR, which is most of them.

DATA DESIGN
-----------
* n = 2000 rows, p = 15 columns.
* x1, x2: linear-additive signal for a logistic ``y = 1 [x1 + 0.7*x2
  > 0]``. Both are individually informative and JOINTLY informative
  - any feature selector that respects relevance should keep at
  least one of them.
* noise_0 .. noise_12: i.i.d. ``Normal(0, 1)`` independent of y.

DCD is enabled by default (Wave 9 flip 2026-05-30). Layer 14 does NOT
override that flip; it tests the production default surface.

NOT PINNED
----------
* Exact ``support_`` length under different ``quantization_nbins`` -
  varying the bin count changes the MI floor and may admit/reject
  marginal noise columns. We pin that x1 is selected (strongest
  signal) and that the selector does not crash, not the exact list.
* Exact numeric predictions from the downstream LogisticRegression -
  CV variance + small-n + bin-quantisation interact, and Layer 14 is
  about the sklearn-integration contract, not classifier accuracy.
"""

from __future__ import annotations

import pickle
import warnings
from functools import cache

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Data builder
# ---------------------------------------------------------------------------


N_TOTAL = 2_000
N_NOISE = 13
SEED = 14_001


def _build_linear_data(seed: int = SEED):
    """Clean two-signal logistic dataset with 13 noise columns.

    y = 1[x1 + 0.7*x2 > 0]. Both x1 and x2 carry signal; x1 is the
    stronger of the two by construction (unit coefficient vs 0.7).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(N_TOTAL)
    x2 = rng.standard_normal(N_TOTAL)
    y_arr = ((x1 + 0.7 * x2) > 0).astype(np.int64)
    cols = {"x1": x1, "x2": x2}
    for k in range(N_NOISE):
        cols[f"noise_{k}"] = rng.standard_normal(N_TOTAL)
    X = pd.DataFrame(cols)
    y = pd.Series(y_arr, name="y")
    return X, y


def _make_mrmr(**overrides):
    """Standard layer-14 MRMR config. Keeps default DCD-enabled flip but
    disables polynom-FE (irrelevant to sklearn-integration contracts and
    drives wall time up an order of magnitude).
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    kwargs = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _fit_with_warnings_silenced(sel, X, y):
    """Fit ``sel`` on ``(X, y)`` with warnings silenced; return the fitted estimator."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sel.fit(X, y)


@cache
def _default_fit():
    """Cached ``(X, y, sel)`` for the default (nbins=10, seed=42) fit shared
    across TestPickleRoundTrip's 3 tests and TestSetOutputPandas's 2 tests
    (the latter fit through a ``set_output(transform='pandas')`` wrapper
    that doesn't change the underlying config). Nothing downstream mutates
    X/y/sel in place -- pickling and transform() are both read-only.
    """
    X, y = _build_linear_data()
    sel = _make_mrmr(quantization_nbins=10, random_seed=42)
    _fit_with_warnings_silenced(sel, X, y)
    return X, y, sel


@cache
def _set_output_pandas_fit():
    """Cached ``(X, y, sel)`` for the ``set_output(transform='pandas')`` fit
    shared across TestSetOutputPandas's 2 tests. Nothing downstream mutates
    X/y/sel in place.
    """
    X, y = _build_linear_data()
    sel = _make_mrmr(quantization_nbins=10, random_seed=42).set_output(transform="pandas")
    _fit_with_warnings_silenced(sel, X, y)
    return X, y, sel


@cache
def _pipeline_fit():
    """Cached ``(X_tr, X_te, y_tr, y_te, pipe)`` for the default MRMR+LR
    pipeline shared across TestPipelineEndToEnd's 2 tests. Nothing
    downstream mutates the frames/pipe in place -- predict()/score() are
    both read-only.
    """
    X, y = _build_linear_data()
    X_tr, X_te = X.iloc[:-200], X.iloc[-200:]
    y_tr, y_te = y.iloc[:-200], y.iloc[-200:]
    pipe = Pipeline(
        [
            ("mrmr", _make_mrmr(quantization_nbins=10, random_seed=42)),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_tr, y_tr)
    return X_tr, X_te, y_tr, y_te, pipe


# ---------------------------------------------------------------------------
# Contract 1: sklearn.clone
# ---------------------------------------------------------------------------


class TestClone:
    """``sklearn.base.clone`` must produce an unfitted estimator with
    identical constructor params and no shared mutable state.
    """

    def test_clone_returns_distinct_unfitted_instance(self):
        """clone() returns a distinct, unfitted object with no fitted attrs."""
        m = _make_mrmr(quantization_nbins=10, random_seed=7)
        m2 = clone(m)
        assert m is not m2, "clone must return a distinct object"
        assert not hasattr(m2, "support_"), "clone must not carry fitted attrs"
        assert not hasattr(m2, "feature_names_in_"), "clone must not carry feature_names_in_ from the source"

    def test_clone_preserves_get_params(self):
        """clone() round-trips get_params(deep=True) bit-exactly."""
        m = _make_mrmr(quantization_nbins=12, random_seed=11)
        m2 = clone(m)
        # ``get_params(deep=True)`` is what GridSearchCV inspects;
        # cloned params MUST round-trip bit-exactly.
        assert m.get_params(deep=True) == m2.get_params(deep=True)

    def test_clone_after_fit_yields_unfitted_clone(self):
        """Fitting the source MUST NOT leak fitted state into the clone."""
        X, y = _build_linear_data()
        m = _make_mrmr(quantization_nbins=10, random_seed=3)
        _fit_with_warnings_silenced(m, X, y)
        assert hasattr(m, "support_")
        m2 = clone(m)
        assert not hasattr(m2, "support_"), "clone of a fitted MRMR must be unfitted; sklearn's " "clone() protocol forbids leaking fitted attrs."
        # Constructor params still survive.
        assert m.get_params(deep=True) == m2.get_params(deep=True)


# ---------------------------------------------------------------------------
# Contract 2: get_params / set_params round-trip
# ---------------------------------------------------------------------------


class TestParamsRoundTrip:
    """``get_params``/``set_params`` must round-trip, directly and via a Pipeline namespace."""

    def test_set_params_then_get_params_matches(self):
        """set_params() updates are visible through get_params()."""
        m = _make_mrmr()
        m.set_params(quantization_nbins=8, random_seed=99)
        assert m.get_params()["quantization_nbins"] == 8
        assert m.get_params()["random_seed"] == 99

    def test_set_params_inside_pipeline_via_namespace(self):
        """Pipeline routes ``step__param`` via set_params -- that path
        must update the inner MRMR.
        """
        pipe = Pipeline(
            [
                ("mrmr", _make_mrmr(quantization_nbins=10)),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )
        pipe.set_params(mrmr__quantization_nbins=5)
        assert pipe.named_steps["mrmr"].quantization_nbins == 5


# ---------------------------------------------------------------------------
# Contract 3: pickle round-trip
# ---------------------------------------------------------------------------


class TestPickleRoundTrip:
    """A fitted MRMR must pickle.dumps -> pickle.loads back to an
    equivalent fitted estimator: same support_, same DCD summary, same
    transform output. This is the joblib persistence contract that
    production model registries (mlflow, sagemaker, dvc) rely on.
    """

    def test_pickle_round_trip_preserves_support(self):
        """pickle round-trip preserves support_ and get_feature_names_out()."""
        _X, _y, sel = _default_fit()
        blob = pickle.dumps(sel)
        sel2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        # support_ identical content (allow ndarray vs list)
        assert list(sel.support_) == list(sel2.support_), f"pickle round-trip changed support_; before={list(sel.support_)} " f"after={list(sel2.support_)}"
        assert list(sel.get_feature_names_out()) == list(sel2.get_feature_names_out())

    def test_pickle_round_trip_preserves_transform(self):
        """pickle round-trip preserves transform() output type and values."""
        X, _y, sel = _default_fit()
        out_a = sel.transform(X)
        sel2 = pickle.loads(pickle.dumps(sel))  # nosec B301 -- round-trip of a locally-created, trusted object
        out_b = sel2.transform(X)
        assert isinstance(out_b, type(out_a)), f"transform return type changed across pickle; " f"before={type(out_a).__name__} after={type(out_b).__name__}"
        if isinstance(out_a, pd.DataFrame):
            assert list(out_a.columns) == list(out_b.columns)
            assert np.array_equal(out_a.values, out_b.values), "pickle round-trip changed transform values"
        else:
            assert np.array_equal(out_a, out_b)

    def test_pickle_round_trip_preserves_dcd_summary(self):
        """``dcd_`` is the Wave-9 fitted attribute the audit harness +
        downstream monitoring rely on. Loading the pickle MUST replay
        it; an unpickled selector with ``dcd_=None`` would silently
        break downstream cluster diagnostics.
        """
        _X, _y, sel = _default_fit()
        # Default flip means DCD is ON; ``dcd_`` is populated.
        assert sel.dcd_ is not None
        sel2 = pickle.loads(pickle.dumps(sel))  # nosec B301 -- round-trip of a locally-created, trusted object
        assert sel2.dcd_ is not None, "pickle round-trip dropped dcd_ summary"
        # Top-level scalar fields must match (lists/dicts compare value-equal).
        for key in ("n_anchors", "n_pruned", "n_swaps"):
            assert sel.dcd_[key] == sel2.dcd_[key], f"dcd_['{key}'] changed across pickle: " f"before={sel.dcd_[key]} after={sel2.dcd_[key]}"


# ---------------------------------------------------------------------------
# Contract 4: fit_transform == fit().transform()
# ---------------------------------------------------------------------------


class TestFitTransformConsistency:
    """``fit_transform(X, y)`` must match ``fit(X, y).transform(X)`` exactly."""

    def test_fit_transform_matches_fit_then_transform(self):
        """fit_transform() and fit().transform() select the same columns and values."""
        X, y = _build_linear_data()
        # Two SEPARATE estimators so any internal state mutation in
        # the fit_transform fast path can't borrow from the fit path.
        a = _make_mrmr(quantization_nbins=10, random_seed=42)
        b = _make_mrmr(quantization_nbins=10, random_seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_a = a.fit_transform(X, y)
            b.fit(X, y)
            out_b = b.transform(X)
        # Same return type.
        assert isinstance(out_a, type(out_b)) or isinstance(out_b, type(out_a)), (
            f"fit_transform / fit-then-transform return types differ: " f"{type(out_a).__name__} vs {type(out_b).__name__}"
        )
        # Same selected columns.
        assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())
        # Same values.
        va = out_a.values if isinstance(out_a, pd.DataFrame) else np.asarray(out_a)
        vb = out_b.values if isinstance(out_b, pd.DataFrame) else np.asarray(out_b)
        assert np.array_equal(va, vb), "fit_transform(X, y) values differ from fit(X, y).transform(X) -- " "the TransformerMixin default contract is violated."


# ---------------------------------------------------------------------------
# Contract 5: set_output('pandas')
# ---------------------------------------------------------------------------


class TestSetOutputPandas:
    """``set_output(transform='pandas')`` must force DataFrame output."""

    def test_set_output_pandas_returns_dataframe(self):
        """transform() on an ndarray still returns a DataFrame under set_output(pandas)."""
        X, _y, sel = _set_output_pandas_fit()
        out = sel.transform(X.to_numpy())
        assert isinstance(out, pd.DataFrame), f"set_output(transform='pandas') must return DataFrame; " f"got {type(out).__name__}"

    def test_set_output_pandas_columns_match_feature_names_out(self):
        """DataFrame column count matches len(get_feature_names_out())."""
        X, _y, sel = _set_output_pandas_fit()
        out = sel.transform(X)
        assert isinstance(out, pd.DataFrame)
        assert len(out.columns) == len(sel.get_feature_names_out()), (
            f"set_output(pandas) DataFrame column count " f"({len(out.columns)}) != len(get_feature_names_out()) " f"({len(sel.get_feature_names_out())})"
        )


# ---------------------------------------------------------------------------
# Contract 6: Pipeline end-to-end
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd:
    """``Pipeline([('mrmr', MRMR()), ('clf', LR())]).fit().predict()``
    is the modal production code path. It must work without
    intermediate manual ``set_output`` config.
    """

    def test_pipeline_fit_predict_runs_and_produces_correct_shape(self):
        """pipe.fit(X_tr, y_tr).predict(X_te) runs end-to-end and returns the right shape."""
        _X_tr, X_te, _y_tr, _y_te, pipe = _pipeline_fit()
        preds = pipe.predict(X_te)
        assert preds.shape == (200,), f"Pipeline predict() shape mismatch: got {preds.shape}, " f"expected (200,)"

    def test_pipeline_beats_random_on_clean_signal(self):
        """Anchor of usefulness: the pipeline must beat a 0.5 baseline
        decisively on a clean two-signal logistic dataset. Otherwise
        the MRMR step is dropping the signal columns and the
        integration contract is technically met but semantically
        broken.
        """
        _X_tr, X_te, _y_tr, y_te, pipe = _pipeline_fit()
        score = pipe.score(X_te, y_te)
        assert score > 0.80, (
            f"Pipeline test-accuracy {score:.3f} is implausibly low for "
            f"a clean two-signal logistic dataset; MRMR likely dropped "
            f"x1/x2 from the support."
        )


# ---------------------------------------------------------------------------
# Contract 7: GridSearchCV over Pipeline
# ---------------------------------------------------------------------------


class TestGridSearchCV:
    """``GridSearchCV`` must tune the inner MRMR via the Pipeline namespace."""

    def test_gridsearchcv_over_mrmr_quantization_nbins(self):
        """GridSearchCV must enumerate {nbins: [5, 10]} via the
        Pipeline ``mrmr__quantization_nbins`` namespace and pick one
        without crashing on either.
        """
        X, y = _build_linear_data()
        pipe = Pipeline(
            [
                ("mrmr", _make_mrmr()),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )
        grid = GridSearchCV(
            pipe,
            param_grid={"mrmr__quantization_nbins": [5, 10]},
            cv=3,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid.fit(X, y)
        chosen = grid.best_params_["mrmr__quantization_nbins"]
        assert chosen in (5, 10), f"GridSearchCV must pick a candidate from the grid; " f"got {chosen}"
        # Sanity: best_score_ should be well above random.
        assert grid.best_score_ > 0.80, (
            f"GridSearchCV best_score_={grid.best_score_:.3f} too low "
            f"for a clean signal dataset; the grid likely crashed on "
            f"one candidate and returned a degenerate fallback."
        )


# ---------------------------------------------------------------------------
# Contract 8: cross-validation support stability
# ---------------------------------------------------------------------------


class TestCVSupportStability:
    """On a clean signal dataset the strongest column (``x1``) must
    appear in EVERY fold's support_. If 5-fold CV admits a fold where
    x1 is dropped, the selector is unstable on signal at this n / p.
    """

    def test_x1_appears_in_every_5fold_support(self):
        """The strongest signal column x1 survives selection in every one of 5 CV folds."""
        X, y = _build_linear_data()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        x1_appearances = 0
        per_fold_supports = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tr_idx, _ in kf.split(X):
                sel = _make_mrmr(quantization_nbins=10, random_seed=42)
                sel.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                names = list(sel.get_feature_names_out())
                per_fold_supports.append(names)
                if "x1" in names:
                    x1_appearances += 1
        assert x1_appearances == 5, f"x1 missing from {5 - x1_appearances}/5 CV folds; " f"supports={per_fold_supports}"
