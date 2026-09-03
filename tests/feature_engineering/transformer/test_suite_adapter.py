"""Tests for ``mlframe.feature_engineering.transformer._suite_adapter.ShortlistTransformerAdapter``.

Only a single meta-test (naming-convention check for the deprecated ``seed`` alias) previously touched
this class -- its actual sklearn-adapter behavior (Mode A/B dispatch, calling-convention auto-detection,
passthrough assembly, get_feature_names_out) had zero coverage. Uses small FAKE ``compute_fn`` stand-ins
(not the real production shortlist transformers) to isolate the adapter's OWN logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.feature_engineering.transformer._suite_adapter import ShortlistTransformerAdapter


def _rff_style_compute(X, *, seed, X_query=None, n_features=2, splitter=None):
    """Fake RFF-style compute_fn: single positional X + X_query kwarg, no y_train/X_train param names
    (matches the real convention -- the adapter detects supervised-style calls by param NAME)."""
    if splitter is not None:
        # Mode A (OOF): X_query is None, one row of features per X row.
        n = np.asarray(X).shape[0]
        out = np.arange(n * n_features, dtype=np.float64).reshape(n, n_features)
    else:
        n = np.asarray(X_query).shape[0]
        out = np.full((n, n_features), float(seed), dtype=np.float64)
    return pl.DataFrame({f"rff_{i}": out[:, i] for i in range(n_features)})


def _knn_style_compute(X_train, y_train, X_query, *, seed, splitter=None, n_features=1):
    """Fake supervised kNN-style compute_fn: (X_train, y_train, X_query, splitter=...) signature."""
    if X_query is None:
        n = np.asarray(X_train).shape[0]
    else:
        n = np.asarray(X_query).shape[0]
    out = np.full((n, n_features), float(np.asarray(y_train).mean()) if y_train is not None else -1.0)
    return pl.DataFrame({f"knn_{i}": out[:, i] for i in range(n_features)})


def _make_xy(n=20, p=3, seed=0):
    """Small numpy train frame + binary target."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = (rng.random(n) < 0.5).astype(int)
    return X, y


class TestCallingConventionDetection:
    """Groups tests covering RFF-style vs kNN-style compute_fn auto-detection in transform()."""

    def test_rff_style_unsupervised_fit_transform(self):
        """An RFF-style (unsupervised) compute_fn is called correctly via the X_query-kwarg convention."""
        X, _y = _make_xy()
        adapter = ShortlistTransformerAdapter(_rff_style_compute, needs_y=False, random_state=7)
        adapter.fit(X)
        out = adapter.transform(X)
        assert isinstance(out, pd.DataFrame)
        assert "rff_0" in out.columns and "rff_1" in out.columns
        # seed=7 was threaded through -- Mode B fills every row with float(seed).
        assert (out["rff_0"] == 7.0).all()

    def test_knn_style_supervised_fit_transform(self):
        """A supervised kNN-style compute_fn is called correctly via the (X_train, y_train, X_query) convention."""
        X, y = _make_xy()
        adapter = ShortlistTransformerAdapter(_knn_style_compute, needs_y=True, random_state=3)
        adapter.fit(X, y)
        out = adapter.transform(X)
        assert "knn_0" in out.columns
        assert np.isclose(out["knn_0"].iloc[0], float(y.mean()))


class TestNeedsYContract:
    """Groups tests covering the needs_y flag's enforcement."""

    def test_needs_y_true_without_y_raises(self):
        """A supervised compute_fn's adapter must reject fit(X) with no y."""
        X, _y = _make_xy()
        adapter = ShortlistTransformerAdapter(_knn_style_compute, needs_y=True)
        with pytest.raises(ValueError, match="requires y_train"):
            adapter.fit(X, y=None)

    def test_needs_y_false_allows_missing_y(self):
        """An unsupervised compute_fn's adapter must accept fit(X) with no y."""
        X, _y = _make_xy()
        adapter = ShortlistTransformerAdapter(_rff_style_compute, needs_y=False)
        adapter.fit(X)  # must not raise
        out = adapter.transform(X)
        assert len(out) == len(X)


class TestPassthrough:
    """Groups tests covering the passthrough concatenation vs feats-only output."""

    def test_passthrough_true_concatenates_raw_and_engineered_columns(self):
        """Passthrough true concatenates raw and engineered columns."""
        X, _y = _make_xy(p=3)
        adapter = ShortlistTransformerAdapter(_rff_style_compute, needs_y=False, passthrough=True)
        adapter.fit(X)
        out = adapter.transform(X)
        # 3 raw columns (f0,f1,f2) + 2 engineered (rff_0, rff_1).
        assert out.shape[1] == 3 + 2

    def test_passthrough_false_returns_only_engineered_columns(self):
        """Passthrough false returns only engineered columns."""
        X, _y = _make_xy(p=3)
        adapter = ShortlistTransformerAdapter(_rff_style_compute, needs_y=False, passthrough=False)
        adapter.fit(X)
        out = adapter.transform(X)
        assert list(out.columns) == ["rff_0", "rff_1"]

    def test_row_count_and_index_preserved_under_passthrough(self):
        """The output row count must match the input, and concatenation must not misalign rows."""
        X, _y = _make_xy(n=15, p=2)
        adapter = ShortlistTransformerAdapter(_rff_style_compute, needs_y=False)
        adapter.fit(X)
        out = adapter.transform(X)
        assert len(out) == 15


class TestDeprecatedSeedAlias:
    """Groups tests covering the deprecated seed= constructor kwarg."""

    def test_seed_kwarg_overrides_random_state_and_warns(self, caplog):
        """Seed kwarg overrides random state and warns."""
        import logging

        with caplog.at_level(logging.WARNING):
            adapter = ShortlistTransformerAdapter(_rff_style_compute, needs_y=False, random_state=1, seed=99)
        assert adapter.random_state == 99
        assert any("deprecated" in r.message for r in caplog.records)


class TestFitTransformModeA:
    """Groups tests covering fit_transform's OOF (Mode A) dispatch."""

    def test_unsupervised_compute_fn_falls_back_to_mode_b(self):
        """A compute_fn with no 'splitter' param has no OOF concern -- fit_transform must fall back to
        plain transform() rather than attempting an OOF split."""
        X, _y = _make_xy()

        def _no_splitter_fn(X, *, seed, X_query=None):
            """A compute_fn signature with no splitter param at all."""
            n = np.asarray(X_query if X_query is not None else X).shape[0]
            return pl.DataFrame({"f": np.zeros(n)})

        adapter = ShortlistTransformerAdapter(_no_splitter_fn, needs_y=False)
        out = adapter.fit_transform(X)
        assert len(out) == len(X)

    def test_splitter_aware_compute_fn_receives_a_kfold_splitter(self):
        """A compute_fn declaring 'splitter' must receive one from _make_oof_splitter, and fit_transform's
        OOF branch (X_query=None) must produce one feature row per training row."""
        X, y = _make_xy(n=12)
        adapter = ShortlistTransformerAdapter(_knn_style_compute, needs_y=True, random_state=0)
        out = adapter.fit_transform(X, y)
        assert len(out) == len(X)


class TestFeatureNamesOut:
    """Groups tests covering get_feature_names_out's before/after-transform contract."""

    def test_before_transform_falls_back_to_input_columns(self):
        """Before any transform, get_feature_names_out must fall back to the fit-time input columns."""
        X, _y = _make_xy(p=2)
        adapter = ShortlistTransformerAdapter(_rff_style_compute, needs_y=False)
        adapter.fit(X)
        names = adapter.get_feature_names_out()
        assert list(names) == ["f0", "f1"]

    def test_after_transform_returns_recorded_output_names(self):
        """After a transform, get_feature_names_out must reflect the actual output column set."""
        X, _y = _make_xy(p=2)
        adapter = ShortlistTransformerAdapter(_rff_style_compute, needs_y=False, passthrough=False)
        adapter.fit(X)
        adapter.transform(X)
        names = list(adapter.get_feature_names_out())
        assert names == ["rff_0", "rff_1"]

    def test_dataframe_input_preserves_real_column_names(self):
        """A pandas input's real column names must be used, not the f0/f1/... fallback."""
        X = pd.DataFrame({"alpha": [1.0, 2.0, 3.0], "beta": [4.0, 5.0, 6.0]})
        adapter = ShortlistTransformerAdapter(_rff_style_compute, needs_y=False, passthrough=False)
        adapter.fit(X)
        assert adapter._input_columns_ == ["alpha", "beta"]
