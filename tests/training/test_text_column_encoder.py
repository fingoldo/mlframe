"""
Tests for the phase-C :class:`TextColumnEncoder` and
:class:`PolarsNativeDispatcher`.

Coverage:
  * Happy-path TF-IDF: shape, vocabulary, sparse output type.
  * Hashing: deterministic n_features regardless of input size.
  * polars / pandas symmetry: same vocab built from either input form.
  * Empty / null / Unicode input doesn't crash (round-3 T8 + T9).
  * fit() then transform(other_df) keeps train vocab (no leak).
  * fit_transform == fit().transform() on the same df (idempotent).
  * Capability detector reports polars-ds version + caps; reset
    works.
  * Dispatcher routes correctly when prefer_polarsds=False.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.sparse import csr_matrix, issparse

from mlframe.training.feature_handling import (
    HashingParams,
    PolarsNativeDispatcher,
    TextColumnEncoder,
    TfidfParams,
    detect_polars_ds_capabilities,
    reset_capability_cache,
)

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def small_text_polars():
    """Small text polars."""
    return pl.DataFrame(
        {
            "txt": [
                "the quick brown fox",
                "jumps over the lazy dog",
                "the lazy dog sleeps",
                "another sentence about foxes",
                "polars data science",
            ],
        }
    )


@pytest.fixture
def small_text_pandas():
    """Small text pandas."""
    return pd.DataFrame(
        {
            "txt": [
                "the quick brown fox",
                "jumps over the lazy dog",
                "the lazy dog sleeps",
                "another sentence about foxes",
                "polars data science",
            ],
        }
    )


# =====================================================================
# 1. TF-IDF happy path
# =====================================================================


class TestTfidfHappyPath:
    """Groups tests covering tfidf happy path."""
    def test_fit_then_transform_polars(self, small_text_polars):
        """Fit then transform polars."""
        enc = TextColumnEncoder(
            column="txt",
            params=TfidfParams(max_features=50, ngram_range=(1, 1)),
        )
        enc.fit(small_text_polars)
        out = enc.transform(small_text_polars)
        assert issparse(out)
        assert isinstance(out, csr_matrix)
        assert out.shape[0] == 5
        assert 0 < out.shape[1] <= 50

    def test_fit_transform_pandas(self, small_text_pandas):
        """Fit transform pandas."""
        enc = TextColumnEncoder(
            column="txt",
            params=TfidfParams(max_features=50, ngram_range=(1, 1)),
        )
        out = enc.fit_transform(small_text_pandas)
        assert issparse(out)
        assert out.shape[0] == 5

    def test_idempotency(self, small_text_polars):
        """fit().transform(df) and fit_transform(df) on the same df
        should produce numerically-identical sparse matrices.

        sklearn 1.x TfidfTransformer accumulates ``np.sum`` in float64
        but the order-of-summation differs between fit-then-transform
        and fit_transform paths (cache invalidation in
        TfidfTransformer.fit_transform). Diffs land at f64-precision
        (~5e-17). Use assert_allclose with a tiny tolerance instead of
        assert_array_equal to absorb that without false-positives."""
        # min_df=1 on this 5-row toy fixture; production default min_df=2 would
        # prune every token to empty vocab on this size.
        enc1 = TextColumnEncoder(column="txt", params=TfidfParams(max_features=20, min_df=1))
        enc1.fit(small_text_polars)
        out1 = enc1.transform(small_text_polars)

        enc2 = TextColumnEncoder(column="txt", params=TfidfParams(max_features=20, min_df=1))
        out2 = enc2.fit_transform(small_text_polars)

        np.testing.assert_allclose(out1.toarray(), out2.toarray(), rtol=0, atol=1e-15)


# =====================================================================
# 2. Hashing happy path
# =====================================================================


class TestHashingHappyPath:
    """Groups tests covering hashing happy path."""
    def test_hashing_deterministic_n_features(self, small_text_polars):
        """Hashing deterministic n features."""
        enc = TextColumnEncoder(
            column="txt",
            params=HashingParams(n_features=128),
        )
        enc.fit(small_text_polars)
        out = enc.transform(small_text_polars)
        assert out.shape == (5, 128)

    def test_hashing_no_fit_required_semantics(self, small_text_polars):
        """Hashing is stateless, but our wrapper still enforces fit
        for API consistency."""
        from sklearn.exceptions import NotFittedError

        enc = TextColumnEncoder(column="txt", params=HashingParams(n_features=64))
        with pytest.raises((NotFittedError, RuntimeError), match="not fitted"):
            enc.transform(small_text_polars)
        enc.fit(small_text_polars)
        out = enc.transform(small_text_polars)
        assert out.shape == (5, 64)


# =====================================================================
# 3. polars / pandas symmetry
# =====================================================================


class TestSymmetry:
    """Groups tests covering symmetry."""
    def test_same_vocab_from_polars_and_pandas(self, small_text_polars, small_text_pandas):
        """A polars frame and a pandas frame with the same content
        produce the same fitted TF-IDF vocabulary."""
        enc_pl = TextColumnEncoder(
            column="txt",
            params=TfidfParams(max_features=50, ngram_range=(1, 1)),
        )
        enc_pd = TextColumnEncoder(
            column="txt",
            params=TfidfParams(max_features=50, ngram_range=(1, 1)),
        )
        enc_pl.fit(small_text_polars)
        enc_pd.fit(small_text_pandas)

        assert enc_pl._vectorizer.vocabulary_ == enc_pd._vectorizer.vocabulary_


# =====================================================================
# 4. Edge inputs
# =====================================================================


class TestEdgeInputs:
    # min_df=1 throughout: these edge fixtures are 4-token, every term is hapax.
    # Production default min_df=2 prunes them to empty vocab (sklearn raises
    # "After pruning, no terms remain").
    """Groups tests covering edge inputs."""
    def test_empty_strings_dont_crash(self):
        """Empty strings dont crash."""
        df = pl.DataFrame({"txt": ["", "", "", "real text"]})
        enc = TextColumnEncoder(column="txt", params=TfidfParams(max_features=10, min_df=1))
        enc.fit(df)
        out = enc.transform(df)
        assert out.shape == (4, len(enc._vectorizer.vocabulary_))
        # Empty rows produce zero rows in the sparse matrix.
        assert out[0].nnz == 0

    def test_null_in_polars_coerced(self):
        """Null in polars coerced."""
        df = pl.DataFrame({"txt": ["hello", None, "world", None]})
        enc = TextColumnEncoder(column="txt", params=TfidfParams(max_features=10, min_df=1))
        enc.fit(df)
        out = enc.transform(df)
        assert out.shape[0] == 4

    def test_nan_in_pandas_coerced(self):
        """Nan in pandas coerced."""
        df = pd.DataFrame({"txt": ["hello", float("nan"), "world", float("nan")]})
        enc = TextColumnEncoder(column="txt", params=TfidfParams(max_features=10, min_df=1))
        enc.fit(df)
        out = enc.transform(df)
        assert out.shape[0] == 4

    def test_unicode_roundtrip(self):
        """Unicode roundtrip."""
        df = pl.DataFrame({"txt": ["привет мир", "🔥 launch", "مرحبا", "english"]})
        enc = TextColumnEncoder(column="txt", params=TfidfParams(max_features=20, min_df=1))
        enc.fit(df)
        out = enc.transform(df)
        assert out.shape == (4, len(enc._vectorizer.vocabulary_))


# =====================================================================
# 5. No leak: train vocab applied to held-out frame
# =====================================================================


class TestNoLeak:
    """Groups tests covering no leak."""
    def test_train_vocab_applied_to_test(self):
        """Train vocab applied to test."""
        train = pl.DataFrame({"txt": ["foo bar", "bar baz", "baz qux"]})
        test = pl.DataFrame({"txt": ["foo unseen", "qux unseen", "completely new"]})

        enc = TextColumnEncoder(
            column="txt",
            params=TfidfParams(max_features=50, ngram_range=(1, 1)),
        )
        enc.fit(train)
        out_train = enc.transform(train)
        out_test = enc.transform(test)

        # Train and test sparse matrices share columns (vocabulary).
        assert out_train.shape[1] == out_test.shape[1]
        # Test should have only TRAIN vocab tokens. "unseen" is not
        # in train -> contributes 0 features.
        # The n_unseen "test row" should be all-zero.
        assert out_test[2].nnz == 0  # "completely new" all OOV


# =====================================================================
# 6. Capability detector + dispatcher
# =====================================================================


class TestCapabilityDetector:
    """Groups tests covering capability detector."""
    def test_detect_returns_set(self):
        """Detect returns set."""
        reset_capability_cache()
        caps = detect_polars_ds_capabilities()
        assert isinstance(caps, frozenset)
        # If polars-ds is installed, at least Blueprint methods we use
        # should be there.
        if any(c.startswith("polars_ds:") for c in caps):
            assert "blueprint.scale" in caps
            assert "blueprint.impute" in caps  # phase M wired this
            assert "blueprint.ordinal_encode" in caps

    def test_dispatcher_prefer_false_disables(self):
        """Dispatcher prefer false disables."""
        d = PolarsNativeDispatcher(prefer_polarsds=False)
        assert d.has("blueprint.scale") is False
        assert d.has("blueprint.tfidf") is False

    def test_dispatcher_prefer_true_uses_caps(self):
        """Dispatcher prefer true uses caps."""
        d = PolarsNativeDispatcher(prefer_polarsds=True)
        # The dispatcher's blueprint.* caps require both ``polars_ds`` AND
        # its ``polars_ds.pipeline`` submodule (the Blueprint factory lives
        # there). Some installed polars_ds builds ship without
        # ``.pipeline`` (lightweight subset); the dispatcher correctly
        # reports caps=0 in that case. Skip when either is missing.
        try:
            importlib.import_module("polars_ds")
            importlib.import_module("polars_ds.pipeline")
        except ImportError:  # pragma: no cover
            pytest.skip("polars-ds (or polars_ds.pipeline) not installed")
        assert d.has("blueprint.impute")
        assert d.has("blueprint.scale")

    def test_get_version_returns_string(self):
        """Get version returns string."""
        reset_capability_cache()
        d = PolarsNativeDispatcher(prefer_polarsds=True)
        v = d.get_version()
        # get_version() depends on the ``polars_ds.pipeline`` submodule
        # being importable; treat absence of either polars_ds OR the
        # .pipeline submodule as the None-case.
        try:
            importlib.import_module("polars_ds")
            importlib.import_module("polars_ds.pipeline")

            _has_full_polars_ds = True
        except ImportError:
            _has_full_polars_ds = False
        if _has_full_polars_ds:
            assert isinstance(v, str)
        else:
            assert v is None


# =====================================================================
# 7. Signature stability
# =====================================================================


class TestSignature:
    """Groups tests covering signature."""
    def test_signature_stable_for_same_params(self):
        """Signature stable for same params."""
        e1 = TextColumnEncoder(
            column="txt",
            params=TfidfParams(max_features=100, ngram_range=(1, 2)),
        )
        e2 = TextColumnEncoder(
            column="txt",
            params=TfidfParams(max_features=100, ngram_range=(1, 2)),
        )
        assert e1.signature() == e2.signature()

    def test_signature_changes_with_params(self):
        """Signature changes with params."""
        e1 = TextColumnEncoder(column="txt", params=TfidfParams(max_features=100))
        e2 = TextColumnEncoder(column="txt", params=TfidfParams(max_features=200))
        assert e1.signature() != e2.signature()
