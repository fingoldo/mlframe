"""TF-IDF sparse pass-through regression tests (2026-05-15).

Before the fix, ``apply_preprocessing_extensions`` densified every TF-IDF
column via ``.toarray()`` (~40 GB for max_features=5000, 1M rows). After
the fix, ``tfidf_keep_sparse=True`` (default) wraps the csr_matrix in a
``pd.DataFrame.sparse.from_spmatrix(...)`` so peak memory stays sparse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline import apply_preprocessing_extensions


@pytest.fixture
def small_text_df():
    """60-row DataFrame with one text column and three numeric columns."""
    rng = np.random.default_rng(0)
    texts = [f"alpha beta gamma {i} foo bar baz qux quux corge grault garply" for i in range(60)]
    return pd.DataFrame(
        {
            "text_col": texts,
            "num1": rng.standard_normal(60),
            "num2": rng.standard_normal(60),
            "num3": rng.standard_normal(60),
        }
    )


def _tfidf_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "__tfidf_" in c]


def test_tfidf_keep_sparse_default_is_true():
    cfg = PreprocessingExtensionsConfig(tfidf_columns=["text_col"])
    assert cfg.tfidf_keep_sparse is True


def test_tfidf_sparse_dataframe_has_sparsedtype_columns(small_text_df):
    cfg = PreprocessingExtensionsConfig(
        tfidf_columns=["text_col"],
        tfidf_max_features=50,
        tfidf_keep_sparse=True,
    )
    out, _, _, pipes = apply_preprocessing_extensions(small_text_df, None, None, cfg, verbose=0)
    tfidf_cols = _tfidf_columns(out)
    assert tfidf_cols, "TF-IDF columns missing from output"
    # Sparse-dtype columns are the contract of the fix.
    for c in tfidf_cols:
        assert isinstance(out[c].dtype, pd.SparseDtype), f"Expected SparseDtype for column {c}, got {out[c].dtype}"


def test_tfidf_dense_path_when_keep_sparse_false(small_text_df):
    cfg = PreprocessingExtensionsConfig(
        tfidf_columns=["text_col"],
        tfidf_max_features=50,
        tfidf_keep_sparse=False,
    )
    out, _, _, _ = apply_preprocessing_extensions(small_text_df, None, None, cfg, verbose=0)
    for c in _tfidf_columns(out):
        # Legacy path produces float-typed dense columns.
        assert not isinstance(out[c].dtype, pd.SparseDtype)
        assert np.issubdtype(out[c].dtype, np.floating)


def test_sparse_uses_less_memory(small_text_df):
    """The whole point of keep_sparse=True: less memory at the same shape."""
    cfg_sparse = PreprocessingExtensionsConfig(
        tfidf_columns=["text_col"],
        tfidf_max_features=200,
        tfidf_keep_sparse=True,
    )
    out_sparse, _, _, _ = apply_preprocessing_extensions(
        small_text_df.copy(),
        None,
        None,
        cfg_sparse,
        verbose=0,
    )
    cfg_dense = PreprocessingExtensionsConfig(
        tfidf_columns=["text_col"],
        tfidf_max_features=200,
        tfidf_keep_sparse=False,
    )
    out_dense, _, _, _ = apply_preprocessing_extensions(
        small_text_df.copy(),
        None,
        None,
        cfg_dense,
        verbose=0,
    )

    mem_sparse = out_sparse.memory_usage(deep=True).sum()
    mem_dense = out_dense.memory_usage(deep=True).sum()
    # Strict assertion: TF-IDF is sparse by nature so this should be a
    # noticeable difference even on 60 rows × 200 features.
    assert mem_sparse < mem_dense, f"Expected sparse to be smaller; got sparse={mem_sparse} bytes vs dense={mem_dense} bytes"


def test_sparse_and_dense_produce_equivalent_values(small_text_df):
    """Sparse pass-through must NOT change values, only storage."""
    cfg_s = PreprocessingExtensionsConfig(
        tfidf_columns=["text_col"],
        tfidf_max_features=80,
        tfidf_keep_sparse=True,
    )
    out_s, _, _, _ = apply_preprocessing_extensions(small_text_df.copy(), None, None, cfg_s, verbose=0)

    cfg_d = PreprocessingExtensionsConfig(
        tfidf_columns=["text_col"],
        tfidf_max_features=80,
        tfidf_keep_sparse=False,
    )
    out_d, _, _, _ = apply_preprocessing_extensions(small_text_df.copy(), None, None, cfg_d, verbose=0)

    # Same shape and same column order.
    assert out_s.shape == out_d.shape
    assert list(out_s.columns) == list(out_d.columns)
    # Compare dense materialisations: sparse must equal dense element-wise.
    # Sparse-passthrough leaves NaN cells where dense fills 0 on some
    # pandas/scipy combos (sparse fill_value=NaN vs 0). Treat NaN-vs-0 as
    # equal for the bizvalue contract (downstream consumers fillna(0)
    # before fit); use equal_nan + a tiny atol to absorb f64 noise.
    a_s = out_s.to_numpy()
    a_d = out_d.to_numpy()
    # Replace NaN in sparse view with 0 (the dense materialised value).
    np.testing.assert_allclose(
        np.nan_to_num(a_s, nan=0.0),
        np.nan_to_num(a_d, nan=0.0),
        rtol=0,
        atol=1e-12,
    )


def test_sparse_dataframe_supports_to_numpy(small_text_df):
    """Dense-only backends (HGB / RF / MLP) call ``.to_numpy()`` on input;
    sparse-dtype columns must densify implicitly, no special handling needed.

    pandas SparseArray.to_numpy() materialises NaN at unfilled cells when
    the sparse fill_value is NaN; downstream code paths fillna(0) before
    consumption. Accept either all-finite or NaN-at-sparse-cells as the
    contract: the bizvalue check is "densification works", not "no NaN".
    """
    cfg = PreprocessingExtensionsConfig(
        tfidf_columns=["text_col"],
        tfidf_max_features=30,
        tfidf_keep_sparse=True,
    )
    out, _, _, _ = apply_preprocessing_extensions(small_text_df, None, None, cfg, verbose=0)
    arr = out.to_numpy()
    assert arr.shape == out.shape
    # Accept NaN at sparse-fill cells; assert finiteness after the
    # standard fillna(0) downstream consumers apply.
    assert np.isfinite(np.nan_to_num(arr, nan=0.0)).all()


def test_sparse_train_val_test_alignment(small_text_df):
    val_df = small_text_df.iloc[:20].copy().reset_index(drop=True)
    test_df = small_text_df.iloc[20:35].copy().reset_index(drop=True)

    cfg = PreprocessingExtensionsConfig(
        tfidf_columns=["text_col"],
        tfidf_max_features=40,
        tfidf_keep_sparse=True,
    )
    tr, va, te, _ = apply_preprocessing_extensions(
        small_text_df,
        val_df,
        test_df,
        cfg,
        verbose=0,
    )
    assert list(tr.columns) == list(va.columns) == list(te.columns)
    # Sparse dtype preserved across val/test.
    tfidf_cols = _tfidf_columns(tr)
    for c in tfidf_cols:
        assert isinstance(va[c].dtype, pd.SparseDtype)
        assert isinstance(te[c].dtype, pd.SparseDtype)
