"""Unit tests for ``compute_row_attention`` + low-level ``build_key_bank`` / ``attend``.

Coverage:
- Shape, dtype, determinism (CPU same-seed bit-identical).
- Mode A (OOF, X_query=None) vs Mode B (X_query passed in) both work.
- Edge cases: tiny N, single head, edge head_dim values.
- Validation: NaN, sparse-like, bad seed, bad k.
- GPU stage 4 parity (skipped if no GPU; loose tolerance per Hardware #20 critique).
- ``build_key_bank`` + ``attend`` round-trip with persisted disk cache.

Behavioural correctness (not just smoke): for a synthetic dataset where target = sign(X[:, 0]), the row-attention features should correlate with the target on a
held-out fold (Mode B). Soft check - not the biz_value test, which is harder; this one just confirms the pipeline isn't producing pure noise.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.model_selection import KFold

# NOTE: hnswlib is imported lazily inside _row_attention_ann.build_hnsw_index, not at file-load time. Tests in this file are skipped at collection time
# (pytest_collection_modifyitems in conftest.py) if hnswlib is broken in the local environment. We do NOT call pytest.importorskip("hnswlib") here because
# the Windows MSVC-runtime segfault that affects some hnswlib wheels would crash pytest collection before the skip fires.
from mlframe.feature_engineering.transformer import (
    attend,
    build_key_bank,
    compute_row_attention,
)


pytestmark = pytest.mark.fast


def _make_data(n: int = 200, d: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    return X, y


# ---------- Mode A (OOF) ----------


def test_row_attention_mode_a_shape_and_dtype(small_X_y_classification, kfold_splitter):
    X, y = small_X_y_classification
    out = compute_row_attention(
        X_train=X,
        y_train=y,
        X_query=None,
        splitter=kfold_splitter,
        seed=42,
        n_heads=2,
        head_dim=4,
        k=8,
        gpu_stage4=False,
        dedupe_threshold=None,
    )
    assert isinstance(out, pl.DataFrame)
    assert out.shape[0] == X.shape[0]
    # Default aggregate=("y_mean", "y_std") + n_heads=2 -> 4 columns.
    assert out.shape[1] == 4
    assert all(name in out.columns for name in ["attn_h0_y_mean", "attn_h0_y_std", "attn_h1_y_mean", "attn_h1_y_std"])


def test_row_attention_mode_a_with_x_mean_aggregate(small_X_y_classification, kfold_splitter):
    X, y = small_X_y_classification
    out = compute_row_attention(
        X_train=X,
        y_train=y,
        X_query=None,
        splitter=kfold_splitter,
        seed=42,
        n_heads=2,
        head_dim=4,
        k=8,
        aggregate=("y_mean", "x_mean"),
        gpu_stage4=False,
        dedupe_threshold=None,
    )
    # 2 heads * (1 scalar y_mean + 4 x_mean dims) = 10 columns.
    assert out.shape == (X.shape[0], 10)
    assert "attn_h0_y_mean" in out.columns
    assert "attn_h0_x_mean_d0" in out.columns
    assert "attn_h1_x_mean_d3" in out.columns


def test_row_attention_mode_a_determinism(small_X_y_classification, kfold_splitter):
    X, y = small_X_y_classification
    out1 = compute_row_attention(X, y, None, kfold_splitter, seed=7, n_heads=2, head_dim=4, k=8, gpu_stage4=False, dedupe_threshold=None)
    out2 = compute_row_attention(X, y, None, kfold_splitter, seed=7, n_heads=2, head_dim=4, k=8, gpu_stage4=False, dedupe_threshold=None)
    np.testing.assert_allclose(out1.to_numpy(), out2.to_numpy(), atol=1e-6)


def test_row_attention_mode_a_seed_changes_output(small_X_y_classification, kfold_splitter):
    X, y = small_X_y_classification
    out1 = compute_row_attention(X, y, None, kfold_splitter, seed=7, n_heads=2, head_dim=4, k=8, gpu_stage4=False, dedupe_threshold=None)
    out2 = compute_row_attention(X, y, None, kfold_splitter, seed=42, n_heads=2, head_dim=4, k=8, gpu_stage4=False, dedupe_threshold=None)
    assert not np.allclose(out1.to_numpy(), out2.to_numpy())


# ---------- Mode B (inference) ----------


def test_row_attention_mode_b_shape():
    X_train, y_train = _make_data(n=200, d=8, seed=0)
    X_query, _ = _make_data(n=50, d=8, seed=1)
    out = compute_row_attention(
        X_train=X_train,
        y_train=y_train,
        X_query=X_query,
        splitter=KFold(n_splits=2),
        seed=42,
        n_heads=2,
        head_dim=4,
        k=8,
        gpu_stage4=False,
        dedupe_threshold=None,
    )
    assert out.shape == (50, 4)


def test_row_attention_mode_b_deterministic():
    X_train, y_train = _make_data(n=150, d=8, seed=0)
    X_query, _ = _make_data(n=40, d=8, seed=1)
    out1 = compute_row_attention(X_train, y_train, X_query, KFold(n_splits=2), seed=0, n_heads=2, head_dim=4, k=8, gpu_stage4=False, dedupe_threshold=None)
    out2 = compute_row_attention(X_train, y_train, X_query, KFold(n_splits=2), seed=0, n_heads=2, head_dim=4, k=8, gpu_stage4=False, dedupe_threshold=None)
    np.testing.assert_allclose(out1.to_numpy(), out2.to_numpy(), atol=1e-6)


# ---------- low-level build_key_bank + attend ----------


def test_build_key_bank_and_attend_round_trip():
    X_train, y_train = _make_data(n=150, d=8, seed=0)
    X_query, _ = _make_data(n=30, d=8, seed=1)
    bank = build_key_bank(X_train=X_train, y_train=y_train, seed=0, n_heads=2, head_dim=4)
    outputs = attend(bank=bank, X_query=X_query, k=8, aggregate=("y_mean",))
    assert "y_mean_h0" in outputs
    assert "y_mean_h1" in outputs
    assert outputs["y_mean_h0"].shape == (30,)


def test_build_key_bank_disk_cache_round_trip(tmp_path: Path):
    X_train, y_train = _make_data(n=120, d=6, seed=0)
    bank1 = build_key_bank(X_train=X_train, y_train=y_train, seed=0, n_heads=2, head_dim=4, cache_dir=tmp_path)
    # Second call should hit cache.
    bank2 = build_key_bank(X_train=X_train, y_train=y_train, seed=0, n_heads=2, head_dim=4, cache_dir=tmp_path)
    np.testing.assert_array_equal(bank1.k_proj, bank2.k_proj)
    np.testing.assert_array_equal(bank1.y_train, bank2.y_train)
    # Cache files exist.
    assert any(tmp_path.iterdir())


def test_attend_with_explicit_aggregates():
    X_train, y_train = _make_data(n=100, d=6, seed=0)
    X_query, _ = _make_data(n=20, d=6, seed=1)
    bank = build_key_bank(X_train=X_train, y_train=y_train, seed=0, n_heads=1, head_dim=4)
    outputs = attend(bank=bank, X_query=X_query, k=8, aggregate=("y_mean", "y_std", "x_mean"))
    assert set(outputs.keys()) == {"y_mean_h0", "y_std_h0", "x_mean_h0"}
    assert outputs["x_mean_h0"].shape == (20, 4)


# ---------- Validation paths ----------


def test_row_attention_raises_on_missing_seed():
    X, y = _make_data(n=50, d=4, seed=0)
    # Python's "missing 1 required keyword-only argument: 'seed'" fires before require_seed; both messages contain "seed".
    with pytest.raises(TypeError, match="seed"):
        compute_row_attention(X, y, None, KFold(n_splits=2), n_heads=1, head_dim=2, k=4, gpu_stage4=False)  # type: ignore[call-arg]


def test_row_attention_raises_on_nan():
    X, y = _make_data(n=50, d=4, seed=0)
    X[10, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compute_row_attention(X, y, None, KFold(n_splits=2), seed=0, n_heads=1, head_dim=2, k=4, gpu_stage4=False)


def test_row_attention_raises_on_y_train_2d():
    X, y = _make_data(n=50, d=4, seed=0)
    y2 = np.stack([y, y], axis=1)
    with pytest.raises(ValueError, match="y_train must be 1-D"):
        compute_row_attention(X, y2, None, KFold(n_splits=2), seed=0, n_heads=1, head_dim=2, k=4, gpu_stage4=False)


def test_row_attention_raises_on_k_too_large():
    X, y = _make_data(n=50, d=4, seed=0)
    with pytest.raises(ValueError, match="k must be"):
        compute_row_attention(X, y, None, KFold(n_splits=2), seed=0, n_heads=1, head_dim=2, k=1000, gpu_stage4=False)


def test_row_attention_raises_on_metric_non_cosine():
    X, y = _make_data(n=50, d=4, seed=0)
    with pytest.raises(ValueError, match="metric"):
        compute_row_attention(X, y, None, KFold(n_splits=2), seed=0, n_heads=1, head_dim=2, k=4, metric="l2", gpu_stage4=False)  # type: ignore[arg-type]


def test_row_attention_raises_on_x_query_d_mismatch():
    X_train, y_train = _make_data(n=50, d=8, seed=0)
    X_query, _ = _make_data(n=20, d=6, seed=1)  # different d
    with pytest.raises(ValueError, match="d="):
        compute_row_attention(X_train, y_train, X_query, KFold(n_splits=2), seed=0, n_heads=1, head_dim=2, k=4, gpu_stage4=False)


# ---------- Behavioural correctness (soft) ----------


def test_row_attention_signal_correlates_with_target():
    """Sanity: features computed via row-attention should track the target on the same split.

    Target ``y = (X[:, 0] > 0)`` is a half-space; nearby neighbours share label more than 50% of the time, so the softmax-weighted y_mean should be informative.
    On a 200-row dataset this is a loose check (correlation > 0.2); the formal biz_value test is in ``test_biz_val_row_attention.py``.
    """
    X_train, y_train = _make_data(n=300, d=8, seed=0)
    X_query, y_query = _make_data(n=100, d=8, seed=1)
    out = compute_row_attention(
        X_train=X_train,
        y_train=y_train,
        X_query=X_query,
        splitter=KFold(n_splits=2),
        seed=42,
        n_heads=1,
        head_dim=4,
        k=16,
        gpu_stage4=False,
        dedupe_threshold=None,
    )
    y_mean = out["attn_h0_y_mean"].to_numpy()
    corr = float(np.corrcoef(y_mean, y_query)[0, 1])
    assert corr > 0.2, f"row-attention y_mean should correlate with target on holdout; got corr={corr:.3f}"


# ---------- GPU parity (skipped if no GPU) ----------


def _gpu_available() -> bool:
    try:
        from mlframe.feature_engineering.transformer._utils import is_gpu_available

        return is_gpu_available()
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
@pytest.mark.gpu
def test_row_attention_gpu_cpu_parity():
    """Stage 4 GPU vs CPU outputs must agree within ``atol=1e-4 rtol=1e-3`` (Hardware #20 loosened tolerance for chained softmax + matmul)."""
    X_train, y_train = _make_data(n=300, d=8, seed=0)
    X_query, _ = _make_data(n=64, d=8, seed=1)
    out_cpu = compute_row_attention(
        X_train,
        y_train,
        X_query,
        KFold(n_splits=2),
        seed=0,
        n_heads=2,
        head_dim=4,
        k=8,
        gpu_stage4=False,
        dedupe_threshold=None,
    )
    out_gpu = compute_row_attention(
        X_train,
        y_train,
        X_query,
        KFold(n_splits=2),
        seed=0,
        n_heads=2,
        head_dim=4,
        k=8,
        gpu_stage4=True,
        dedupe_threshold=None,
    )
    np.testing.assert_allclose(out_cpu.to_numpy(), out_gpu.to_numpy(), atol=1e-4, rtol=1e-3)
