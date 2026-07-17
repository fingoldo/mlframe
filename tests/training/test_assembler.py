"""
Tests for the phase-E assembler + routing layer.

Coverage (round-3 audits):
  * Routing table (sparse-aware vs dense-only) -- single source of truth.
  * Auto-SVD trigger fires only on dense-only models above threshold;
    WARN message format pinned (round-3 user-confirmed).
  * Multi-handler concat: insertion order, disambiguated names,
    same-col + same-method -> ValueError (round-3 A8).
  * Two-track output for sparse-aware models (round-3 CC5: never
    densify TF-IDF for XGB/CB/LGB/linear).
  * Single-track dense output for dense-only models with mixed
    sparse + dense + numeric blocks.
  * SVD pinned ``n_iter=5`` so cross-version sklearn drift caught
    (round-3 R3-08).
  * HGB tfidf_max_features cap WARN+cap (round-3 A18).
"""

from __future__ import annotations

import logging
import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from mlframe.training.feature_handling import (
    AssembledMatrix,
    DENSE_ONLY_MODELS,
    HandlerOutput,
    SPARSE_AWARE_MODELS,
    accepts_sparse,
    assemble_for_model,
    assembled_column_names,
    hgb_max_features_cap,
    is_dense_only,
    should_apply_svd,
)

try:
    from tests.conftest import fast_subset
except ImportError:  # pragma: no cover

    def fast_subset(values, **_):
        return list(values)


# --fast collapses the routing matrix to one representative per arm. ``cb`` is the
# canonical sparse-aware booster; ``mlp`` is the canonical dense-only neural model.
_SPARSE_AWARE_FAST = fast_subset(["cb", "xgb", "lgb", "linear", "ridge", "sgd"], representative="cb")
_DENSE_ONLY_FAST = fast_subset(["hgb", "rf", "ngb", "mlp", "recurrent", "tabnet"], representative="mlp")


# =====================================================================
# 1. Routing table
# =====================================================================


class TestRouting:
    @pytest.mark.parametrize("kind", _SPARSE_AWARE_FAST)
    def test_sparse_aware_models(self, kind):
        assert accepts_sparse(kind) is True
        assert is_dense_only(kind) is False

    @pytest.mark.parametrize("kind", _DENSE_ONLY_FAST)
    def test_dense_only_models(self, kind):
        assert accepts_sparse(kind) is False
        assert is_dense_only(kind) is True

    def test_sets_disjoint(self):
        # Sanity: a model is either sparse-aware OR dense-only, never both.
        assert SPARSE_AWARE_MODELS.isdisjoint(DENSE_ONLY_MODELS)


class TestSvdTrigger:
    def test_sparse_aware_never_triggers_svd(self):
        assert should_apply_svd("xgb", 100_000) is False
        assert should_apply_svd("cb", 100_000) is False

    def test_dense_only_above_threshold_triggers(self):
        assert should_apply_svd("hgb", 1024) is True
        assert should_apply_svd("mlp", 600) is True

    def test_dense_only_below_threshold_does_not(self):
        assert should_apply_svd("hgb", 500) is False
        assert should_apply_svd("mlp", 100) is False

    def test_threshold_override(self):
        assert should_apply_svd("hgb", 200, threshold=100) is True
        assert should_apply_svd("hgb", 50, threshold=100) is False


# =====================================================================
# 2. HGB tfidf cap
# =====================================================================


class TestHgbCap:
    def test_hgb_below_cap_unchanged(self, caplog):
        caplog.set_level(logging.WARNING)
        out = hgb_max_features_cap("hgb", 200)
        assert out == 200
        # No warning when under cap
        assert not [r for r in caplog.records if "Capping" in r.getMessage()]

    def test_hgb_above_cap_warns_and_caps(self, caplog):
        # Cap derived from the production constant so default-tuning of the cap doesn't break
        # this test (the contract is "input above cap -> output equals cap, WARN names both
        # numbers", not "output equals literal 500").
        from mlframe.training.feature_handling.routing import HGB_TFIDF_MAX_FEATURES_CAP

        caplog.set_level(logging.WARNING)
        requested = HGB_TFIDF_MAX_FEATURES_CAP * 10  # well above the cap
        out = hgb_max_features_cap("hgb", requested)
        assert out == HGB_TFIDF_MAX_FEATURES_CAP
        msgs = [r.getMessage() for r in caplog.records]
        assert any(f"Capping to {HGB_TFIDF_MAX_FEATURES_CAP}" in m for m in msgs)
        assert any(str(requested) in m for m in msgs)

    def test_hgb_allow_high_bypasses(self, caplog):
        caplog.set_level(logging.INFO)
        out = hgb_max_features_cap("hgb", 5000, allow_high=True)
        assert out == 5000
        msgs = [r.getMessage() for r in caplog.records]
        assert any("bypassed" in m for m in msgs)

    def test_non_hgb_pass_through(self):
        assert hgb_max_features_cap("xgb", 5000) == 5000
        assert hgb_max_features_cap("mlp", 5000) == 5000


# =====================================================================
# 3. Single-handler assembly
# =====================================================================


class TestSingleHandler:
    def test_one_sparse_block_xgb_two_track(self):
        sp = csr_matrix(np.eye(10, dtype=np.float32))
        b = HandlerOutput(column="txt", method="tfidf", data=sp, n_features=10, output_kind="sparse")
        asm = assemble_for_model([b], model_kind="xgb")
        assert asm.is_two_track is False  # no dense block
        assert asm.sparse_block is not None
        assert asm.sparse_block.shape == (10, 10)
        assert asm.dense_block is None
        assert len(asm.feature_names) == 10
        assert all(n.startswith("txt__tfidf__") for n in asm.feature_names)

    def test_one_dense_block_xgb(self):
        d = np.zeros((5, 8), dtype=np.float32)
        b = HandlerOutput(
            column="txt",
            method="frozen_text_embedding",
            data=d,
            n_features=8,
            output_kind="dense",
        )
        asm = assemble_for_model([b], model_kind="xgb")
        assert asm.sparse_block is None
        assert asm.dense_block is not None
        assert asm.dense_block.shape == (5, 8)
        assert all(n.startswith("txt__frozen_emb__") for n in asm.feature_names)

    def test_one_sparse_block_hgb_densifies_under_threshold(self):
        sp = csr_matrix(np.eye(10, dtype=np.float32))  # 10 cols < 512 trigger
        b = HandlerOutput(column="txt", method="tfidf", data=sp, n_features=10, output_kind="sparse")
        asm = assemble_for_model([b], model_kind="hgb")
        assert asm.sparse_block is None
        assert asm.dense_block is not None
        assert asm.dense_block.shape == (10, 10)

    def test_one_sparse_block_hgb_svd_above_threshold(self, caplog):
        caplog.set_level(logging.WARNING)
        rng = np.random.RandomState(0)
        # n_rows must exceed n_components for TruncatedSVD; 200 rows + svd_dim=64 OK.
        sp = csr_matrix(rng.randn(200, 800).astype(np.float32))  # 800 > 512
        b = HandlerOutput(column="big", method="tfidf", data=sp, n_features=800, output_kind="sparse")
        asm = assemble_for_model([b], model_kind="hgb", svd_default_dim=64)
        assert asm.dense_block.shape == (200, 64)
        assert any("Auto-applying TruncatedSVD" in r.getMessage() for r in caplog.records)


# =====================================================================
# 4. Multi-handler concat + disambiguated names
# =====================================================================


class TestMultiHandler:
    def test_two_handlers_same_col_different_methods(self):
        """Round-3 A8: same col + different methods -> both blocks
        with disambiguated names."""
        rng = np.random.RandomState(0)
        sp1 = csr_matrix(rng.rand(5, 3).astype(np.float32))
        sp2 = csr_matrix(rng.rand(5, 4).astype(np.float32))
        b1 = HandlerOutput(column="desc", method="tfidf", data=sp1, n_features=3, output_kind="sparse")
        b2 = HandlerOutput(column="desc", method="hashing", data=sp2, n_features=4, output_kind="sparse")
        asm = assemble_for_model([b1, b2], model_kind="xgb")
        assert asm.sparse_block.shape == (5, 7)  # 3 + 4
        # Names disambiguated by method short form.
        names = asm.feature_names
        assert sum(n.startswith("desc__tfidf__") for n in names) == 3
        assert sum(n.startswith("desc__hash__") for n in names) == 4

    def test_same_col_same_method_raises(self):
        """Round-3 A8: same col + same method = config error."""
        sp = csr_matrix(np.eye(3, dtype=np.float32))
        b1 = HandlerOutput(column="desc", method="tfidf", data=sp, n_features=3, output_kind="sparse")
        b2 = HandlerOutput(column="desc", method="tfidf", data=sp, n_features=3, output_kind="sparse")
        with pytest.raises(ValueError, match="duplicate handler"):
            assemble_for_model([b1, b2], model_kind="xgb")

    def test_insertion_order_preserved(self):
        """Output column order matches input insertion order."""
        # Both sparse blocks must share row count for hstack to work.
        sp_a = csr_matrix(np.zeros((5, 3), dtype=np.float32))
        sp_b = csr_matrix(np.zeros((5, 2), dtype=np.float32))
        b_a = HandlerOutput(column="a", method="tfidf", data=sp_a, n_features=3, output_kind="sparse")
        b_b = HandlerOutput(column="b", method="tfidf", data=sp_b, n_features=2, output_kind="sparse")
        # Order: a then b
        asm_ab = assemble_for_model([b_a, b_b], model_kind="xgb")
        # Order: b then a
        asm_ba = assemble_for_model([b_b, b_a], model_kind="xgb")
        assert asm_ab.feature_names != asm_ba.feature_names
        assert asm_ab.feature_names[0].startswith("a__")
        assert asm_ba.feature_names[0].startswith("b__")

    def test_xgb_mixed_sparse_dense_two_track(self):
        """Round-3 CC5: XGB gets two-track (sparse_block, dense_block);
        TF-IDF NOT densified."""
        sp = csr_matrix(np.eye(5, dtype=np.float32))
        d = np.zeros((5, 4), dtype=np.float32)
        b1 = HandlerOutput(column="txt", method="tfidf", data=sp, n_features=5, output_kind="sparse")
        b2 = HandlerOutput(column="txt", method="frozen_text_embedding", data=d, n_features=4, output_kind="dense")
        asm = assemble_for_model([b1, b2], model_kind="xgb")
        assert asm.is_two_track is True
        assert asm.sparse_block.shape == (5, 5)
        assert asm.dense_block.shape == (5, 4)
        # Sparse block names come first, then dense block names.
        assert asm.feature_names[0].startswith("txt__tfidf__")
        assert asm.feature_names[5].startswith("txt__frozen_emb__")

    def test_hgb_mixed_sparse_dense_concat(self):
        """Dense-only path: sparse densified (or SVD'd), then
        all-dense concatenation."""
        sp = csr_matrix(np.eye(10, dtype=np.float32))  # 10 cols < threshold -> densify
        d = np.zeros((10, 4), dtype=np.float32)
        b1 = HandlerOutput(column="txt", method="tfidf", data=sp, n_features=10, output_kind="sparse")
        b2 = HandlerOutput(column="txt", method="frozen_text_embedding", data=d, n_features=4, output_kind="dense")
        asm = assemble_for_model([b1, b2], model_kind="hgb")
        assert asm.sparse_block is None
        assert asm.dense_block.shape == (10, 14)


class TestNumericBlock:
    def test_numeric_block_appended_xgb(self):
        sp = csr_matrix(np.eye(5, dtype=np.float32))
        b = HandlerOutput(column="txt", method="tfidf", data=sp, n_features=5, output_kind="sparse")
        num = np.zeros((5, 3), dtype=np.float32)
        asm = assemble_for_model([b], model_kind="xgb", numeric_block=num, numeric_feature_names=["age", "score", "n"])
        assert asm.dense_block.shape == (5, 3)
        assert asm.feature_names[-3:] == ["age", "score", "n"]

    def test_numeric_block_only_no_handlers(self):
        num = np.zeros((10, 2), dtype=np.float32)
        asm = assemble_for_model([], model_kind="hgb", numeric_block=num, numeric_feature_names=["x", "y"])
        assert asm.dense_block.shape == (10, 2)
        assert asm.feature_names == ["x", "y"]


# =====================================================================
# 5. SVD pinned solver params (round-3 R3-08)
# =====================================================================


class TestSvdPinnedSolver:
    def test_svd_n_iter_pinned(self):
        """Round-3 R3-08: pinned ``n_iter=5`` so cross-version sklearn
        drift detected. Inspect the assembler's svd call signature
        via patching."""
        from unittest import mock

        rng = np.random.RandomState(0)
        big_sp = csr_matrix(rng.rand(100, 1024).astype(np.float32))
        b = HandlerOutput(column="big", method="tfidf", data=big_sp, n_features=1024, output_kind="sparse")

        with mock.patch("mlframe.training.feature_handling.assembler.TruncatedSVD") as mock_svd_cls:
            instance = mock.MagicMock()
            instance.fit_transform.return_value = np.zeros((100, 64), dtype=np.float32)
            mock_svd_cls.return_value = instance
            assemble_for_model([b], model_kind="hgb", svd_default_dim=64)
        # Args
        kwargs = mock_svd_cls.call_args.kwargs
        assert kwargs["n_iter"] == 5
        assert kwargs["power_iteration_normalizer"] == "auto"


# =====================================================================
# 6. Empty assembly
# =====================================================================


class TestEmptyAssembly:
    def test_no_blocks_no_numeric_returns_empty(self):
        asm = assemble_for_model([], model_kind="xgb")
        assert asm.sparse_block is None
        assert asm.dense_block is None
        assert asm.feature_names == []


# =====================================================================
# 7. assembled_column_names public API
# =====================================================================


class TestPublicApi:
    def test_assembled_column_names_returns_copy(self):
        sp = csr_matrix(np.eye(3, dtype=np.float32))
        b = HandlerOutput(column="x", method="tfidf", data=sp, n_features=3, output_kind="sparse")
        asm = assemble_for_model([b], model_kind="xgb")
        names = assembled_column_names(asm)
        names.append("mutated")
        assert "mutated" not in asm.feature_names
