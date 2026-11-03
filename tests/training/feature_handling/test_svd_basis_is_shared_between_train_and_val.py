"""Dense-only assembly projects val/test onto the SVD fitted on train instead of decomposing each split on its own."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mlframe.training.feature_handling.assembler import HandlerOutput, assemble_for_model


def _block(mat):
    """A wide TF-IDF-like sparse block."""
    return HandlerOutput(column="text", method="tfidf", data=sp.csr_matrix(mat), n_features=mat.shape[1], output_kind="sparse")


def _assemble(mat, fitted=None):
    """Assemble for a dense-only model with the SVD trigger forced low."""
    return assemble_for_model([_block(mat)], model_kind="mlp", svd_default_dim=4, svd_trigger_ncols=10, fitted_svds=fitted)


def test_val_rows_equal_the_train_projection_of_the_same_rows():
    """The first 20 rows are in both train and val: val must give them the exact train-basis coordinates, not a basis refitted on val."""
    rng = np.random.default_rng(0)
    train = rng.random((200, 40))
    val = train[:20]
    train_asm = _assemble(train)
    assert train_asm.fitted_svds
    val_asm = _assemble(val, fitted=train_asm.fitted_svds)
    np.testing.assert_allclose(val_asm.dense_block, train_asm.dense_block[:20], atol=1e-5)


def test_val_with_fewer_rows_than_components_no_longer_fails():
    """Refitting on 3 val rows with 4 components raised; the projection needs no fit."""
    rng = np.random.default_rng(1)
    train_asm = _assemble(rng.random((100, 40)))
    val_asm = _assemble(rng.random((3, 40)), fitted=train_asm.fitted_svds)
    assert val_asm.dense_block.shape == (3, 4)
