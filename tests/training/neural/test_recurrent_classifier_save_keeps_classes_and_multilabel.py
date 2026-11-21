"""Regression tests for the audit findings in this module."""
import pytest
"""A saved recurrent classifier keeps its label mapping and multilabel flag across save() and load() (audit 2026-11-20, finding 40-F1)."""
import numpy as np

pytest.importorskip("lightning")
from mlframe.training.neural._recurrent_config import InputMode, RecurrentConfig
from mlframe.training.neural.recurrent import RecurrentClassifierWrapper, RecurrentTorchModel


def _cfg():
    """Tiny feature-only config so the test builds a model without training."""
    return RecurrentConfig(
        input_mode=InputMode.FEATURES_ONLY,
        hidden_size=8,
        num_layers=1,
        mlp_hidden_sizes=(8,),
        max_epochs=1,
        batch_size=4,
        precision="32-true",
    )


def test_classes_survive_save_and_load(tmp_path):
    """predict() maps argmax positions through classes_, so a loaded classifier without it raises NotFittedError."""
    cfg = _cfg()
    wrapper = RecurrentClassifierWrapper(config=cfg)
    wrapper.classes_ = np.array([3, 7, 9])
    wrapper.model = RecurrentTorchModel(config=cfg, seq_input_size=4, aux_input_size=4, is_regression=False)
    wrapper._aux_input_size = 4
    wrapper._seq_input_size = 4
    path = tmp_path / "clf.pt"
    wrapper.save(path)
    loaded = RecurrentClassifierWrapper.load(path)
    np.testing.assert_array_equal(loaded.classes_, np.array([3, 7, 9]))
    assert loaded._is_multilabel is False


def test_multilabel_flag_and_task_type_survive_save_and_load(tmp_path):
    """A multilabel model reloads with the BCE task type instead of silently falling back to the single-label loss."""
    cfg = _cfg()
    wrapper = RecurrentClassifierWrapper(config=cfg)
    wrapper._is_multilabel = True
    wrapper.model = RecurrentTorchModel(config=cfg, seq_input_size=4, aux_input_size=4, is_regression=False, task_type="multilabel")
    wrapper._aux_input_size = 4
    wrapper._seq_input_size = 4
    path = tmp_path / "ml.pt"
    wrapper.save(path)
    loaded = RecurrentClassifierWrapper.load(path)
    assert loaded._is_multilabel is True
    assert loaded.model.task_type == "multilabel"
