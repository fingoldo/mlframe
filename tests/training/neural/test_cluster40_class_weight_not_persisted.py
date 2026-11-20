"""Audit 2026-11-20 finding 40-F8: class weights are not a state_dict key, so a class-weighted model round-trips through load_state_dict."""

import pytest

torch = pytest.importorskip("torch")

from mlframe.training.neural._recurrent_config import InputMode, RecurrentConfig
from mlframe.training.neural.recurrent import RecurrentTorchModel


def _model_with_class_weight():
    """A classification model carrying class weights."""
    cfg = RecurrentConfig(input_mode=InputMode.FEATURES_ONLY, hidden_size=8, num_layers=1, mlp_hidden_sizes=(8,), max_epochs=1, batch_size=4, precision="32-true")
    return RecurrentTorchModel(config=cfg, seq_input_size=4, aux_input_size=4, is_regression=False, class_weight=torch.tensor([1.0, 2.0]))


def test_class_weight_is_not_in_the_state_dict_but_stays_on_the_model():
    """The weights are not persisted (no state_dict key), yet the model still holds them for training."""
    model = _model_with_class_weight()
    assert "class_weight" not in model.state_dict()
    assert torch.equal(model.class_weight, torch.tensor([1.0, 2.0]))
