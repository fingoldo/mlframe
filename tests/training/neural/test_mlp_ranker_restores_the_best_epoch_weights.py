"""The ranker callback that restores the weights of the best validation epoch."""
from types import SimpleNamespace

import torch
from torch import nn

from mlframe.training.neural.ranker import _BestWeightsCallback


def _trainer(val_loss):
    """A stand-in trainer carrying one validation metric."""
    return SimpleNamespace(sanity_checking=False, callback_metrics={"val_loss": torch.tensor(val_loss)})


def test_last_epoch_weights_are_replaced_by_the_best_epoch_weights():
    """Epoch 1 is best (val_loss 0.2); epochs 2 and 3 are worse and change the weights; training end must bring epoch 1 back."""
    module = nn.Linear(2, 1)
    cb = _BestWeightsCallback()
    with torch.no_grad():
        module.weight.fill_(1.0)
    cb.on_validation_epoch_end(_trainer(0.2), module)
    for value, w in ((0.5, 2.0), (0.9, 3.0)):
        with torch.no_grad():
            module.weight.fill_(w)
        cb.on_validation_epoch_end(_trainer(value), module)
    cb.on_train_end(_trainer(0.9), module)
    assert torch.allclose(module.weight, torch.ones_like(module.weight))


def test_sanity_check_pass_is_ignored():
    """Lightning's pre-training sanity pass must not become the 'best' snapshot."""
    module = nn.Linear(2, 1)
    cb = _BestWeightsCallback()
    cb.on_validation_epoch_end(SimpleNamespace(sanity_checking=True, callback_metrics={"val_loss": torch.tensor(0.0)}), module)
    assert cb.best_state is None
