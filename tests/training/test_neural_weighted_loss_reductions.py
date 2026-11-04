"""Weighted neural losses: the multilabel flat loss equals the unweighted mean for uniform weights, and a float16 all-zero-weight batch gives 0, not nan."""
from types import SimpleNamespace

import torch
from torch import nn

from mlframe.training.neural._flat_torch_module._flat_torch_loss import _LossMixin
from mlframe.training.neural._recurrent_torch_model import RecurrentTorchModel


def _flat_self():
    """A stand-in for the flat module: a BCE-with-logits loss and the unreduced helper bound to it."""
    ns = SimpleNamespace(loss_fn=nn.BCEWithLogitsLoss())
    ns._loss_unreduced = lambda p, y: _LossMixin._loss_unreduced(ns, p, y)
    return ns


def test_multilabel_weighted_loss_is_not_k_times_the_unweighted_one():
    """K = 5 labels, uniform weights: the weighted loss must equal the plain mean-reduced loss."""
    ns = _flat_self()
    logits = torch.randn(4, 5)
    labels = (torch.rand(4, 5) > 0.5).float()
    plain = _LossMixin._compute_weighted_loss(ns, logits, labels, None)
    weighted = _LossMixin._compute_weighted_loss(ns, logits, labels, torch.ones(4))
    assert torch.allclose(plain, weighted, atol=1e-6), (float(plain), float(weighted))


def test_float16_batch_of_zero_weights_gives_zero_loss_not_nan():
    """The 1e-12 floor used to round to 0 in float16, so 0 / 0 reached the optimizer as nan."""
    ns = SimpleNamespace(is_regression=True, _loss_fn_unreduced=nn.MSELoss(reduction="none"), _loss_fn_mean=nn.MSELoss())
    logits = torch.randn(3, 1).half()
    labels = torch.randn(3).half()
    loss = RecurrentTorchModel._compute_weighted_loss(ns, logits, labels, torch.zeros(3).half())
    assert torch.isfinite(loss) and float(loss) == 0.0
