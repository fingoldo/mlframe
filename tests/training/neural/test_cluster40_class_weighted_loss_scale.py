"""Cluster 40 regression: the weighted classification loss keeps the unweighted loss scale (audit 2026-11-20, finding 40-F30)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from mlframe.training.neural._recurrent_torch_model import RecurrentTorchModel


def _fake_model(class_weight):
    """Minimal stand-in carrying the attributes _compute_weighted_loss reads."""
    return SimpleNamespace(
        is_regression=False,
        task_type="multiclass",
        class_weight=class_weight,
        _loss_fn_mean=torch.nn.CrossEntropyLoss(weight=class_weight),
        _loss_fn_unreduced=torch.nn.CrossEntropyLoss(weight=class_weight, reduction="none"),
    )


def test_unit_sample_weights_match_the_unweighted_mean():
    """With all sample weights equal to 1 the weighted path equals CrossEntropy(weight) mean (40-F30)."""
    torch.manual_seed(0)
    logits = torch.randn(12, 3)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 1])
    class_weight = torch.tensor([0.5, 2.0, 1.0])
    model = _fake_model(class_weight)
    unweighted = RecurrentTorchModel._compute_weighted_loss(model, logits, labels, None)
    weighted = RecurrentTorchModel._compute_weighted_loss(model, logits, labels, torch.ones(12))
    assert torch.allclose(weighted, unweighted, atol=1e-6), (float(weighted), float(unweighted))
