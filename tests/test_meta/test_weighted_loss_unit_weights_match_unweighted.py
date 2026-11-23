"""Invariant: with every sample weight equal to 1, each weighted loss path equals its unweighted loss (found while fixing audit 40-F30).

The 40-F30 bug was a weighted path whose normalisation differed from the unweighted one once class weights were set. This property
covers every task branch of the recurrent loss, so a new branch or a normaliser change that breaks it fails here.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from mlframe.training.neural._recurrent_torch_model import RecurrentTorchModel


def _model(*, is_regression, task_type, class_weight=None):
    """Stand-in carrying the attributes ``_compute_weighted_loss`` reads, with the same loss modules the trainer builds."""
    if is_regression:
        mean_fn, unreduced_fn = torch.nn.MSELoss(), torch.nn.MSELoss(reduction="none")
    elif task_type == "multilabel":
        mean_fn, unreduced_fn = torch.nn.BCEWithLogitsLoss(), torch.nn.BCEWithLogitsLoss(reduction="none")
    else:
        mean_fn = torch.nn.CrossEntropyLoss(weight=class_weight)
        unreduced_fn = torch.nn.CrossEntropyLoss(weight=class_weight, reduction="none")
    return SimpleNamespace(
        is_regression=is_regression,
        task_type=task_type,
        class_weight=class_weight,
        _loss_fn_mean=mean_fn,
        _loss_fn_unreduced=unreduced_fn,
    )


_CASES = {
    "regression": dict(is_regression=True, task_type="regression"),
    "multiclass": dict(is_regression=False, task_type="multiclass"),
    "multiclass_class_weights": dict(is_regression=False, task_type="multiclass", class_weight=torch.tensor([0.5, 2.0, 1.0])),
    "multilabel": dict(is_regression=False, task_type="multilabel"),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_unit_sample_weights_reproduce_the_unweighted_loss(case):
    """Each path: weighted with ones == unweighted (the invariant behind 40-F30)."""
    torch.manual_seed(0)
    cfg = _CASES[case]
    model = _model(**cfg)
    if cfg["is_regression"]:
        logits = torch.randn(12, 1)
        labels = torch.randn(12)
    elif cfg["task_type"] == "multilabel":
        logits = torch.randn(12, 3)
        labels = torch.randint(0, 2, (12, 3)).float()
    else:
        logits = torch.randn(12, 3)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 1])
    unweighted = RecurrentTorchModel._compute_weighted_loss(model, logits, labels, None)
    weighted = RecurrentTorchModel._compute_weighted_loss(model, logits, labels, torch.ones(12))
    assert torch.allclose(weighted, unweighted, atol=1e-5), (case, float(weighted), float(unweighted))
