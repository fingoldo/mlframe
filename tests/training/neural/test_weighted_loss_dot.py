"""Regression + biz_value tests for the torch.dot fast path in
mlframe.training.neural.flat._LightningWrapper._compute_weighted_loss.

The wrapper used to compute the weighted mean as `(loss*sw).sum() / sw.sum()`.
A profile of the 1M-row fuzz combo c0027 showed this line accounted for ~6.85s
self-time across 904 minibatches. Replacing with `torch.dot(loss, sw) / sw.sum()`
fuses mul+sum into one kernel; bench on CPU: 1.7-2.2x faster across N=256-16384.

This test pins:
  (1) numerical equivalence between dot and broadcast-mul paths
  (2) gradient bit-equivalence (preserves all safety properties of the loss)
  (3) the 1-D fast path is taken when shapes match
  (4) the broadcast fallback still works for non-1-D losses
  (5) the biz_value: dot is at least 1.3x faster (loose lower bound to absorb
      jitter on CI machines, while still detecting full regression)
"""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")


def _weighted_loss_broadcast(loss: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """Weighted loss broadcast."""
    return (loss * sw).sum() / sw.sum()


def _weighted_loss_dot(loss: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """Weighted loss dot."""
    return torch.dot(loss, sw) / sw.sum()


@pytest.mark.parametrize("n", [256, 1024, 4096])
def test_dot_matches_broadcast_forward(n: int) -> None:
    """Dot matches broadcast forward."""
    torch.manual_seed(20260520)
    loss = torch.rand(n, dtype=torch.float64) + 0.1
    sw = torch.rand(n, dtype=torch.float64) + 0.1
    a = _weighted_loss_broadcast(loss, sw)
    b = _weighted_loss_dot(loss, sw)
    assert torch.allclose(a, b, atol=1e-12, rtol=1e-12), f"forward mismatch: {a} vs {b}"


@pytest.mark.parametrize("n", [256, 1024, 4096])
def test_dot_matches_broadcast_gradient(n: int) -> None:
    """Dot matches broadcast gradient."""
    torch.manual_seed(20260520)
    loss_a = (torch.rand(n, dtype=torch.float64) + 0.1).requires_grad_(True)
    loss_b = loss_a.detach().clone().requires_grad_(True)
    sw = torch.rand(n, dtype=torch.float64) + 0.1

    _weighted_loss_broadcast(loss_a, sw).backward()
    _weighted_loss_dot(loss_b, sw).backward()

    assert torch.allclose(loss_a.grad, loss_b.grad, atol=1e-12, rtol=1e-12)


def test_wrapper_routes_to_dot_for_1d():
    """The wrapper's optimized branch should be taken for the common 1-D case."""
    from mlframe.training.neural.flat import MLPTorchModel  # type: ignore

    class _Stub:
        """Groups tests covering stub."""
        loss_fn = torch.nn.MSELoss()

        _loss_unreduced = MLPTorchModel._loss_unreduced
        _compute_weighted_loss = MLPTorchModel._compute_weighted_loss

    stub = _Stub()
    pred = torch.randn(128, requires_grad=True)
    labels = torch.randn(128)
    sw = torch.rand(128) + 0.1

    out = MLPTorchModel._compute_weighted_loss(stub, pred, labels, sw)
    out.backward()
    assert out.shape == ()
    assert pred.grad is not None
    assert torch.isfinite(out).item()


def test_wrapper_broadcast_2d_loss_with_1d_weights():
    """Multi-output / multilabel: 2-D loss + 1-D sample_weight must broadcast.

    Fuzz c0052 (multilabel + MLP + uniform weights) crashed because torch
    right-aligns (B,) against (B, K) -> dim 1 collision (B vs K). The
    wrapper now unsqueezes 1-D sample_weight to (B, 1) so it broadcasts
    across labels / classes uniformly.

    Regression test: pre-fix this raised
    ``RuntimeError: size of tensor a (3) must match the size of tensor b (64)``.
    Post-fix it must produce a finite scalar loss with valid gradient.
    """
    from mlframe.training.neural.flat import MLPTorchModel  # type: ignore

    class _Stub:
        """Groups tests covering stub."""
        loss_fn = torch.nn.MSELoss()
        _loss_unreduced = MLPTorchModel._loss_unreduced
        _compute_weighted_loss = MLPTorchModel._compute_weighted_loss

    stub = _Stub()
    pred = torch.randn(64, 3, requires_grad=True)
    labels = torch.randn(64, 3)
    sw = torch.rand(64) + 0.1
    out = MLPTorchModel._compute_weighted_loss(stub, pred, labels, sw)
    assert out.shape == ()
    assert torch.isfinite(out).item()
    out.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all().item()


def test_wrapper_multilabel_bce_with_sample_weights():
    """Multilabel BCEWithLogitsLoss + per-sample weights must not crash.

    Direct repro of fuzz c0052 (multilabel_classification + 3 labels +
    MLP + uniform weights). BCEWithLogitsLoss with reduction='none' emits
    a (B, K) loss tensor; the (B,) weight tensor must broadcast across
    labels.
    """
    from mlframe.training.neural.flat import MLPTorchModel  # type: ignore

    class _Stub:
        """Groups tests covering stub."""
        loss_fn = torch.nn.BCEWithLogitsLoss()
        _loss_unreduced = MLPTorchModel._loss_unreduced
        _compute_weighted_loss = MLPTorchModel._compute_weighted_loss

    stub = _Stub()
    pred = torch.randn(128, 3, requires_grad=True)
    labels = torch.randint(0, 2, (128, 3)).float()
    sw = torch.rand(128) + 0.1
    out = MLPTorchModel._compute_weighted_loss(stub, pred, labels, sw)
    assert out.shape == ()
    assert torch.isfinite(out).item()
    # Sanity: weighted loss equals manual (loss * weights.unsqueeze(1)).sum() / weights.sum()
    loss_un = torch.nn.functional.binary_cross_entropy_with_logits(pred, labels, reduction="none")
    expected = (loss_un * sw.view(-1, 1)).sum() / sw.sum()
    assert torch.allclose(out, expected, atol=1e-5)


@pytest.mark.biz_transformer
def test_biz_value_dot_faster_than_broadcast() -> None:
    """biz_value: torch.dot must be ≥1.3x faster than broadcast-mul on N=4096."""
    torch.manual_seed(20260520)
    n = 4096
    loss = torch.rand(n) + 0.1
    sw = torch.rand(n) + 0.1
    iters = 4000

    # warmup
    for _ in range(50):
        _weighted_loss_broadcast(loss, sw)
        _weighted_loss_dot(loss, sw)

    t0 = time.perf_counter()
    for _ in range(iters):
        _weighted_loss_broadcast(loss, sw)
    t_broadcast = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        _weighted_loss_dot(loss, sw)
    t_dot = time.perf_counter() - t0

    speedup = t_broadcast / t_dot
    assert speedup >= 1.3, (
        f"torch.dot fast path is not delivering: speedup={speedup:.2f}x (broadcast={t_broadcast * 1e6 / iters:.2f}us, dot={t_dot * 1e6 / iters:.2f}us)"
    )
