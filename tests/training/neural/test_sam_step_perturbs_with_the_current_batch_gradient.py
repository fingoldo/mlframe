"""SAM.step(closure) runs the closure at the original weights before the ascent perturbation."""
import torch

from mlframe.training.neural._sam_optimizer import SAM


def _run(use_sam):
    """One optimisation step on L = 0.5 * a * w^2 (a=4) from w = 1, starting with NO gradients, like the first batch under Lightning."""
    w = torch.nn.Parameter(torch.tensor([1.0]))
    base = torch.optim.SGD([w], lr=0.1)
    opt = SAM(base, rho=0.5) if use_sam else base

    def closure():
        """Loss and backward, as the Lightning closure does."""
        opt.zero_grad()
        loss = 2.0 * (w**2).sum() + 0.0 * w.sum()
        loss.backward()
        return loss

    opt.step(closure)
    return float(w.detach())


def test_first_step_with_no_prior_gradient_still_uses_the_perturbed_gradient():
    """grad at w=1 is 4, the ascent step moves to w=1.5 where grad is 6: SAM ends at 1 - 0.1*6 = 0.4, plain SGD at 0.6. Without the first closure call no gradient existed, the perturbation was skipped and SAM reproduced SGD's 0.6."""
    assert abs(_run(False) - 0.6) < 1e-5
    assert abs(_run(True) - 0.4) < 1e-4
