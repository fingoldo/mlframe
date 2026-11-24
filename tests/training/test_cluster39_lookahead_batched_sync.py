"""Cluster 39 regression: the lookahead slow/fast sync is batched and matches the per-tensor result (audit 2026-11-20, finding 39-F58).

The sync used a Python loop with per-parameter ``lerp_`` / ``copy_``. It now issues one ``torch._foreach_lerp_`` and one ``torch._foreach_copy_``.
The test compares the result of one Lookahead step against a reference: an identical model takes one plain SGD step (the fast weights), and the
expected parameters are slow + alpha * (fast - slow) with slow equal to the initial weights.
"""

import copy

import torch

from mlframe.training.neural._lookahead_optimizer import Lookahead


def test_sync_matches_reference_lerp_then_copy():
    """After one Lookahead step each parameter equals slow + alpha * (fast - slow) (39-F58)."""
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    reference = copy.deepcopy(model)
    initial = [p.detach().clone() for p in model.parameters()]
    x = torch.ones(4, 3)

    opt = Lookahead(torch.optim.SGD(model.parameters(), lr=0.1), k=1, alpha=0.5)
    model(x).sum().backward()
    opt.step()

    reference(x).sum().backward()
    torch.optim.SGD(reference.parameters(), lr=0.1).step()
    for p, slow0, p_ref in zip(model.parameters(), initial, reference.parameters()):
        expected = slow0 + 0.5 * (p_ref.detach() - slow0)
        torch.testing.assert_close(p.detach(), expected)
