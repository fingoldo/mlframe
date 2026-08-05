"""_optimizer_wrapper_shared's param_groups/state getters must forward the base optimizer's live
list/dict objects, not a copy -- torch's own Optimizer.add_param_group does
self.param_groups.append(...), which is silently lost against a fresh copy each access."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural._lookahead_optimizer import Lookahead
from mlframe.training.neural._sam_optimizer import SAM


def test_lookahead_param_groups_is_the_live_base_list():
    """Identity check: the wrapper's param_groups IS base_optimizer.param_groups, not a copy."""
    p = torch.nn.Parameter(torch.zeros(4))
    base = torch.optim.AdamW([p], lr=1e-3)
    lh = Lookahead(base, k=5, alpha=0.5)
    assert lh.param_groups is base.param_groups


def test_sam_param_groups_is_the_live_base_list():
    """Identity check: the wrapper's param_groups IS base_optimizer.param_groups, not a copy."""
    p = torch.nn.Parameter(torch.zeros(2))
    base = torch.optim.SGD([p], lr=0.1)
    sam = SAM(base, rho=0.05)
    assert sam.param_groups is base.param_groups


def test_lookahead_add_param_group_reaches_base_optimizer():
    """A new param group added via the wrapper's .param_groups.append(...) (the pattern torch's
    own Optimizer.add_param_group uses) must actually register on the base optimizer -- lost
    against a fresh dict()/list() copy returned on every property access."""
    p1 = torch.nn.Parameter(torch.zeros(4))
    p2 = torch.nn.Parameter(torch.zeros(3))
    base = torch.optim.AdamW([p1], lr=1e-3)
    lh = Lookahead(base, k=5, alpha=0.5)

    lh.param_groups.append({"params": [p2], "lr": 5e-4})

    assert len(base.param_groups) == 2, "appended group did not reach the base optimizer"
    assert base.param_groups[1]["params"][0] is p2
