"""TRAINING_NEURAL-3: a torch LR scheduler's in-place param_groups mutation must reach the real
Muon/AdamW sub-optimizers, not a disconnected placeholder group."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural._muon_optimizer import MuonAdamWHybrid


def test_lr_scheduler_step_updates_wrapped_muon_and_adamw_lrs():
    """StepLR halving the outer lr must actually change self._muon/self._adamw's real
    param_groups[0]['lr'], not just an unread placeholder group."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.Linear(16, 4))
    opt = MuonAdamWHybrid(net.parameters(), lr=1e-2, muon_lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

    muon_lr_before = opt._muon.param_groups[0]["lr"]
    adamw_lr_before = opt._adamw.param_groups[0]["lr"]
    assert muon_lr_before == 0.1
    assert adamw_lr_before == 1e-2

    scheduler.step()

    assert opt._muon.param_groups[0]["lr"] == muon_lr_before * 0.5, "scheduler step did not reach the real Muon param_groups"
    assert opt._adamw.param_groups[0]["lr"] == adamw_lr_before * 0.5, "scheduler step did not reach the real AdamW param_groups"


def test_param_groups_getter_reflects_live_sub_optimizer_state():
    """opt.param_groups must be a live concatenated view, not a snapshot frozen at construction."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.Linear(16, 4))
    opt = MuonAdamWHybrid(net.parameters(), lr=1e-2, muon_lr=0.1)

    opt._muon.param_groups[0]["lr"] = 0.5
    opt._adamw.param_groups[0]["lr"] = 0.05

    lrs = [g["lr"] for g in opt.param_groups]
    assert 0.5 in lrs
    assert 0.05 in lrs
