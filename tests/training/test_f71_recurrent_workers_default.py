"""F-71 (2026-05-31): RecurrentConfig.num_workers default is OS-aware.

Windows: 0 (spawn semantics expensive).
Linux / macOS: 2 (fork is cheap; F-46/F-51/F-52 make per-worker cost
amortize well).

The default is exposed via a ``field(default_factory=...)`` so a user
can still override via RecurrentConfig(num_workers=...).
"""
from __future__ import annotations

import os

import pytest

from mlframe.training.neural._recurrent_config import (
    RecurrentConfig, _default_num_workers,
)


def test_default_num_workers_helper_is_os_aware():
    """Pure helper test independent of dataclass plumbing."""
    expected = 0 if os.name == "nt" else 2
    assert _default_num_workers() == expected


def test_recurrent_config_num_workers_default_matches_os_helper():
    """The dataclass field MUST consult the OS-aware factory at every
    instantiation (not a class-level cached 0)."""
    cfg = RecurrentConfig()
    expected = 0 if os.name == "nt" else 2
    assert cfg.num_workers == expected


def test_recurrent_config_num_workers_explicit_override_preserved():
    """User-supplied num_workers MUST win over the OS-aware default."""
    cfg = RecurrentConfig(num_workers=7)
    assert cfg.num_workers == 7


def test_recurrent_config_two_instances_get_independent_defaults():
    """Sanity that the default_factory is called per-instance, not
    once at class-creation time (catches the common dataclass mutable-
    default pitfall the OS-helper alternative could re-introduce)."""
    cfg_a = RecurrentConfig()
    cfg_b = RecurrentConfig()
    # Same OS in this process, so same default value:
    assert cfg_a.num_workers == cfg_b.num_workers
    # But each was independently evaluated -- mutate one's num_workers
    # and verify the other's is untouched.
    cfg_a.num_workers = 999
    assert cfg_b.num_workers != 999


# --- F-71b (2026-05-31): MLP TorchDataModule mirrors the OS-aware default --


def test_mlp_dataloader_num_workers_os_aware_default():
    """F-71b: When the user doesn't pass num_workers in dataloader_params,
    _create_dataloader fills it in with 0 on Windows / 2 on Linux+macOS.
    Mirrors F-71's RecurrentConfig behaviour for the MLP path."""
    import numpy as np
    import torch
    from mlframe.training.neural.data import TorchDataModule

    X = np.random.randn(32, 4).astype(np.float32)
    y = np.random.randn(32).astype(np.float32)
    dm = TorchDataModule(
        train_features=X, train_labels=y,
        features_dtype=torch.float32, labels_dtype=torch.float32,
        dataloader_params={"batch_size": 8},  # no num_workers!
    )
    loader = dm.train_dataloader()
    expected = 0 if os.name == "nt" else 2
    assert loader.num_workers == expected, (
        f"F-71b: expected MLP DataLoader.num_workers={expected} (OS-aware "
        f"default); got {loader.num_workers}"
    )


def test_mlp_dataloader_explicit_num_workers_preserved():
    """User-supplied num_workers MUST win over the OS-aware default."""
    import numpy as np
    import torch
    from mlframe.training.neural.data import TorchDataModule

    X = np.random.randn(32, 4).astype(np.float32)
    y = np.random.randn(32).astype(np.float32)
    dm = TorchDataModule(
        train_features=X, train_labels=y,
        features_dtype=torch.float32, labels_dtype=torch.float32,
        dataloader_params={"batch_size": 8, "num_workers": 5},
    )
    loader = dm.train_dataloader()
    assert loader.num_workers == 5
