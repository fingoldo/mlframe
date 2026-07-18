"""Locks the Windows-aware (and Linux-aware) MLP runtime defaults.

The 2026-05-12 fix turned MLP DataLoader workers / AMP precision / predict
batch_size into auto-resolved defaults. The user explicitly warned that
``num_workers > 0`` had caused **big problems on Windows** in past projects
(spawn-based multiprocessing + CUDA tensor sharing on Windows hits silent
hangs / OOM / 'Pin memory thread exited unexpectedly' errors).

A second audit pass exposed that the Linux/Mac path is no safer in our
codebase: ``TorchDataset`` (mlframe/training/neural/data.py:62) holds the
entire Polars / pandas frame in its closure. Fork-based workers on Linux
share Arrow buffers via CoW initially, BUT Python refcount writes break CoW
on every access and ``polars.DataFrame[indices]`` materialises copies on the
per-row path -- on a 100 GB frame even partial CoW breakage swap-thrashes the
box. So the contract this suite locks:

    1. ``num_workers`` defaults to 0 on EVERY OS (Windows / Linux / Mac).
       Users opt in explicitly via
       ``mlp_kwargs["dataloader_params"]["num_workers"]``.
    2. ``persistent_workers`` / ``prefetch_factor`` are FORCED off when
       ``num_workers == 0`` regardless of user override (PyTorch ValueError
       guard).
    3. ``pin_memory`` defaults to True on CUDA hosts, False otherwise -- no
       worker-pickling landmine.
    4. AMP precision auto-resolves to ``"bf16-mixed"`` only on Ampere+ (CC
       major >= 8); older GPUs and CPU mode stay on ``"32-true"``. User
       override always wins.
    5. Train batch_size="auto" resolves at DataModule time, where the
       feature width is known, and logs the selected value.
    6. Predict batch_size adapts to memory + dataframe width: it picks the
       biggest batch that fits ``mem_fraction`` of free memory at the
       assumed dtype + activation overhead, clamped to ``[64, 16384]``.
       Without a width hint or memory probe it returns the conservative
       fallback (1024) -- still beats the old 64 default, won't OOM 30K-
       feature scenarios where the user previously had to drop to 64.
"""

from __future__ import annotations

import pytest

from mlframe.training.mlp_runtime_defaults import (
    _PREDICT_BATCH_FALLBACK,
    _PREDICT_BATCH_MAX,
    _PREDICT_BATCH_MIN,
    _TRAIN_BATCH_FALLBACK,
    _TRAIN_BATCH_MAX,
    _TRAIN_BATCH_MIN,
    resolve_mlp_dataloader_defaults,
    resolve_mlp_precision_default,
    resolve_mlp_predict_batch_size,
    resolve_mlp_train_batch_size,
)

# ---------------------------------------------------------------------------
# DataLoader defaults
# ---------------------------------------------------------------------------


class TestDataLoaderDefaultsWindows:
    """Lock the Windows-safe behaviour: never auto-raise workers."""

    def test_windows_num_workers_is_zero_by_default(self) -> None:
        """Windows num workers is zero by default."""
        res = resolve_mlp_dataloader_defaults(
            cpu_count=32,
            cuda_available=True,
            force_windows=True,
        )
        assert (
            res["num_workers"] == 0
        ), "Windows must keep num_workers=0 by default; raising auto on Windows hits spawn + CUDA-tensor-sharing bugs the user has previously been bitten by."

    def test_windows_zero_workers_implies_no_persistent_no_prefetch(self) -> None:
        """Windows zero workers implies no persistent no prefetch."""
        res = resolve_mlp_dataloader_defaults(
            cpu_count=32,
            cuda_available=True,
            force_windows=True,
        )
        assert res["persistent_workers"] is False
        assert res["prefetch_factor"] is None

    def test_windows_user_can_explicitly_override_workers(self) -> None:
        """User opts in by passing num_workers > 0 -- we honour it (with the
        associated PyTorch-required flags) but the DEFAULT path remains 0."""
        res = resolve_mlp_dataloader_defaults(
            user_overrides={"num_workers": 4},
            cpu_count=32,
            cuda_available=True,
            force_windows=True,
        )
        assert res["num_workers"] == 4
        assert res["persistent_workers"] is True
        assert res["prefetch_factor"] == 4

    def test_windows_pin_memory_still_true_on_cuda(self) -> None:
        """pin_memory has no Windows landmine -- True when CUDA is up."""
        res = resolve_mlp_dataloader_defaults(
            cpu_count=8,
            cuda_available=True,
            force_windows=True,
        )
        assert res["pin_memory"] is True

    def test_windows_pin_memory_false_without_cuda(self) -> None:
        """Windows pin memory false without cuda."""
        res = resolve_mlp_dataloader_defaults(
            cpu_count=8,
            cuda_available=False,
            force_windows=True,
        )
        assert res["pin_memory"] is False


class TestDataLoaderDefaultsPosix:
    """Lock the Linux/Mac behaviour -- now ALSO defaulting to 0 workers
    because ``TorchDataset`` holds the full Polars frame in its closure.
    Fork-based workers + Polars indexing breaks Arrow CoW on big frames."""

    def test_posix_num_workers_is_zero_by_default(self) -> None:
        """Same as Windows -- the Polars-frame-in-closure landmine."""
        res = resolve_mlp_dataloader_defaults(
            cpu_count=32,
            cuda_available=True,
            force_windows=False,
        )
        assert res["num_workers"] == 0

    def test_posix_zero_workers_implies_no_persistent_no_prefetch(self) -> None:
        """Posix zero workers implies no persistent no prefetch."""
        res = resolve_mlp_dataloader_defaults(
            cpu_count=32,
            cuda_available=True,
            force_windows=False,
        )
        assert res["persistent_workers"] is False
        assert res["prefetch_factor"] is None

    def test_posix_user_can_explicitly_override_workers(self) -> None:
        """User opts in -- we honour the value AND set the PyTorch-required
        persistent / prefetch defaults to non-trivial values."""
        res = resolve_mlp_dataloader_defaults(
            user_overrides={"num_workers": 4},
            cpu_count=8,
            cuda_available=True,
            force_windows=False,
        )
        assert res["num_workers"] == 4
        assert res["persistent_workers"] is True
        assert res["prefetch_factor"] == 4

    def test_posix_user_overrides_persistent_workers(self) -> None:
        """Posix user overrides persistent workers."""
        res = resolve_mlp_dataloader_defaults(
            user_overrides={"num_workers": 4, "persistent_workers": False},
            cpu_count=8,
            cuda_available=True,
            force_windows=False,
        )
        assert res["num_workers"] == 4
        # User explicitly turned it off -- honour even though num_workers > 0.
        assert res["persistent_workers"] is False

    def test_posix_user_overrides_prefetch_factor(self) -> None:
        """Posix user overrides prefetch factor."""
        res = resolve_mlp_dataloader_defaults(
            user_overrides={"num_workers": 4, "prefetch_factor": 16},
            cpu_count=8,
            cuda_available=True,
            force_windows=False,
        )
        assert res["prefetch_factor"] == 16


class TestDataLoaderDefaultsExtra:
    """Groups tests covering data loader defaults extra."""
    def test_user_extra_kwargs_pass_through(self) -> None:
        """Unrecognised user kwargs (timeout, sampler, etc.) survive unchanged."""
        res = resolve_mlp_dataloader_defaults(
            user_overrides={"timeout": 30, "sampler": "weighted"},
            cpu_count=8,
            cuda_available=False,
            force_windows=False,
        )
        assert res["timeout"] == 30
        assert res["sampler"] == "weighted"


# ---------------------------------------------------------------------------
# AMP precision
# ---------------------------------------------------------------------------


class TestPrecisionDefault:
    """Groups tests covering precision default."""
    def test_user_override_wins(self) -> None:
        """User override wins."""
        assert (
            resolve_mlp_precision_default(
                user_override="16-mixed",
                cuda_available=True,
                cuda_compute_capability_major=8,
            )
            == "16-mixed"
        )

    def test_cpu_only_returns_32_true(self) -> None:
        """Cpu only returns 32 true."""
        assert (
            resolve_mlp_precision_default(
                cuda_available=False,
            )
            == "32-true"
        )

    def test_ampere_returns_bf16_mixed(self) -> None:
        """Ampere returns bf16 mixed."""
        assert (
            resolve_mlp_precision_default(
                cuda_available=True,
                cuda_compute_capability_major=8,
            )
            == "bf16-mixed"
        )

    def test_ada_lovelace_returns_bf16_mixed(self) -> None:
        # RTX 40-series -- sm_89, major=8.
        """Ada lovelace returns bf16 mixed."""
        assert (
            resolve_mlp_precision_default(
                cuda_available=True,
                cuda_compute_capability_major=8,
            )
            == "bf16-mixed"
        )

    def test_hopper_returns_bf16_mixed(self) -> None:
        # H100 -- sm_90, major=9.
        """Hopper returns bf16 mixed."""
        assert (
            resolve_mlp_precision_default(
                cuda_available=True,
                cuda_compute_capability_major=9,
            )
            == "bf16-mixed"
        )

    def test_volta_t4_stays_32_true(self) -> None:
        # V100=sm_70 (major=7), T4=sm_75 (major=7) -- bf16 is emulated, slower.
        """Volta t4 stays 32 true."""
        assert (
            resolve_mlp_precision_default(
                cuda_available=True,
                cuda_compute_capability_major=7,
            )
            == "32-true"
        )

    def test_pascal_stays_32_true(self) -> None:
        # P100=sm_60 (major=6).
        """Pascal stays 32 true."""
        assert (
            resolve_mlp_precision_default(
                cuda_available=True,
                cuda_compute_capability_major=6,
            )
            == "32-true"
        )


# ---------------------------------------------------------------------------
# Predict batch_size -- adaptive on memory + width
# ---------------------------------------------------------------------------


class TestPredictBatchSize:
    """Groups tests covering predict batch size."""
    def test_user_override_wins(self) -> None:
        """User override wins."""
        assert (
            resolve_mlp_predict_batch_size(
                user_override=512,
                n_features=25,
                available_memory_bytes=8_000_000_000,
            )
            == 512
        )

    def test_user_override_clamped_to_min_one(self) -> None:
        """User override clamped to min one."""
        assert resolve_mlp_predict_batch_size(user_override=0) == 1
        assert resolve_mlp_predict_batch_size(user_override=-5) == 1

    def test_no_width_hint_returns_fallback(self) -> None:
        """No width hint returns fallback."""
        assert (
            resolve_mlp_predict_batch_size(
                n_features=None,
                available_memory_bytes=8_000_000_000,
            )
            == _PREDICT_BATCH_FALLBACK
        )

    def test_no_memory_probe_returns_fallback(self) -> None:
        """No memory probe returns fallback."""
        assert resolve_mlp_predict_batch_size(
            n_features=25,
            available_memory_bytes=None,
            cuda_available=False,
        ) in (_PREDICT_BATCH_FALLBACK, _PREDICT_BATCH_MAX)
        # The first form (constant) fires when psutil + torch are both
        # unavailable to probe. On a typical dev box psutil is installed -> memory IS
        # probed and the resolver picks the adaptive size; that path is
        # locked by the explicit-memory tests below.

    def test_narrow_dataframe_picks_max_batch(self) -> None:
        """25 features + 8 GB free -> batch capped at 16384, not unlimited."""
        # budget = 8e9 * 0.25 = 2e9 ; per_row = 25 * 4 * 4 = 400 ; 2e9 / 400 = 5e6
        # clamp to max=16384.
        res = resolve_mlp_predict_batch_size(
            n_features=25,
            available_memory_bytes=8_000_000_000,
        )
        assert res == _PREDICT_BATCH_MAX

    def test_wide_30k_features_4gb_gpu_picks_small_batch(self) -> None:
        """The user's prior 30K-features case where batch=64 was the max.

        With 4 GB free, mem_fraction=0.25, activation_mult=4, dtype=4:
          per_row = 30000 * 4 * 4 = 480000 bytes
          budget  = 4e9 * 0.25    = 1e9
          raw     = 1e9 / 480000  = 2083
        -> clamped within [64, 16384] = 2083.
        Still beats the old hardcoded 64 by 30x, AND won't OOM because we
        only use 25% of free memory (room for the model + cuda streams).
        """
        res = resolve_mlp_predict_batch_size(
            n_features=30_000,
            available_memory_bytes=4_000_000_000,
        )
        assert 1000 <= res <= 4096, f"Wide-30K-feature case should pick a small batch (~2K); got {res}"

    def test_extremely_wide_features_picks_min_batch(self) -> None:
        """Pathologically wide dataframe + low memory -> floor at min_batch (64)."""
        res = resolve_mlp_predict_batch_size(
            n_features=500_000,
            available_memory_bytes=100_000_000,  # 100 MB
        )
        assert res == _PREDICT_BATCH_MIN, f"Pathologically wide case must clamp to min_batch={_PREDICT_BATCH_MIN}, got {res}"

    def test_mem_fraction_lower_picks_smaller_batch(self) -> None:
        """Knob mem_fraction reduces the picked batch proportionally."""
        big = resolve_mlp_predict_batch_size(
            n_features=100,
            available_memory_bytes=4_000_000_000,
            mem_fraction=0.5,
        )
        small = resolve_mlp_predict_batch_size(
            n_features=100,
            available_memory_bytes=4_000_000_000,
            mem_fraction=0.1,
        )
        assert big >= small

    def test_zero_memory_returns_min_batch(self) -> None:
        """Zero memory returns min batch."""
        res = resolve_mlp_predict_batch_size(
            n_features=10,
            available_memory_bytes=0,
        )
        # available_memory_bytes <= 0 path returns fallback (probe failed).
        assert res == _PREDICT_BATCH_FALLBACK

    def test_negative_features_returns_fallback(self) -> None:
        """Negative features returns fallback."""
        assert (
            resolve_mlp_predict_batch_size(
                n_features=-1,
                available_memory_bytes=8_000_000_000,
            )
            == _PREDICT_BATCH_FALLBACK
        )


# ---------------------------------------------------------------------------
# Train batch_size -- auto at DataModule time
# ---------------------------------------------------------------------------


class TestTrainBatchSize:
    """Groups tests covering train batch size."""
    def test_no_width_hint_returns_fallback(self) -> None:
        """No width hint returns fallback."""
        assert (
            resolve_mlp_train_batch_size(
                n_features=None,
                available_memory_bytes=8_000_000_000,
            )
            == _TRAIN_BATCH_FALLBACK
        )

    def test_narrow_dataframe_keeps_historical_ceiling(self) -> None:
        """Narrow dataframe keeps historical ceiling."""
        res = resolve_mlp_train_batch_size(
            n_features=25,
            available_memory_bytes=8_000_000_000,
        )
        assert res == _TRAIN_BATCH_MAX

    def test_extremely_wide_features_picks_min_batch(self) -> None:
        """Extremely wide features picks min batch."""
        res = resolve_mlp_train_batch_size(
            n_features=500_000,
            available_memory_bytes=100_000_000,
        )
        assert res == _TRAIN_BATCH_MIN

    def test_datamodule_auto_batch_logs_selected_value(self, caplog, monkeypatch) -> None:
        """Datamodule auto batch logs selected value."""
        import logging
        import numpy as np

        torch = pytest.importorskip("torch")
        pytest.importorskip("lightning")
        from mlframe.training import mlp_runtime_defaults
        from mlframe.training.neural.data import TorchDataModule

        monkeypatch.setattr(
            mlp_runtime_defaults,
            "_probe_available_memory_bytes",
            lambda *, cuda_available=None: 100_000_000,
        )
        dm = TorchDataModule(
            train_features=np.zeros((2, 500_000), dtype=np.float32),
            train_labels=np.zeros(2, dtype=np.float32),
            dataloader_params={"batch_size": "auto", "num_workers": 0},
            features_dtype=torch.float32,
            labels_dtype=torch.float32,
        )

        with caplog.at_level(logging.INFO, logger="mlframe.training.neural.data"):
            assert dm._resolve_batch_size("auto", dm.train_features, "train") == _TRAIN_BATCH_MIN

        assert any("MLP train DataLoader auto-selected batch_size=" in record.getMessage() for record in caplog.records)
