"""Regression tests for the F-44..F-51 recurrent perf bundle (committed 2e52d298).

These tests pin the *gating* behaviour of the perf knobs -- they don't
measure wall time. Whether the underlying optimization fires is HW-
specific (Pascal vs Ampere+, CPU vs CUDA), but the gates themselves are
pure config decisions and verifiable on any host.

Covered:
  F-44 fused-AdamW gate (skip under 16-mixed; install under bf16-mixed / fp32)
  F-49 bf16-mixed auto-promote (16-mixed -> bf16-mixed on cc >= 8.0)
  F-45 cuDNN benchmark skip on Transformer / non-CUDA / pre-Ampere
  F-51 shared-memory promotion on RecurrentDataset CPU tensors
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch

from mlframe.training.neural._recurrent_config import RNNType, InputMode, RecurrentConfig
from mlframe.training.neural._recurrent_data import RecurrentDataset
from mlframe.training.neural._recurrent_perf import (
    auto_precision,
    maybe_enable_cudnn_rnn_autotune,
)


# --------------------------------------------------------------------- F-49
def test_auto_precision_passes_through_explicit_user_choice():
    # An explicit non-"16-mixed" choice MUST NOT be rewritten by the promoter.
    # The promoter only acts on the implicit Lightning default ("16-mixed").
    """Auto precision passes through explicit user choice."""
    for explicit in ("32-true", "bf16-mixed", "16-true", "64-true"):
        assert auto_precision(explicit) == explicit


def test_auto_precision_pre_ampere_keeps_16_mixed():
    # On a host without CUDA we know cc-detection returns nothing usable,
    # so the function must fall back to the user-supplied value.
    """Auto precision pre ampere keeps 16 mixed."""
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc >= (8, 0):
            pytest.skip("test asserts pre-Ampere behaviour; this host is Ampere+")
    assert auto_precision("16-mixed") == "16-mixed"


# --------------------------------------------------------------------- F-44
def test_f44_fused_adamw_skipped_when_trainer_runs_16_mixed(monkeypatch):
    """F-44 critical safety gate: Lightning AMP plugin's grad-clipping pass
    cannot reconcile with fused AdamW under a live GradScaler. The gate
    inspects ``self.trainer.precision``; when it's "16-mixed", fused MUST
    stay off regardless of CUDA availability.

    Verified by introspecting the AdamW state: ``param_groups[0]["fused"]``
    is False (or unset, default False) when the gate skips it.
    """
    from mlframe.training.neural._recurrent_torch_model import RecurrentTorchModel

    if not torch.cuda.is_available():
        pytest.skip("F-44 fused gate only matters when CUDA is available")

    cfg = RecurrentConfig(
        input_mode=InputMode.FEATURES_ONLY,
        rnn_type=RNNType.LSTM,
        hidden_size=8,
        num_layers=1,
    )
    model = RecurrentTorchModel(cfg, aux_input_size=4, is_regression=True)

    class _FakeTrainer:
        """Groups tests covering fake trainer."""
        precision = "16-mixed"
        estimated_stepping_batches = 0

    model.trainer = _FakeTrainer()
    out = model.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    fused = opt.param_groups[0].get("fused", False)
    # PyTorch stores ``fused`` as None (not False) when not explicitly set;
    # check ``not True`` to catch both None and False as "fused is off".
    assert fused is not True, f"F-44 gate failed: fused={fused!r} under 16-mixed -> AMP grad-clip conflict"


def test_f44_fused_adamw_skipped_under_bf16_mixed():
    """F-44 follow-up (2026-05-31 EMA-test failure on CPU+bf16-mixed):
    Lightning's AMP precision plugin's ``_clip_gradients`` rejects fused
    AdamW under ANY mixed precision, not just 16-mixed -- bf16-mixed has
    no GradScaler but still routes through the same plugin path. Pin the
    stricter gate that excludes both.
    """
    from mlframe.training.neural._recurrent_torch_model import RecurrentTorchModel

    if not torch.cuda.is_available():
        pytest.skip("F-44 fused gate only matters when CUDA is available")

    cfg = RecurrentConfig(
        input_mode=InputMode.FEATURES_ONLY,
        rnn_type=RNNType.LSTM,
        hidden_size=8,
        num_layers=1,
    )
    model = RecurrentTorchModel(cfg, aux_input_size=4, is_regression=True)

    class _FakeTrainer:
        """Groups tests covering fake trainer."""
        precision = "bf16-mixed"
        estimated_stepping_batches = 0

    model.trainer = _FakeTrainer()
    out = model.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    fused = opt.param_groups[0].get("fused", False)
    assert (
        fused is not True
    ), f"F-44 gate failed: fused={fused!r} under bf16-mixed -> AMP plugin grad-clip rejection. The gate must skip fused for any 'mixed' precision."


def test_f44_fused_adamw_active_under_fp32():
    """F-44 positive case: under 32-true (no AMP plugin), fused AdamW
    is safe and MUST install. This is the path that gives the 1.05-1.15x
    optim step lift when the user explicitly disables mixed precision."""
    from mlframe.training.neural._recurrent_torch_model import RecurrentTorchModel

    if not torch.cuda.is_available():
        pytest.skip("F-44 fused only fires on CUDA")

    cfg = RecurrentConfig(
        input_mode=InputMode.FEATURES_ONLY,
        rnn_type=RNNType.LSTM,
        hidden_size=8,
        num_layers=1,
    )
    model = RecurrentTorchModel(cfg, aux_input_size=4, is_regression=True)

    class _FakeTrainer:
        """Groups tests covering fake trainer."""
        precision = "32-true"
        estimated_stepping_batches = 0

    model.trainer = _FakeTrainer()
    out = model.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    assert opt.param_groups[0].get("fused", False) is True, "F-44 expected fused=True under 32-true; got fused=False or unset"


# --------------------------------------------------------------------- F-45
def test_f45_cudnn_autotune_skipped_for_transformer():
    # Transformer path doesn't use the cuDNN RNN persistent-kernel; the
    # helper must early-return without mutating cudnn.benchmark.
    """F45 cudnn autotune skipped for transformer."""
    prev = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = False
        maybe_enable_cudnn_rnn_autotune(RNNType.TRANSFORMER)
        assert torch.backends.cudnn.benchmark is False, "F-45 should be a no-op on TRANSFORMER rnn_type"
    finally:
        torch.backends.cudnn.benchmark = prev


def test_f45_cudnn_autotune_skipped_pre_ampere():
    # On Pascal/Turing/Volta (cc < 8.0) the persistent-RNN kernel doesn't
    # exist and benchmark autotune costs more than it returns.
    """F45 cudnn autotune skipped pre ampere."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required to read compute capability")
    cc = torch.cuda.get_device_capability()
    if cc >= (8, 0):
        pytest.skip("test asserts pre-Ampere behaviour; this host is Ampere+")
    prev = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = False
        maybe_enable_cudnn_rnn_autotune(RNNType.LSTM)
        assert torch.backends.cudnn.benchmark is False, f"F-45 should be a no-op on cc < 8.0 (got {cc})"
    finally:
        torch.backends.cudnn.benchmark = prev


# --------------------------------------------------------------------- F-51
def test_f51_recurrent_dataset_promotes_cpu_tensors_to_shared_memory():
    """F-51 (2026-05-31): RecurrentDataset.{labels, aux_features,
    sample_weights} must be share_memory()-promoted so DataLoader workers
    attach to one backing buffer instead of pickling a fresh copy.

    Verifies that ``is_shared()`` returns True on each CPU tensor attr.
    """
    n = 32
    sequences = [np.random.randn(10, 4).astype(np.float32) for _ in range(n)]
    aux = np.random.randn(n, 8).astype(np.float32)
    labels = np.random.randn(n).astype(np.float32)
    weights = np.ones(n, dtype=np.float32)
    ds = RecurrentDataset(
        sequences=sequences,
        aux_features=aux,
        labels=labels,
        sample_weights=weights,
        is_regression=True,
    )
    assert ds.labels.is_shared(), "F-51: labels tensor not promoted to shared memory"
    assert ds.aux_features.is_shared(), "F-51: aux_features not promoted to shared memory"
    assert ds.sample_weights.is_shared(), "F-51: sample_weights not promoted to shared memory"
