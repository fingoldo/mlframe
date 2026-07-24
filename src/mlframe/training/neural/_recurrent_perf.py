"""F-44 / F-45 / F-49 (2026-05-31): PyTorch performance helpers for
the recurrent estimator family.

Carved out of ``recurrent.py`` to keep the facade under the 1k-LOC
threshold per the project's split convention. The functions are pure
free functions (taking config / precision strings) so they don't depend
on the wrapper internals; the wrapper just delegates.

Layout:
  * ``auto_precision`` -- promote 16-mixed to bf16-mixed on Ampere+
    (mirrors base.py's F-27); fp32 / bf16-mixed / explicit user choice
    pass through unchanged. fp16 has GradScaler; bf16 does not, so we
    avoid the GradScaler+RNN gradient-NaN failure mode for free.
  * ``maybe_enable_cudnn_rnn_autotune`` -- toggle
    ``torch.backends.cudnn.benchmark`` for LSTM/GRU/RNN to opt into the
    persistent-RNN kernel (Ampere+ only). On Pascal/Turing/Volta the
    kernel doesn't exist; benchmark autotune burns ~1.4-1.8 s on first
    LSTM forward without recovering it. Gated to cc >= 8.0 so Pascal
    hosts get strictly the old behaviour. Skipped when deterministic
    mode is on (benchmark + deterministic conflict + Pascal CUDA crash
    in foreach AdamW path, bench 2026-05-31).
"""
from __future__ import annotations

import logging

import torch

from ._recurrent_config import RNNType

logger = logging.getLogger(__name__)


def auto_precision(user_precision: str) -> str:
    """F-49: promote 16-mixed to bf16-mixed on Ampere+ (cc >= 8.0).

    Why: fp16 + GradScaler can NaN-out on RNN gradients (LSTM/GRU
    recurrences are particularly sensitive). bf16 has fp32 range so no
    scaler is needed; throughput is the same on Ampere+. Pre-Ampere
    pass through unchanged (typically stays ``16-mixed``).
    """
    if user_precision != "16-mixed":
        return user_precision
    try:
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            if cc >= (8, 0):
                return "bf16-mixed"
    except Exception:  # nosec B110 - best-effort path
        pass
    return user_precision


def maybe_enable_cudnn_rnn_autotune(rnn_type: RNNType) -> "bool | None":
    """F-45: enable cuDNN persistent-RNN kernel via benchmark=True.

    Documented 2-3x lift on Ampere+ where the persistent kernel lives.
    On Pascal bench (2026-05-31, GTX 1050 Ti): warmup 2.45s -> 4.20s,
    profiled 2.53s -> 2.78s = net regression. Hard-gate to cc >= 8.0.
    Lift the gate once an Ampere+ host bench confirms.

    Skipped when:
      * rnn_type is TRANSFORMER (the gain is on LSTM/GRU/RNN cuDNN kernels)
      * CUDA unavailable
      * deterministic flag is on (benchmark + deterministic conflict)
      * compute capability < 8.0 (no persistent-RNN kernel exists)

    Returns the PRIOR value of ``torch.backends.cudnn.benchmark`` when this call actually flips it, or
    ``None`` when skipped (nothing to restore). ``torch.backends.cudnn.benchmark`` is a process-global
    PyTorch flag -- the caller MUST restore the returned prior value (in a ``finally``) once its own
    ``trainer.fit()`` call returns, or every OTHER model trained later in the same process silently
    inherits ``benchmark=True`` and its non-deterministic cuDNN algorithm selection.
    """
    if rnn_type == RNNType.TRANSFORMER:
        return None
    if not torch.cuda.is_available():
        return None
    if torch.backends.cudnn.deterministic:
        return None
    try:
        cc = torch.cuda.get_device_capability()
    except Exception as exc:
        logger.debug("maybe_enable_cudnn_rnn_autotune: compute-capability probe failed: %s", exc)
        return None
    if cc < (8, 0):
        return None
    prior = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    return prior
