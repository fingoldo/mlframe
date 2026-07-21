"""MC-dropout predictive spread for torch neural nets (Workstream B1).

Keeps ONLY the dropout modules stochastic at inference (everything else, incl. BatchNorm/LayerNorm, stays
in eval so running stats are not corrupted), runs ``n`` forward passes, and returns the predictive mean +
spread. The spread is an APPROXIMATE, UNCALIBRATED variational estimate (Gal & Ghahramani) -- its scale
depends on the dropout rate, which was tuned for regularisation, not uncertainty -- so it ships as a
diagnostic / a conditional-scale input to conformal, never as a standalone calibrated interval. For
classification, prefer predictive entropy / BALD over the std of probabilities (std on a bounded simplex is
not meaningful); ``predictive_entropy`` is provided.

No-op-safe: a model with no dropout layers returns spread ~0 (mean over identical passes).
"""

from __future__ import annotations

import numpy as np


def _set_dropout_train(module) -> int:
    """Put only dropout-family modules into train mode; return how many were toggled."""
    import torch.nn as nn

    n = 0
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()
            n += 1
    return n


def mc_dropout_predict(module, X, *, n: int = 16):
    """Return ``(mean, std, n_dropout_layers)`` over ``n`` stochastic-dropout forward passes.

    ``module`` is a ``torch.nn.Module``; ``X`` a tensor it accepts. The module's original train/eval mode is
    restored on exit. ``mean``/``std`` are numpy arrays shaped like one forward pass. ``n_dropout_layers==0``
    means the spread is ~0 (no stochasticity) -- the caller can treat that as "MC-dropout unavailable".

    NOTE: ``std`` is the POPULATION standard deviation over the ``n`` passes (numpy default ``ddof=0``). For
    small ``n`` this is a low-biased estimate of the true predictive sd (it divides by ``n`` rather than
    ``n-1``); do NOT treat it as a calibrated sample sd. This is a diagnostic spread, so the bias is
    acceptable, but a consumer computing calibrated intervals should either use a large ``n`` or rescale by
    ``sqrt(n/(n-1))``.
    """
    import torch

    was_training = module.training
    module.eval()
    n_drop = _set_dropout_train(module)
    try:
        with torch.no_grad():
            preds = [np.asarray(module(X).detach().cpu().numpy()) for _ in range(max(1, n))]
    finally:
        # A raise mid-loop (OOM, bad input) must not leave dropout submodules stuck in train mode
        # while the rest of the module sits in eval -- the docstring promises restoration on exit.
        module.train(was_training)
    stacked = np.stack(preds, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0), n_drop


def predictive_entropy(mean_probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-row Shannon entropy of the mean predictive distribution (classification uncertainty summary).

    Use this instead of the std of probabilities for classification -- entropy is the meaningful uncertainty
    on the simplex. ``mean_probs`` is ``(rows, classes)`` (e.g. the MC-dropout mean of softmax outputs).
    """
    p = np.asarray(mean_probs, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    return np.asarray(-np.sum(p * np.log(p), axis=1))
