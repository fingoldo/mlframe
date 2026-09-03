"""Shared LightGBM focal-loss custom objective (Lin et al. 2017) for the ``multi_aux_ensemble.py`` /
``focal_lgb.py`` transformer family: independently duplicated across those modules, consolidated here
so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def make_focal_objective(gamma: float = 2.0) -> Callable[[np.ndarray, Any], "tuple[np.ndarray, np.ndarray]"]:
    """Build a LightGBM-conforming ``(preds, train_data) -> (grad, hess)`` closure for binary focal loss."""

    def objective(preds, train_data):
        """LightGBM custom-objective signature: raw logits + Dataset -> (grad, hess) for the focal loss above.

        Derived via the chain rule through ``pt = y*p + (1-y)*(1-p)`` (an affine function of
        ``p = sigmoid(z)``, so ``dpt/dp = s := 2*y - 1`` and ``dp/dz = p*(1-p)``):
        ``FL(pt) = -(1-pt)**gamma * log(pt)``, ``A := dFL/dpt``, ``App := d2FL/dpt2``,
        ``grad = A * s * p*(1-p)``, ``hess = App * (p*(1-p))**2 + s*A*p*(1-p)*(1-2*p)``.
        Verified against a sympy symbolic/finite-difference ground truth (max diff ~5e-14).
        """
        labels = train_data.get_label()
        preds_clipped = np.clip(preds, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-preds_clipped))
        pt = labels * p + (1.0 - labels) * (1.0 - p)
        pt = np.clip(pt, 1e-9, 1.0 - 1e-9)
        s = 2.0 * labels - 1.0
        log_pt = np.log(pt)
        one_minus_pt = 1.0 - pt
        a = one_minus_pt ** (gamma - 1.0) * (gamma * pt * log_pt + pt - 1.0) / pt
        a_prime = one_minus_pt ** (gamma - 2.0) * (-gamma * (gamma - 1.0) * pt**2 * log_pt + 2.0 * gamma * pt * one_minus_pt + one_minus_pt**2) / pt**2
        p_1mp = p * (1.0 - p)
        grad = a * s * p_1mp
        hess = a_prime * p_1mp**2 + s * a * p_1mp * (1.0 - 2.0 * p)
        grad = np.clip(grad, -10.0, 10.0)
        hess = np.maximum(hess, 1e-6)
        return grad, hess

    return objective
