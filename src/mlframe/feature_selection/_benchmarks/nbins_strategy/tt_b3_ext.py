"""Proposed extensions to the frozen B3 reference (agent B3, 2026-11-24). They are NOT part of the wave-1 recipe; each is validated on fresh seeds in the report before being recommended.

``polish_lambda``: when the chosen transform is Yeo-Johnson, replace the profile-likelihood lambda (best for the Gaussian density of f(y), fitted on the training half) by the lambda of a 33-point grid that
minimises the held-out ORIGINAL-scale error of the same closed-form head (smeared RMSE or median MAE). Motivation measured in the B3 report: the density-optimal lambda is not the error-optimal one on skewed targets (``exp_add``, n 2000, seed 100: profile
lambda -0.36 gives an RMSE ratio of 0.60 to identity, the error-minimising lambda -0.6 gives 0.52). The grid error is computed on the rows that already served for the second gate, so it
is mildly optimistic; the identity comparison of the second gate is repeated with the polished lambda.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import tt_b3_reference as R

_POLISH_GRID = np.linspace(-1.0, 3.0, 33)


def polish_lambda(an: R.TransformAnalysis, ch: R.TargetTransformChoice) -> R.TargetTransformChoice:
    """Return ``ch`` with the Yeo-Johnson lambda re-chosen by held-out original-scale error (the choice is returned unchanged for any other transform or for an abstention)."""
    if ch.abstained or ch.name != "yeo_johnson" or not an.fold_list:
        return ch
    y = an.y_ref
    idx = 0 if ch.metric == "rmse" else 1
    best_lam, best_err = None, np.inf
    for lam in _POLISH_GRID:
        if abs(lam - 1.0) < R._YJ_COLLAPSE:
            continue
        parts = [R._heldout_losses(f, y, "yeo_johnson", float(lam), an.scale) for f in an.fold_list]
        loss = np.concatenate([p[idx] for p in parts])
        err = float(np.sqrt(loss.mean())) if idx == 0 else float(loss.mean())
        if err < best_err:
            best_lam, best_err = float(lam), err
    if best_lam is None or not best_err <= ch.err_identity:
        return ch
    tr = R.build_transform("yeo_johnson", y, best_lam)
    return dataclasses.replace(ch, params=tr.params, err_chosen=best_err, transform=tr)
