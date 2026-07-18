"""SA26 regression: WAIC must measure OOF accuracy, not self-normalize per fold.

Pre-fix each fold's Gaussian variance was estimated from the held-out fold's OWN
residuals, so the per-point density self-normalized to each transform's held-out
scale: a transform with genuinely smaller out-of-fold error got (nearly) the SAME
elpd as a noisier one, suppressing the OOF-accuracy signal WAIC is supposed to
reward. Estimating the variance from the TRAIN-fold residuals (a scale the held-out
points did not see) restores the signal: the better-predicting transform now scores
a clearly higher WAIC.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery._eval_waic import (
    waic_from_oof_residuals,
    compute_transform_waic,
)


def test_overfit_candidate_penalized_with_train_scale_but_hidden_when_self_normalized():
    """An OVERFIT candidate has SMALL train-fold residuals but LARGE out-of-fold
    residuals (it memorised the screen, generalises poorly). A GENERALISING candidate
    has matched train and OOF residuals.

    Self-normalizing each fold's density to its OWN held-out residuals (pre-fix) hides
    the overfit: sigma^2 inflates with the OOF error, so r_oof^2/sigma^2 ~ 1 for both
    and the overfit candidate is NOT penalised -- it can even score HIGHER. Using the
    TRAIN-fold scale, the overfit candidate pays r_oof^2/sigma_train^2 >> 1 and ranks
    BELOW the generalising one, which is the whole point of the OOF signal."""
    rng = np.random.default_rng(0)
    n_folds = 5
    # Generalising candidate: train and OOF residuals both ~N(0, 0.5).
    gen_train = [rng.normal(0.0, 0.5, 240) for _ in range(n_folds)]
    gen_oof = [rng.normal(0.0, 0.5, 80) for _ in range(n_folds)]
    # Overfit candidate: tiny train residuals (0.1), large OOF residuals (1.0).
    of_train = [rng.normal(0.0, 0.1, 240) for _ in range(n_folds)]
    of_oof = [rng.normal(0.0, 1.0, 80) for _ in range(n_folds)]

    # Post-fix: train-fold scale -> generalising candidate wins clearly.
    s_gen = waic_from_oof_residuals(gen_oof, target_scale=1.0, fold_scale_residuals=gen_train)
    s_of = waic_from_oof_residuals(of_oof, target_scale=1.0, fold_scale_residuals=of_train)
    assert s_gen.valid and s_of.valid
    assert s_gen.waic > s_of.waic, f"train-scale WAIC must penalise the overfit candidate: gen waic={s_gen.waic:.3f} vs overfit waic={s_of.waic:.3f}"

    # Pre-fix path (self-normalized to held-out residuals): the overfit candidate's
    # sigma^2 inflates with its own large OOF error, so r_oof^2/sigma^2 ~ 1 and the
    # mismatch between memorised (train) fit and held-out error is washed out. The
    # overfit is barely penalised. The fix amplifies the penalty by an ORDER OF
    # MAGNITUDE, because the train-scale denominator no longer absorbs the OOF error.
    p_gen = waic_from_oof_residuals(gen_oof, target_scale=1.0)
    p_of = waic_from_oof_residuals(of_oof, target_scale=1.0)
    selfnorm_sep = p_gen.waic - p_of.waic
    trainscale_sep = s_gen.waic - s_of.waic
    assert (
        trainscale_sep > 10.0 * selfnorm_sep
    ), f"train-scale must penalise the overfit far harder than self-norm: trainscale gap={trainscale_sep:.2f} vs selfnorm gap={selfnorm_sep:.2f}"


def test_compute_transform_waic_rewards_better_predictor():
    """End-to-end: a target that the features genuinely predict well gets a higher
    WAIC than a near-noise target."""
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=(n, 3))
    y_good = x[:, 0] * 2.0 + x[:, 1] - 0.5 * x[:, 2] + rng.normal(0.0, 0.2, n)
    y_bad = rng.normal(0.0, 1.0, n)  # unrelated to x
    s_good = compute_transform_waic(y_good, x, n_folds=4, random_state=0)
    s_bad = compute_transform_waic(y_bad, x, n_folds=4, random_state=0)
    assert s_good.valid and s_bad.valid
    assert s_good.waic > s_bad.waic, f"good waic {s_good.waic:.3f} !> bad {s_bad.waic:.3f}"
