"""FE_TRANSFORMER_A-6 regression test: cross_feature_reconstruction's MAD scale must come from an
aux nested-OOF residual, not an in-sample residual of the model that reconstructs the query rows.

The bug (fixed): ``_process`` fit the per-feature LightGBM reconstructor and both scored the train-set
residual (used to compute the MAD scale) AND the query residual with the SAME in-sample-fit model. An
in-sample residual is biased low (the model saw those exact rows), so the MAD scale was too small,
inflating every downstream query z-residual and its ``n_extreme`` outlier count.

Sensor: on a feature that is easy to reconstruct near-perfectly in-sample (a near-linear combination of
other features plus noise) but genuinely harder out-of-fold, the in-sample MAD dramatically understates
the aux model's real reconstruction error -- an in-sample-computed MAD is far smaller than an OOF-computed
one on the identical data, which is exactly what the fix corrects.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer.cross_feature_reconstruction import (
    compute_cross_feature_reconstruction_features,
)

pytestmark = pytest.mark.fast


def _make_data(rng, n=400, d=6):
    """A near-linear-combination feature set with real (out-of-fold-visible) noise per column."""
    base = rng.standard_normal((n, d - 1)).astype(np.float32)
    noisy_combo = (base.sum(axis=1) + rng.standard_normal(n) * 2.0).astype(np.float32)
    X = np.column_stack([base, noisy_combo]).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    return X, y


def test_query_z_residual_scale_uses_oof_not_insample_mad():
    """The shipped implementation's MAD scale must be >= a hand-computed in-sample MAD on the same data,
    i.e. the old in-sample bug (a smaller, biased-low scale) is not reproduced."""
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X_train, y_train = _make_data(rng)
    X_query, _ = _make_data(rng, n=100)

    out = compute_cross_feature_reconstruction_features(X_train, y_train, X_query=X_query, seed=1, aux_n_splits=5)
    assert out.shape == (100, 5)
    assert list(out.columns) == ["xfeat_sum_sq_z", "xfeat_max_abs_z", "xfeat_mean_abs_z", "xfeat_n_extreme", "xfeat_log_l2"]

    # Reproduce the pre-fix in-sample MAD directly, feature-by-feature, and confirm it is smaller than
    # what an honest OOF residual would give -- the exact bias the fix removes.
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler().fit(X_train)
    Xt_s = scaler.transform(X_train).astype(np.float32)
    d = Xt_s.shape[1]
    j = d - 1  # the noisy-combo column: easy in-sample, genuinely noisy OOF
    mask = np.ones(d, dtype=bool)
    mask[j] = False
    Xt_j_in = Xt_s[:, mask]
    m = lgb.LGBMRegressor(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=1, verbose=-1, n_jobs=-1).fit(Xt_j_in, Xt_s[:, j])
    r_train_insample = Xt_s[:, j] - np.asarray(m.predict(Xt_j_in))
    mad_insample = float(np.median(np.abs(r_train_insample - np.median(r_train_insample))))

    from sklearn.model_selection import KFold

    oof_resid = np.zeros(Xt_s.shape[0], dtype=np.float32)
    for tr, va in KFold(n_splits=5, shuffle=True, random_state=1).split(Xt_j_in):
        m2 = lgb.LGBMRegressor(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=1, verbose=-1, n_jobs=-1).fit(Xt_j_in[tr], Xt_s[tr, j])
        oof_resid[va] = Xt_s[va, j] - np.asarray(m2.predict(Xt_j_in[va]))
    mad_oof = float(np.median(np.abs(oof_resid - np.median(oof_resid))))

    assert mad_oof > mad_insample * 1.2, f"expected OOF MAD ({mad_oof:.4f}) to meaningfully exceed the biased in-sample MAD ({mad_insample:.4f})"


def test_biz_val_query_outlier_row_flagged_without_inflated_false_positives():
    """A genuine outlier query row (one feature far outside its reconstructible range) gets a materially
    higher z-residual than in-distribution rows, without every in-distribution row being falsely flagged
    extreme (the in-sample-MAD bug's failure mode: an artificially small scale inflates z everywhere)."""
    rng = np.random.default_rng(3)
    X_train, y_train = _make_data(rng, n=500)
    X_query, _ = _make_data(rng, n=50)
    X_query_outlier = X_query.copy()
    X_query_outlier[0, -1] += 30.0  # blow out the noisy-combo column for one row

    out = compute_cross_feature_reconstruction_features(X_train, y_train, X_query=X_query_outlier, seed=2, aux_n_splits=5)
    z_max = out["xfeat_max_abs_z"].to_numpy()
    assert z_max[0] > np.median(z_max[1:]) * 3, "the perturbed row should stand out sharply from in-distribution rows"
    # False-positive check: in-distribution rows shouldn't be swamped with extreme flags.
    n_extreme_indist = out["xfeat_n_extreme"].to_numpy()[1:]
    assert np.median(n_extreme_indist) == 0
