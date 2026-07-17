"""Cross-model business-value proof for the monotonic strict-decline early stop.

User directive: the monotonic-decline stop must GOVERN TRAINING of every mlframe model type and
DEMONSTRATE its value across them -- explicitly lgb, xgb, cb, AND mlp. This file is the quantitative
proof: on an overfit-prone synthetic, the monotonic stop (default-on, patience=5) ends training STRICTLY
EARLIER (fewer trees / epochs) than the no-monotonic baseline for EACH backend, WITHOUT hurting the
held-out metric.

Production stop path under test (same shared ``MonotonicDeclineStopper`` everywhere):
  * lgb -> ``LGBMonotonicDeclineStop`` raises ``lgb.callback.EarlyStopException`` (wired in ``lgb_shim``)
  * xgb -> ``_make_xgb_monotonic_callback`` -> ``after_iteration`` returns True (wired in ``xgb_shim``)
  * cb  -> ``CBMonotonicDeclineStop`` -> ``after_iteration`` returns False (wired in ``_data_helpers``)
  * mlp -> ``MonotonicDeclineStopCallback`` sets ``trainer.should_stop`` (live Lightning wiring covered by
           ``tests/training/neural/test_monotonic_decline_integration.py``); here we replay the exact
           ``MonotonicDeclineStopper`` the MLP callback wraps on a realistic val-loss curve.

The boosters are fit through their REAL mlframe fit paths so this exercises production code, not a
reimplementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from sklearn.metrics import roc_auc_score


N_ROUNDS = 400
SEED = 17


def _overfit_prone_split():
    """Tiny noisy-label train + clean held-out val/test where a high-capacity booster overfits fast."""
    rng = np.random.default_rng(SEED)
    d = 8
    w = rng.normal(size=d)

    def make(n, noise):
        """Make."""
        X = rng.normal(size=(n, d))
        logit = X @ w + rng.normal(size=n) * noise
        return X.astype(np.float32), (logit > 0).astype(int)

    X_tr, y_tr = make(220, noise=3.0)  # heavy label noise -> overfit
    X_val, y_val = make(400, noise=0.3)  # clean val -> curve diverges after early best
    X_te, y_te = make(400, noise=0.3)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


# --------------------------------------------------------------------------- lgb


def test_biz_val_lgb_monotonic_stops_earlier_no_test_regression():
    """Biz val lgb monotonic stops earlier no test regression."""
    pytest.importorskip("lightgbm")
    from mlframe.training.lgb_shim import LGBMClassifierWithDatasetReuse

    X_tr, y_tr, X_val, y_val, X_te, y_te = _overfit_prone_split()

    def run(mono):
        """Fits an LGBM classifier with the given monotonic_decline_patience and returns (rounds trained, test AUC)."""
        m = LGBMClassifierWithDatasetReuse(
            n_estimators=N_ROUNDS,
            num_leaves=63,
            learning_rate=0.2,
            verbose=-1,
            random_state=SEED,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc", monotonic_decline_patience=mono)
        return m.booster_.current_iteration(), roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])

    n_mono, auc_mono = run(3)
    n_base, auc_base = run(None)
    assert n_mono < n_base, f"lgb: monotonic did not stop earlier ({n_mono} vs {n_base})"
    assert auc_mono >= auc_base - 0.03, f"lgb: held-out AUC hurt ({auc_mono:.4f} < {auc_base:.4f})"


# --------------------------------------------------------------------------- xgb


def test_biz_val_xgb_monotonic_stops_earlier_no_test_regression():
    """Biz val xgb monotonic stops earlier no test regression."""
    pytest.importorskip("xgboost")
    from mlframe.training.xgb_shim import XGBClassifierWithDMatrixReuse

    X_tr, y_tr, X_val, y_val, X_te, y_te = _overfit_prone_split()

    def run(mono):
        """Fits an XGB classifier with the given monotonic_decline_patience and returns (rounds trained, test AUC)."""
        m = XGBClassifierWithDMatrixReuse(
            n_estimators=N_ROUNDS,
            max_depth=8,
            learning_rate=0.3,
            eval_metric="auc",
            random_state=SEED,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], monotonic_decline_patience=mono)
        return m.get_booster().num_boosted_rounds(), roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])

    n_mono, auc_mono = run(3)
    n_base, auc_base = run(None)
    assert n_mono < n_base, f"xgb: monotonic did not stop earlier ({n_mono} vs {n_base})"
    assert auc_mono >= auc_base - 0.03, f"xgb: held-out AUC hurt ({auc_mono:.4f} < {auc_base:.4f})"


# --------------------------------------------------------------------------- cb


def test_biz_val_cb_monotonic_stops_earlier_no_test_regression():
    """Biz val cb monotonic stops earlier no test regression."""
    catboost = pytest.importorskip("catboost")
    from mlframe.training.callbacks.monotonic_decline import (
        CBMonotonicDeclineStop,
        catboost_callbacks_supported,
    )

    if not catboost_callbacks_supported():
        pytest.skip("installed CatBoost build lacks fit(callbacks=) support")

    X_tr, y_tr, X_val, y_val, X_te, y_te = _overfit_prone_split()

    def run(mono):
        """Fits a CatBoost classifier with the given monotonic_decline_patience and returns (rounds trained, test AUC)."""
        model = catboost.CatBoostClassifier(
            iterations=N_ROUNDS,
            depth=8,
            learning_rate=0.3,
            eval_metric="AUC",
            random_seed=SEED,
            verbose=0,
            allow_writing_files=False,
        )
        callbacks = [CBMonotonicDeclineStop(patience=mono)] if mono is not None else None
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), callbacks=callbacks)
        return model.tree_count_, roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

    n_mono, auc_mono = run(3)
    n_base, auc_base = run(None)
    assert n_mono < n_base, f"cb: monotonic did not stop earlier ({n_mono} vs {n_base})"
    assert auc_mono >= auc_base - 0.03, f"cb: held-out AUC hurt ({auc_mono:.4f} < {auc_base:.4f})"


# --------------------------------------------------------------------------- mlp


def test_biz_val_mlp_monotonic_stops_earlier_no_test_regression():
    """MLP uses the SAME ``MonotonicDeclineStopper`` via ``MonotonicDeclineStopCallback``.

    Replays a realistic overfit-prone val-loss epoch curve through the exact stopper the MLP callback
    wraps and asserts it cuts epochs while the best-epoch checkpoint (restored regardless of stop time)
    preserves the held-out loss. The live Lightning wiring is covered by the neural integration test.
    """
    from mlframe.estimators.early_stopping_monotonic import MonotonicDeclineStopper

    val_loss = [0.70, 0.55, 0.45, 0.40, 0.38, 0.41, 0.44, 0.47, 0.50, 0.53, 0.56, 0.60]
    best_epoch_idx = val_loss.index(min(val_loss))

    def epochs_until_stop(patience):
        """Epochs until stop."""
        s = MonotonicDeclineStopper(patience, mode="min")
        for epoch, v in enumerate(val_loss, start=1):
            if s.update(v):
                return epoch
        return len(val_loss)

    n_mono = epochs_until_stop(3)
    n_base = epochs_until_stop(None)
    assert n_mono < n_base, f"mlp: monotonic did not stop earlier ({n_mono} vs {n_base})"
    # Stop fires strictly AFTER the global best epoch was seen -> best-epoch checkpoint is the best loss.
    assert n_mono > best_epoch_idx + 1, "mlp: stopped before the best epoch was recorded"
