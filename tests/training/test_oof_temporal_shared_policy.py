"""The trainer's OOF pass forward-chains over the train rows' TIMESTAMPS (not row position), and ``oof_has_time`` defaults to the suite's shared split decision."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from mlframe.training.core._cv_policy_setup import _decide_suite_cv_policy
from mlframe.training.trainer import _compute_oof_preds

N = 200


class _Spy(DecisionTreeRegressor):
    """Regressor recording the newest fit row and every predict-time row's time (column 0 carries the timestamp)."""

    log: list = []

    def fit(self, X, y, **kw):
        """Record the newest training timestamp, then fit."""
        self._newest = np.asarray(X)[:, 0].max()
        return super().fit(X, y, **kw)

    def predict(self, X):
        """Log (newest training time, oldest prediction time), then predict."""
        type(self).log.append((self._newest, np.asarray(X)[:, 0].min()))
        return super().predict(X)


def _frame():
    """Shuffled timestamps, a feature frame holding them and a random target."""
    ts = np.random.default_rng(0).permutation(N)
    X = pd.DataFrame({"t": ts.astype(float), "x": np.random.default_rng(1).normal(size=N)})
    return ts, X, np.random.default_rng(2).normal(size=N)


def test_oof_time_folds_follow_timestamps_not_row_position():
    """Oof time folds follow timestamps not row position."""
    ts, X, y = _frame()
    _Spy.log = []
    oof_preds, _ = _compute_oof_preds(model=_Spy(), train_df=X, train_target=y, is_classifier_model=False, n_splits=3, random_seed=0, has_time=True, timestamps=ts)
    assert _Spy.log and all(newest_fit < oldest_test for newest_fit, oldest_test in _Spy.log)
    assert np.isnan(oof_preds).sum() == N // 4 and np.isfinite(oof_preds[np.argsort(ts)[-10:]]).all()


def test_oof_time_folds_without_timestamps_fall_back_to_row_order_positional_chain():
    """Oof time folds without timestamps fall back to row order positional chain."""
    _, X, y = _frame()
    _Spy.log = []
    oof_preds, _ = _compute_oof_preds(model=_Spy(), train_df=X, train_target=y, is_classifier_model=False, n_splits=3, random_seed=0, has_time=True)
    assert oof_preds is not None and np.isnan(oof_preds[: N // 4]).all()


def test_oof_time_with_groups_isolates_groups_in_time_order():
    """Oof time with groups isolates groups in time order."""
    ts = np.arange(N).astype(float)
    groups = np.repeat(np.arange(20), 10)
    X = pd.DataFrame({"t": ts, "x": np.random.default_rng(1).normal(size=N)})
    y = np.random.default_rng(2).normal(size=N)
    _Spy.log = []
    oof_preds, _ = _compute_oof_preds(
        model=_Spy(), train_df=X, train_target=y, is_classifier_model=False, n_splits=3, random_seed=0, has_time=True, timestamps=ts, group_ids=groups,
    )
    assert oof_preds is not None and _Spy.log and all(newest_fit < oldest_test for newest_fit, oldest_test in _Spy.log)


def _split(**kw):
    """Build a split-config namespace with defaults, overridden by kwargs."""
    base = dict(cv_strategy="random", test_size=0.2, val_size=0.2, shuffle_test=False, shuffle_val=False,
                test_sequential_fraction=None, val_sequential_fraction=None, use_groups=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _hp(**kw):
    """Build a hyperparameter namespace that reports its kwargs as explicitly set fields."""
    ns = SimpleNamespace(**kw)
    ns.model_fields_set = set(kw)
    return ns


def test_suite_policy_is_temporal_with_timestamps_and_opt_out_returns_none():
    """Suite policy is temporal with timestamps and opt out returns none."""
    ts = np.arange(50)
    kw = dict(timestamps=ts, train_idx=np.arange(30), group_ids=None, split_config=_split(), hyperparams_config=_hp())
    policy = _decide_suite_cv_policy(feature_selection_config=SimpleNamespace(unified_cv_policy=True), **kw)
    assert policy.temporal and len(policy.timestamps) == 30
    assert _decide_suite_cv_policy(feature_selection_config=SimpleNamespace(unified_cv_policy=False), **kw) is None
    iid = _decide_suite_cv_policy(feature_selection_config=SimpleNamespace(), **{**kw, "hyperparams_config": _hp(has_time=False)})
    assert iid.kind == "iid" and not iid.temporal


def test_behavior_config_oof_has_time_defaults_to_suite_decision():
    """Behavior config oof has time defaults to suite decision."""
    from mlframe.training._model_configs_behavior import TrainingBehaviorConfig

    assert TrainingBehaviorConfig().oof_has_time is None
    assert TrainingBehaviorConfig(oof_has_time=False).oof_has_time is False
