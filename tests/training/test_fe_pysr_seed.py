"""Regression: PySR feature engineering must thread ``random_state``
through to ``run_pysr_feature_engineering`` so two fits with the same
seed produce identical equation column set + identical first-row values.

Pre-fix: ``_apply_pysr_fe`` never forwarded ``random_state`` so the
internal ``df.sample(...)`` drew a fresh row subset on every call - the
equation set drifted run-to-run.

To keep the test fast and offline (PySR/Julia is heavy and absent on CI),
this test patches ``run_pysr_feature_engineering`` to a deterministic
spy that records the ``random_state`` argument actually received.
"""
from __future__ import annotations

import sys
import types
import numpy as np
import pandas as pd
import pytest


def _make_toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n).astype(np.float32)
    x2 = rng.normal(size=n).astype(np.float32)
    y = (x1 * x1 + 0.3 * x2 - 0.1).astype(np.float32)
    return pd.DataFrame({"x1": x1, "x2": x2}), y


class _SpyModel:
    """Minimal stand-in for a fitted PySRRegressor.

    ``equations_`` is a tiny dataframe with one ``score`` row so
    ``_apply_pysr_fe`` proceeds past the ``len(eq_df) == 0`` early-out.
    ``predict(df, index)`` returns a column derived from the input frame
    plus the recorded ``random_state`` so identical seeds give identical
    output but different seeds diverge.
    """

    def __init__(self, random_state):
        self.random_state = random_state
        self.equations_ = pd.DataFrame({"score": [1.0]}, index=[0])

    def predict(self, df, index):
        # Output depends on seed so different seeds produce different
        # first-row values - lets the test distinguish "seed plumbed"
        # vs "seed dropped".
        return np.asarray(df["x1"].values, dtype=np.float32) + float(self.random_state or 0)


_SEEN_RANDOM_STATES: list = []


def _spy_run_pysr_feature_engineering(*args, **kwargs):
    rs = kwargs.get("random_state")
    _SEEN_RANDOM_STATES.append(rs)
    return _SpyModel(random_state=rs)


@pytest.fixture
def patched_pysr(monkeypatch):
    """Inject a fake ``mlframe.feature_engineering.bruteforce`` module so the
    pipeline's local ``from ... import run_pysr_feature_engineering`` resolves
    to our spy without ever requiring Julia."""
    _SEEN_RANDOM_STATES.clear()
    mod = types.ModuleType("mlframe.feature_engineering.bruteforce")
    mod.run_pysr_feature_engineering = _spy_run_pysr_feature_engineering
    # Make sure the existing real module (if importable) is shadowed.
    monkeypatch.setitem(sys.modules, "mlframe.feature_engineering.bruteforce", mod)
    yield
    _SEEN_RANDOM_STATES.clear()


def test_pysr_random_state_is_forwarded_from_config(patched_pysr):
    """The spy should observe random_state matching config.random_seed."""
    from mlframe.training.pipeline import _apply_pysr_fe
    from mlframe.training.configs import PreprocessingExtensionsConfig

    cfg = PreprocessingExtensionsConfig(pysr_enabled=True, random_seed=1234)
    df, y = _make_toy()
    _apply_pysr_fe(train_df=df, val_df=None, test_df=None,
                   y_train=y, config=cfg, verbose=0)
    assert _SEEN_RANDOM_STATES, "spy was not invoked"
    assert _SEEN_RANDOM_STATES[-1] == 1234, (
        f"random_state forwarded as {_SEEN_RANDOM_STATES[-1]} but expected 1234"
    )


def test_pysr_same_seed_gives_identical_predictions(patched_pysr):
    """Two calls with same seed -> identical equation column values."""
    from mlframe.training.pipeline import _apply_pysr_fe
    from mlframe.training.configs import PreprocessingExtensionsConfig

    cfg = PreprocessingExtensionsConfig(pysr_enabled=True, random_seed=99)
    df_a, y_a = _make_toy(seed=7)
    df_b, y_b = _make_toy(seed=7)
    cols_a = _apply_pysr_fe(train_df=df_a, val_df=None, test_df=None,
                            y_train=y_a, config=cfg, verbose=0)
    cols_b = _apply_pysr_fe(train_df=df_b, val_df=None, test_df=None,
                            y_train=y_b, config=cfg, verbose=0)
    assert cols_a and cols_b, "spy produced no columns"
    assert set(cols_a) == set(cols_b)
    for c in cols_a:
        assert np.isclose(float(df_a[c].iloc[0]), float(df_b[c].iloc[0]))


def test_pysr_default_seed_when_config_missing_random_seed(patched_pysr):
    """If config has no random_seed attr, _apply_pysr_fe falls back to 42."""
    from mlframe.training.pipeline import _apply_pysr_fe
    from mlframe.training.configs import PreprocessingExtensionsConfig

    cfg = PreprocessingExtensionsConfig(pysr_enabled=True)
    df, y = _make_toy()
    _apply_pysr_fe(train_df=df, val_df=None, test_df=None,
                   y_train=y, config=cfg, verbose=0)
    assert _SEEN_RANDOM_STATES[-1] == 42, (
        f"expected default seed 42, got {_SEEN_RANDOM_STATES[-1]}"
    )
