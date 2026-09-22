"""Every registered transform's wrapper must survive pickling, through both construction paths and a fresh process.

The earlier pickle gate covered 5 of 51 transforms, all built through ``.fit()`` and all in-process, so a params-heavy
transform (knot arrays, phase means, rank tables) losing state in ``__getstate__``, the ``from_fitted_inner`` path the
suite actually uses, or an auto-chain transform that exists only in the training process's registry, went untested.
"""

from __future__ import annotations

import os
import pickle  # nosec B403 - test-only local round trip, never untrusted data
import subprocess  # nosec B404 - the fresh-process contract needs a real second interpreter
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms import get_transform, list_transforms


def _frame(n: int = 900, seed: int = 0):
    """A positive, time-ordered frame with a second base and a group key, so every transform family can fit."""
    rng = np.random.default_rng(seed)
    base = np.linspace(1.0, 20.0, n) + rng.normal(0.0, 0.05, n)
    base2 = rng.uniform(1.0, 5.0, n)
    feat = rng.normal(size=n)
    grp = rng.integers(0, 3, n)
    y = 0.7 * base + 0.3 * base2 + 0.4 * feat + rng.normal(0.0, 0.1, n) + 2.0
    return pd.DataFrame({"base": base, "base2": base2, "feat": feat, "grp": grp}), y


def _kwargs(name: str) -> dict:
    """Construction kwargs a transform needs beyond the defaults."""
    kw: dict = {"base_column": "base"}
    if get_transform(name).requires_groups:
        kw["group_column"] = "grp"
    if "multi" in name:
        kw["base_columns"] = ("base", "base2")
    return kw


def _fitted(name: str):
    """A wrapper fitted on the frame through ``.fit()``."""
    X, y = _frame()
    est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name=name, **_kwargs(name)).fit(X, y)
    return est, X, y


@pytest.mark.parametrize("name", list_transforms())
def test_a_fitted_wrapper_predicts_identically_after_pickling(name):
    """Bit-identical predictions and the same fitted params after a pickle round trip, for every transform."""
    est, X, _ = _fitted(name)
    before = np.asarray(est.predict(X))
    restored = pickle.loads(pickle.dumps(est))  # nosec B301 -- round-trip of a locally-created, trusted object
    np.testing.assert_array_equal(np.asarray(restored.predict(X)), before)
    assert restored.fitted_params_.keys() == est.fitted_params_.keys()


@pytest.mark.parametrize("name", list_transforms())
def test_a_wrapper_built_from_a_fitted_inner_round_trips(name):
    """The suite builds wrappers with ``from_fitted_inner``; that path must predict like ``.fit()`` and survive pickling."""
    est, X, y = _fitted(name)
    kw = _kwargs(name)
    built = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=est.estimator_, transform_name=name, base_column=kw["base_column"],
        base_columns=kw.get("base_columns"), group_column=kw.get("group_column"),
        transform_fitted_params=est.fitted_params_, y_train=y,
        base_train=np.asarray(X["base"], dtype=np.float64),
    )
    direct = np.asarray(built.predict(X))
    restored = pickle.loads(pickle.dumps(built))  # nosec B301 -- round-trip of a locally-created, trusted object
    np.testing.assert_array_equal(np.asarray(restored.predict(X)), direct)
    assert np.isfinite(direct).all()


def test_an_auto_chain_wrapper_predicts_in_a_fresh_process(tmp_path, request):
    """A chain transform exists only in the registry of the process that built it; unpickling elsewhere must restore it."""
    from mlframe.training.composite.discovery._auto_chain import reregister_auto_chain_transforms

    from mlframe.training.composite.transforms.registry import _TRANSFORMS_REGISTRY

    chain_name = "chain_linear_residual_yj"
    was_registered = chain_name in _TRANSFORMS_REGISTRY
    assert chain_name in reregister_auto_chain_transforms([chain_name]) or was_registered
    if not was_registered:  # leave the process-wide registry as this test found it
        request.addfinalizer(lambda: _TRANSFORMS_REGISTRY.pop(chain_name, None))
    X, y = _frame()
    est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name=chain_name, base_column="base").fit(X, y)
    model_path, frame_path, preds_path = tmp_path / "chain.pkl", tmp_path / "X.pkl", tmp_path / "preds.npy"
    model_path.write_bytes(pickle.dumps(est))
    X.to_pickle(frame_path)
    script = (
        "import pickle, numpy as np, pandas as pd\n"
        f"est = pickle.loads(open(r'{model_path}', 'rb').read())\n"
        f"X = pd.read_pickle(r'{frame_path}')\n"
        f"np.save(r'{preds_path}', np.asarray(est.predict(X)))\n"
    )
    env = dict(os.environ)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=600)  # nosec B603 - fixed interpreter and generated script
    assert proc.returncode == 0, f"fresh process failed to load or predict:\n{proc.stderr[-2000:]}"
    np.testing.assert_array_equal(np.load(preds_path), np.asarray(est.predict(X)))
