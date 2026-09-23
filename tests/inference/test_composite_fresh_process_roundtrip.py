"""Every composite wrapper saved in-process loads and predicts identically in a fresh interpreter.

Auto-chain transforms existed only in the training process's registry, so a pickled chain composite raised
``UnknownTransformError`` in a new interpreter, and no test ever loaded a composite anywhere but the process that built it.
Here every registry transform and every name the auto-chain proposer can generate is wrapped, saved through
``save_mlframe_model``, and loaded + predicted in ONE subprocess through the production loader.
"""

from __future__ import annotations

import orjson
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from mlframe.training.composite import CompositeCrossTargetEnsemble, CompositeTargetEstimator
from mlframe.training.composite.discovery._auto_chain import _RESIDUAL_STAGE_NAMES, _TAIL_UNARIES, reregister_auto_chain_transforms
from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY, get_transform, list_transforms

_ROOT = Path(__file__).resolve().parents[2]
_CHAIN_NAMES = sorted(f"chain_{r}_{u}" for r in _RESIDUAL_STAGE_NAMES for u in _TAIL_UNARIES)


def _frame(n: int = 300, seed: int = 0):
    """Positive y and bases, a noise feature and a group column."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"b": rng.uniform(2.0, 10.0, n), "b2": rng.uniform(1.0, 5.0, n), "x": rng.normal(size=n), "g": np.arange(n) % 6})
    y = 2.0 * X["b"].to_numpy() + 0.5 * X["b2"].to_numpy() + 3.0 + rng.normal(0.0, 0.2, n)
    return X, y


def _build(name: str, X: pd.DataFrame, y: np.ndarray):
    """A fitted wrapper for transform ``name`` around a linear inner, or None when the fixture is outside its domain."""
    t = get_transform(name)
    base = X[["b", "b2"]].to_numpy() if t.n_bases > 1 else X["b"].to_numpy()
    kw = {"groups": X["g"].to_numpy()} if t.requires_groups else {}
    params = t.fit(y, base if t.requires_base else None, **kw)
    T = np.asarray(t.forward(y, base if t.requires_base else None, params, **kw), dtype=np.float64)
    ok = np.isfinite(T)
    if ok.sum() < 50:
        return None
    # Selects ``x`` by name, so it serves grouped wrappers (which drop the group column) and the rest alike.
    inner = make_pipeline(ColumnTransformer([("x", "passthrough", ["x"])], remainder="drop"), LinearRegression()).fit(X.loc[ok], T[ok])
    return CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner, transform_name=name, base_column="b", transform_fitted_params=params, y_train=y,
        base_columns=("b", "b2") if t.n_bases > 1 else None, base_train=X["b"].to_numpy(),
        group_column="g" if t.requires_groups else None,
    )


def test_every_composite_loads_and_predicts_identically_in_a_fresh_process(tmp_path):
    """One subprocess loads every saved wrapper (and a CT ensemble of two) through the production loader and matches."""
    from mlframe.training._io_save import save_mlframe_model

    reregister_auto_chain_transforms(_CHAIN_NAMES)
    X, y = _frame()
    X_pred = X.iloc[:40]
    expected: dict[str, list] = {}
    built = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in sorted(set(list_transforms()) | set(_CHAIN_NAMES)):
            est = _build(name, X, y)
            if est is None:
                continue
            built[name] = est
            save_mlframe_model(est, str(tmp_path / f"{name}.dump"), verbose=0)
            expected[name] = np.asarray(est.predict(X_pred), dtype=np.float64).tolist()
        pair = [built["linear_residual"], built["chain_linear_residual_cbrt"]]
        P = np.column_stack([m.predict(X) for m in pair])
        ens = CompositeCrossTargetEnsemble.from_nnls_stack(component_models=pair, component_names=["a", "b"], component_predictions=P, y_train=y)
        save_mlframe_model(ens, str(tmp_path / "ct_ensemble.dump"), verbose=0)
        expected["ct_ensemble"] = np.asarray(ens.predict(X_pred), dtype=np.float64).tolist()
    assert len(built) >= 0.9 * (len(list_transforms()) + len(_CHAIN_NAMES)), f"only {len(built)} wrappers could be built"
    assert all(c in built for c in _CHAIN_NAMES), sorted(set(_CHAIN_NAMES) - set(built))
    X_pred.to_pickle(tmp_path / "X_pred.pkl")
    (tmp_path / "expected.json").write_text(orjson.dumps(expected).decode(), encoding="utf-8")

    script = textwrap.dedent(f"""
        import orjson, sys
        import numpy as np, pandas as pd
        from mlframe.training.io import load_mlframe_model
        d = {str(tmp_path)!r}
        X = pd.read_pickle(d + "/X_pred.pkl")
        expected = orjson.loads(open(d + "/expected.json", encoding="utf-8").read())
        bad = []
        for name, want in expected.items():
            try:
                model = load_mlframe_model(d + "/" + name + ".dump")
                got = np.asarray(model.predict(X), dtype=np.float64)
                if not np.allclose(got, np.asarray(want), rtol=1e-9, atol=1e-9, equal_nan=True):
                    bad.append((name, "predictions differ"))
            except Exception as err:
                bad.append((name, type(err).__name__ + ": " + str(err)[:150]))
        print(orjson.dumps(bad).decode())
    """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(_ROOT / "src"), str(_ROOT)]))
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=600, check=False)
    assert out.returncode == 0, out.stderr[-3000:]
    bad = orjson.loads(out.stdout.strip().splitlines()[-1])
    assert not bad, f"{len(bad)} of {len(expected)} models failed the fresh-process round trip: {bad}"


def test_the_chain_name_space_is_the_proposers():
    """The chain names covered are every (residual stage, tail unary) pair the auto-chain proposer can emit."""
    assert len(_CHAIN_NAMES) == len(_RESIDUAL_STAGE_NAMES) * len(_TAIL_UNARIES) and len(_CHAIN_NAMES) >= 6
    assert all(r in TRANSFORMS_REGISTRY for r in _RESIDUAL_STAGE_NAMES)


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    """Run from the temp dir so no dump lands in the repository."""
    monkeypatch.chdir(tmp_path)
