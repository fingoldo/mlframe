"""Rows a transform's params were fit on are never the rows those params score.

A spy replaces a registry transform's ``fit`` and ``inverse``. Every base value is unique, so the set of bases a ``fit``
saw names its rows exactly (compared in float32, the precision discovery reads bases at); each fit stamps its params with its call number, and every ``inverse`` records which fit's
params reconstructed which bases. A scored reconstruction whose bases overlap its params' fit rows is a leak: the
params had already absorbed the rows they are being judged on (DSC-04's tiny CV on all-row params, DSC-13's gate
fallback on params that saw the held-out groups, EST-16's OOF refit falling back to the full-train params).

Each path runs twice: as shipped (no overlap) and with its per-fold refit disabled, which must register the leak.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.transforms.registry import _TRANSFORMS_REGISTRY

_TAG = "_spy_fit_id"


class _FitScoreSpy:
    """Record the bases each ``fit`` saw and the bases each ``inverse`` scored with that fit's params."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.original = _TRANSFORMS_REGISTRY[name]
        self.fit_bases: list[np.ndarray] = []
        self.scored: list[tuple[int | None, np.ndarray]] = []

    def transform(self):
        """The registry transform with the spying ``fit`` / ``inverse``, signatures preserved for the call gateway."""
        orig = self.original

        def _fit(y, base, *args, **kwargs):
            params = dict(orig.fit(y, base, *args, **kwargs))
            params[_TAG] = len(self.fit_bases)
            self.fit_bases.append(np.unique(np.asarray(base, dtype=np.float32).ravel()))
            return params

        def _inverse(t, base, params, *args, **kwargs):
            tag = params.get(_TAG) if isinstance(params, dict) else None
            self.scored.append((None if tag is None else int(tag), np.asarray(base, dtype=np.float32).ravel().copy()))
            return orig.inverse(t, base, {k: v for k, v in params.items() if k != _TAG}, *args, **kwargs)

        def _forward(y, base, params, *args, **kwargs):
            return orig.forward(y, base, {k: v for k, v in params.items() if k != _TAG}, *args, **kwargs)

        _fit.__signature__ = inspect.signature(orig.fit)
        _inverse.__signature__ = inspect.signature(orig.inverse)
        _forward.__signature__ = inspect.signature(orig.forward)
        return dataclasses.replace(orig, fit=_fit, inverse=_inverse, forward=_forward)

    def overlaps(self) -> list[int]:
        """Per scored reconstruction, how many of its rows were in its params' fit rows (untagged params count as all)."""
        out = []
        for tag, bases in self.scored:
            if tag is None:
                out.append(int(bases.size))
            else:
                out.append(int(np.isin(bases, self.fit_bases[tag]).sum()))
        return out


@pytest.fixture
def spy(monkeypatch):
    """Install the spy for ``linear_residual`` and ``linear_residual_grouped`` in the registry."""
    spies = {}
    for name in ("linear_residual", "linear_residual_grouped"):
        s = _FitScoreSpy(name)
        monkeypatch.setitem(_TRANSFORMS_REGISTRY, name, s.transform())
        spies[name] = s
    return spies


def _frame(n: int = 600, seed: int = 0):
    """Unique continuous bases, four groups with their own level, and one noise feature."""
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, 4, n)
    base = rng.uniform(1.0, 10.0, n)
    feat = rng.normal(size=n)
    y = (1.0 + 0.3 * grp) * base + 0.5 * feat + rng.normal(0.0, 0.1, n)
    return pd.DataFrame({"base": base, "feat": feat, "grp": grp}), y, grp.astype(np.int64), base


# ---------------------------------------------------------------------------
# Tiny-model CV (DSC-04)
# ---------------------------------------------------------------------------


def _tiny_cv(spy):
    """Run the tiny CV on one linear_residual spec with the all-row params discovery hands in; return the spy."""
    from mlframe.training.composite.discovery._screening_tiny_perbin import _tiny_cv_rmse_y_scale

    X, y, _grp, base = _frame()
    s = spy["linear_residual"]
    t = _TRANSFORMS_REGISTRY["linear_residual"]
    params = t.fit(y, base)  # the all-row params discovery hands in
    s.scored.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rmse = _tiny_cv_rmse_y_scale(y, base, t, params, X[["feat", "grp"]].to_numpy(dtype=float), family="lightgbm", n_estimators=10,
                                     num_leaves=7, learning_rate=0.1, cv_folds=3, random_state=0)
    assert np.isfinite(rmse)
    return s


def test_the_tiny_cv_scores_each_fold_with_params_fit_on_the_other_folds(spy):
    """Each fold is reconstructed with params refit on the other folds."""
    s = _tiny_cv(spy)
    assert s.scored, "no fold reached the inverse"
    assert s.overlaps() == [0] * len(s.scored)


def test_the_canary_sees_the_tiny_cv_reuse_all_row_params(spy, monkeypatch):
    """Without the per-fold refit every fold is scored with params that were fit on it."""
    from mlframe.training.composite.discovery import _screening_tiny_perbin

    monkeypatch.setattr(_screening_tiny_perbin, "refit_transform_on_fold", lambda *a, **k: None)
    s = _tiny_cv(spy)
    assert s.scored and all(o > 0 for o in s.overlaps())


# ---------------------------------------------------------------------------
# OOF holdout predictions (EST-16)
# ---------------------------------------------------------------------------


def _oof(spy, name: str):
    """Run the OOF holdout path over one fitted component of transform ``name``; return its spy."""
    from mlframe.training.composite import CompositeTargetEstimator
    from mlframe.training.composite.ensemble import compute_oof_holdout_predictions

    X, y, grp, base = _frame()
    kw = {"group_column": "grp"} if name.endswith("grouped") else {}
    est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name=name, base_column="base", **kw).fit(X, y)
    spec = {"name": "y-spec-base", "transform_name": name, "base_column": "base", "fitted_params": dict(est.fitted_params_)}
    s = spy[name]
    s.scored.clear()
    _p, _yh, surviving = compute_oof_holdout_predictions(
        component_models=[est], component_names=["c"], component_specs=[spec], train_X=X, y_train_full=y,
        base_train_full_per_spec={"y-spec-base": base}, holdout_frac=0.2, random_state=0, kfold=3, group_ids=grp,
    )
    assert surviving == ["c"]
    return s


@pytest.mark.parametrize("name", ["linear_residual", "linear_residual_grouped"])
def test_oof_rows_are_scored_with_params_that_never_saw_them(spy, name):
    """Every OOF fold and the holdout are reconstructed with params fit without them."""
    s = _oof(spy, name)
    assert s.scored, "the OOF path never inverted a scored fold"
    assert s.overlaps() == [0] * len(s.scored)


def test_the_canary_sees_an_oof_fold_reuse_the_full_train_params(spy, monkeypatch):
    """The refit falling back to the full-train params (EST-16's grouped failure mode) scores rows those params saw."""
    from mlframe.training.composite import ensemble

    monkeypatch.setattr(ensemble, "_refit_fold_params", lambda transform, spec, *a, **k: dict(spec["fitted_params"]))
    s = _oof(spy, "linear_residual")
    assert s.scored and all(o > 0 for o in s.overlaps())


# ---------------------------------------------------------------------------
# y-scale gate, no-val fallback (DSC-13)
# ---------------------------------------------------------------------------


def _gate(spy):
    """The fallback gate over one grouped spec; the shipped params are fit on every row, held-out groups included."""
    from mlframe.training.composite.discovery import CompositeTargetDiscovery
    from mlframe.training.composite.discovery._yscale_holdout_gate import apply_yscale_holdout_gate
    from mlframe.training.composite.spec import CompositeSpec
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 120)
    base = rng.uniform(1.0, 10.0, groups.size)
    df = pd.DataFrame({"group": groups, "base": base, "x1": rng.normal(size=groups.size)})
    y = (1.0 + 4.0 * groups) * base + rng.normal(scale=0.2, size=groups.size)
    df["y"] = y
    t = _TRANSFORMS_REGISTRY["linear_residual_grouped"]
    spec = CompositeSpec(name="y-lrg-base", target_col="y", transform_name="linear_residual_grouped", base_column="base",
                         fitted_params=t.fit(y, base, groups=groups), mi_gain=1.0, mi_y=0.0, mi_t=1.0, valid_domain_frac=1.0,
                         n_train_rows=len(df))
    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, yscale_holdout_gate_enabled=True, yscale_holdout_gate_min_groups=4,
        yscale_holdout_gate_tolerance=1e9, tiny_model_n_estimators=25,
    ))
    disc._group_ids_for_rerank = groups
    s = spy["linear_residual_grouped"]
    s.scored.clear()
    apply_yscale_holdout_gate(disc, df, "y", [spec], ["base", "x1", "group"], np.arange(len(df)), y)
    return s


def test_the_gate_fallback_scores_the_unseen_groups_with_leak_free_params(spy):
    """The gate keeps the worse of the shipped and the refit reconstruction, so one of them must be leak-free."""
    s = _gate(spy)
    assert s.scored, "the gate never inverted its eval rows"
    assert 0 in s.overlaps(), f"every reconstruction used params that saw the eval groups: {s.overlaps()}"


def test_the_canary_sees_the_gate_fallback_score_only_with_the_shipped_params(spy, monkeypatch):
    """Without the fit-group refit the gate scores the held-out groups only with params that saw them."""
    from mlframe.training.composite.discovery import _yscale_holdout_gate

    monkeypatch.setattr(_yscale_holdout_gate, "_fold_local_params", lambda *a, **k: None)
    s = _gate(spy)
    assert s.scored and all(o > 0 for o in s.overlaps())
