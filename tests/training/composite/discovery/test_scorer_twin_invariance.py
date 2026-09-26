"""Every per-spec scorer in composite discovery is registered, and a y-scale scorer cannot tell a transform from its twin.

A transform ``t`` and its scaled twin (``T x 100`` forward, ``/ 100`` before the inverse) reconstruct the same y, so a
score in y units must be the same for both: a scorer that differs is reading the T scale, and would rank a compressive
transform above an additive one for shrinking T (the WAIC defect, DSC-10). T-scale scorers exist on purpose (WAIC, the
region-adaptive variance ratio); each is registered with the guard that keeps it from comparing different T scales.

``SPEC_SCORERS`` holds every discovery function whose name marks a score (``*_rmse*``, ``*waic*``, ``*mi_gain*``,
``*_score*``); the meta-guard fails on a new one until it is registered.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

import mlframe
from mlframe.training.composite.transforms import get_transform

pytest.importorskip("lightgbm")

_DISCOVERY = Path(mlframe.__file__).resolve().parent / "training" / "composite" / "discovery"
_SCORER_NAME = re.compile(r"_rmse|waic|mi_gain|_score")
_K = 100.0


def scaled_twin(t, k: float = _K):
    """``t`` with its T multiplied by ``k``: the same y reconstruction, a different T scale."""
    fields = {f.name for f in dataclasses.fields(t)}
    extra = {"additive_in_t": False} if "additive_in_t" in fields else {}
    return dataclasses.replace(
        t, name=f"{t.name}_x{k:g}",
        forward=lambda y, base, params, **kw: k * np.asarray(t.forward(y, base, params, **kw), dtype=np.float64),
        inverse=lambda T, base, params, **kw: t.inverse(np.asarray(T, dtype=np.float64) / k, base, params, **kw),
        **extra,
    )


def _data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(1.0, 5.0, n)
    x = rng.normal(size=(n, 4))
    y = 2.0 * base + np.sin(2 * x[:, 0]) + 0.3 * rng.normal(size=n)
    return y, base, x


def _first(v):
    return float(v[0] if isinstance(v, tuple) else v)


def _tiny_cv(transform, family):
    from mlframe.training.composite.discovery._screening_tiny_perbin import _tiny_cv_rmse_y_scale

    y, base, x = _data()
    params = transform.fit(y, base)
    return _first(_tiny_cv_rmse_y_scale(y, base, transform, params, x, family=family, n_estimators=40, num_leaves=7, learning_rate=0.1,
                                        cv_folds=3, random_state=0, deterministic=True))


def _auto_chain_cv(transform):
    from mlframe.training.composite.discovery._auto_chain import _y_scale_cv_rmse

    y, base, x = _data()
    return _first(_y_scale_cv_rmse(transform, y=y, base=base, x_matrix=x, cv_folds=3, random_state=0, family="linear",
                                   n_estimators=40, num_leaves=7, learning_rate=0.1))


def _mi_gain(transform):
    from mlframe.training.composite.discovery._auto_chain import _mi_gain_of

    y, base, x = _data()
    return _mi_gain_of(transform, y=y, base=base, x_matrix=x, mi_y=0.0, mi_estimator="bin", mi_nbins=16, mi_n_neighbors=3, random_state=0)


def _calibration(k):
    from mlframe.training.composite.discovery._calibration_gate import calibration_adjusted_score

    rng = np.random.default_rng(1)
    oof, infold = rng.normal(0.1, 1.2, 500), rng.normal(0.0, 1.0, 500)
    return calibration_adjusted_score(0.3, k * oof, k * infold).adjusted_score


# name -> ("y_scale", scorer(transform) -> value) | ("t_scale", the guard that keeps it to one T scale) | ("n/a", why)
SPEC_SCORERS = {
    "_screening_tiny_perbin._tiny_cv_rmse_y_scale": ("y_scale", lambda t: _tiny_cv(t, "linear")),
    "_screening_tiny_perbin._tiny_cv_rmse_y_scale[lgb]": ("y_scale", lambda t: _tiny_cv(t, "lgb")),
    "_auto_chain._y_scale_cv_rmse": ("y_scale", _auto_chain_cv),
    "_auto_chain._mi_gain_of": ("y_scale", _mi_gain),  # MI on quantile bins: rank-based, so invariant to a positive rescale
    "_calibration_gate.calibration_adjusted_score": ("y_scale", None),  # residual-scale invariance, checked below
    "_eval_waic.compute_transform_waic": ("t_scale", "test_scorer_invariance.py::test_waic_reorders_only_bands_whose_members_share_the_y_scale"),
    "_eval_waic.waic_from_oof_residuals": ("t_scale", "the density of compute_transform_waic; same guard"),
    "_eval_waic.rank_transforms_by_waic": ("t_scale", "orders by compute_transform_waic; callers pass one T scale (the band guard)"),
    "_tiny_rerank_waic._apply_waic_tiebreak": ("t_scale", "re-orders only bands whose members are all additive in T (_additive_in_t)"),
    "_tiny_rerank_waic._waic_for": ("t_scale", "scores members of those bands only"),
    "_region_adaptive._oof_score_transform": ("t_scale", "fit_region_adaptive refuses candidates that are not additive in T"),
    "_screening_tiny._tiny_cv_rmse_y_scale_multiseed": ("n/a", "seed repeats of _tiny_cv_rmse_y_scale, registered above"),
    "_screening_tiny_perbin._per_bin_rmse": ("n/a", "per-bin breakdown of the y-scale predictions _tiny_cv_rmse_y_scale makes"),
    "_auto_chain._single_stage_rmses": ("n/a", "calls _y_scale_cv_rmse for the chain's stages"),
    "_honest_oof_select.honest_oof_reconstruction_rmse": ("n/a", "invert-to-y RMSE on the holdout through the same inverse path; pinned by test_scorer_invariance"),
    "_honest_oof_select._score_one": ("n/a", "the per-spec body of honest_oof_reconstruction_rmse"),
    "_honest_rmse_gate.apply_honest_rmse_gate": ("n/a", "y-scale RMSE of inverted predictions against the raw-y baseline"),
    "_honest_rmse_gate._paired_rmse_gain_se": ("n/a", "standard error of a y-scale RMSE gain from paired residuals"),
    "_honest_rmse_gate._move_rmse_stamps_to_selection": ("n/a", "moves recorded numbers between fields; scores nothing"),
    "_screening_tiny._tiny_cv_rmse_raw_y": ("n/a", "raw-y baseline: no transform"),
    "_screening_tiny._tiny_cv_rmse_raw_y_multiseed": ("n/a", "raw-y baseline: no transform"),
    "_causal_lag.causal_lag_predict_rmse": ("n/a", "raw-y AR failsafe baseline: no transform"),
    "_structural_hints.structural_affinity_scores": ("n/a", "scores feature columns, before any transform"),
    "forward_stepwise._cv_rmse_with_folds": ("n/a", "y-scale CV RMSE of base subsets for one linear residual"),
    "__init__.tiny_rerank_scores_": ("n/a", "read-only view of the rerank's scores"),
    "__init__.raw_y_baseline_rmse_": ("n/a", "read-only view of the raw-y baseline"),
}


def _scorer_functions() -> set[str]:
    out = set()
    for path in _DISCOVERY.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _SCORER_NAME.search(node.name):
                out.add(f"{path.stem}.{node.name}")
    return out


def test_every_scorer_named_function_is_registered():
    registered = {k.split("[")[0] for k in SPEC_SCORERS}
    found = _scorer_functions()
    assert not (found - registered), f"register these in SPEC_SCORERS (y-scale runner, T-scale guard, or why n/a): {sorted(found - registered)}"
    assert not (registered - found), f"stale SPEC_SCORERS entries: {sorted(registered - found)}"


@pytest.mark.parametrize("name", sorted(k for k, (kind, run) in SPEC_SCORERS.items() if kind == "y_scale" and run is not None))
@pytest.mark.parametrize("transform_name", ["linear_residual", "diff"])
def test_a_y_scale_scorer_gives_a_transform_and_its_scaled_twin_the_same_score(name, transform_name):
    run = SPEC_SCORERS[name][1]
    t = get_transform(transform_name)
    a, b = run(t), run(scaled_twin(t))
    assert np.isfinite(a), (name, a)
    np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-9, err_msg=f"{name} reads the T scale")


def test_the_calibration_score_is_invariant_to_the_residual_scale():
    np.testing.assert_allclose(_calibration(_K), _calibration(1.0), rtol=1e-9)


def test_the_region_adaptive_fit_refuses_a_candidate_of_another_t_scale():
    from mlframe.training.composite.discovery._region_adaptive import fit_region_adaptive

    y, base, _ = _data()
    with pytest.raises(ValueError, match="additive in T"):
        fit_region_adaptive(y, base, candidates=("linear_residual", "logratio"))
    spec = fit_region_adaptive(y, base)  # the default candidates are all additive in T, so the fit goes through
    assert len(spec.region_transforms) >= 1 and "logratio" not in spec.region_transforms, spec.region_transforms


def test_the_scaled_twin_reconstructs_the_same_y():
    y, base, _ = _data(200)
    t = get_transform("linear_residual")
    p = t.fit(y, base)
    tw = scaled_twin(t)
    T = tw.forward(y, base, p)
    np.testing.assert_allclose(T, _K * np.asarray(t.forward(y, base, p)))
    np.testing.assert_allclose(tw.inverse(T, base, p), y, rtol=1e-9)
