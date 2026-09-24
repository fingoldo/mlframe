"""On the no-val path, the y-scale gate's verdict does not rest on params that saw the holdout groups (DSC-13).

Without a val frame the gate carves its "unseen group" holdout out of ``screen_idx`` - the same rows the spec's
``fitted_params`` were fit on. Only the tiny model was group-disjoint: the forward and the inverse used alpha/beta,
spline knots and per-group levels that had already absorbed the held-out groups, so the collapse test the gate exists
for came out optimistic exactly where it matters, on grouped and level transforms.

The gate now also reconstructs with params refit on the fit groups alone and keeps the worse of the two scores, so a
spec survives only when it holds up both as it will ship and with the leak removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery import CompositeTargetDiscovery
from mlframe.training.composite.discovery._yscale_holdout_gate import _fold_local_params
from mlframe.training.composite.transforms import get_transform


def _shifted_level_frame(n_groups: int = 10, per_group: int = 120):
    """Groups whose base-to-y level differs, so params fit on all groups describe no single held-out group."""
    rng = np.random.default_rng(0)
    rows = []
    for g in range(n_groups):
        base = rng.uniform(1.0, 10.0, per_group)
        rows.append(pd.DataFrame({
            "group": g, "base": base, "x1": rng.normal(size=per_group),
            "y": (1.0 + 4.0 * g) * base + rng.normal(scale=0.2, size=per_group),
        }))
    df = pd.concat(rows, ignore_index=True)
    return df, df["group"].to_numpy(), df["y"].to_numpy(dtype=float)


def test_the_refit_params_describe_the_fit_groups_not_the_pooled_frame():
    """``_fold_local_params`` re-estimates alpha on the fit rows only, away from the pooled-frame value."""
    df, groups, y = _shifted_level_frame()
    fit = groups < 5
    transform = get_transform("linear_residual")
    y_fit, base_fit = y[fit], df["base"].to_numpy()[fit]
    pooled_alpha = float(transform.fit(y, df["base"].to_numpy())["alpha"])

    refit = _fold_local_params(transform, y_fit, base_fit, np.ones(y_fit.shape, dtype=bool), groups[fit])
    assert refit is not None
    fold_params, valid = refit
    assert valid.all()
    assert abs(fold_params["alpha"] - pooled_alpha) > 1.0, (fold_params["alpha"], pooled_alpha)


def test_the_val_split_path_keeps_the_shipped_params():
    """With a val frame there is nothing to refit: the params were fit on train and the eval rows come from elsewhere."""
    df, _groups, y = _shifted_level_frame()
    transform = get_transform("linear_residual")
    assert _fold_local_params(transform, y, df["base"].to_numpy(), np.ones(y.shape, dtype=bool), None) is None


def _grouped_spec(df, y, groups):
    """A ``linear_residual_grouped`` spec whose per-group levels were fit on every group, held-out ones included."""
    from mlframe.training.composite.spec import CompositeSpec

    params = get_transform("linear_residual_grouped").fit(y, df["base"].to_numpy(), groups=groups)
    return CompositeSpec(
        name="y-linres-grouped-base", target_col="y", transform_name="linear_residual_grouped", base_column="base",
        fitted_params=params, mi_gain=1.0, mi_y=0.0, mi_t=1.0, valid_domain_frac=1.0, n_train_rows=len(df),
    )


def _gate(df, y, groups, *, tolerance: float):
    """Run the fallback-path gate over one grouped spec and return ``(survivors, discovery)``."""
    from mlframe.training.composite.discovery._yscale_holdout_gate import apply_yscale_holdout_gate
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, yscale_holdout_gate_enabled=True, yscale_holdout_gate_min_groups=4,
        yscale_holdout_gate_tolerance=tolerance, tiny_model_n_estimators=25,
    ))
    disc._group_ids_for_rerank = groups
    # ``group`` is a feature, so the raw-y baseline can learn the per-group level: the comparison is not "both sides
    # fail on an unseen group", it is "the composite side had the answer in its own parameters".
    return apply_yscale_holdout_gate(disc, df, "y", [_grouped_spec(df, y, groups)], ["base", "x1", "group"], np.arange(len(df)), y), disc


def test_the_recorded_rmse_is_the_worse_of_the_shipped_and_the_leak_free_reconstruction(monkeypatch):
    """The verdict number cannot come from params that saw the holdout groups: it is the worse of the two runs."""
    from mlframe.training.composite.discovery import _yscale_holdout_gate as gate_mod

    df, groups, y = _shifted_level_frame()
    survivors, _ = _gate(df, y, groups, tolerance=1e9)
    assert survivors, "the tolerance is wide open here; the spec must reach the stamping step"
    honest = float(survivors[0].yscale_holdout_rmse)

    monkeypatch.setattr(gate_mod, "_fold_local_params", lambda *a, **k: None)  # the pre-fix behaviour: shipped params only
    leaky_survivors, _ = _gate(df, y, groups, tolerance=1e9)
    leaky = float(leaky_survivors[0].yscale_holdout_rmse)
    assert honest > 10 * leaky, (honest, leaky)


def test_a_grouped_spec_is_scored_rather_than_rejected_for_want_of_group_labels():
    """The forward and the inverse get the group labels; without them every grouped spec read as a collapsed inverse."""
    df, groups, y = _shifted_level_frame()
    _survivors, disc = _gate(df, y, groups, tolerance=1e9)
    reasons = [r.get("reason", "") for r in disc.rejection_ledger]
    assert not any("groups kwarg is required" in r for r in reasons), reasons


def test_the_leaky_grouped_spec_does_not_survive_the_default_tolerance():
    """At the shipped tolerance the spec whose per-group levels saw the holdout is dropped, and the ledger says why."""
    df, groups, y = _shifted_level_frame()
    survivors, disc = _gate(df, y, groups, tolerance=1.10)
    assert [s.name for s in survivors] == []
    assert any(r["stage"] == "yscale_holdout" for r in disc.rejection_ledger)
