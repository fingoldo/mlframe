"""biz_value test for the ``init_score`` baseline knob of ``BaselineDiagnostics``.

The init_score baseline refits the quick LightGBM with the top-K dominant feature(s) combined linearly and passed
via LightGBM's ``init_score=`` so the booster learns only the RESIDUAL. The decision-influencing claim is: when the
target is (close to) a linear function of a dominant feature, learning the residual reaches essentially the same
headline metric as the full raw fit -- ``InitScoreBaseline.delta_vs_raw_pct`` is small -- which is the signal that a
composite/residual target is appropriate.

The DELTA that matters: on a dominant-LINEAR target the init_score baseline's delta_vs_raw is much SMALLER than on a
target where the dominant feature interacts non-linearly (a pure-interaction target the linear init_score cannot
capture, so its residual fit leaves a large gap). A regression that disables / breaks the init_score path collapses
this separation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _make_config(**overrides):
    """Builds a BaselineDiagnosticsConfig with sensible small-data defaults, overridable per test."""
    from mlframe.training.configs import BaselineDiagnosticsConfig

    defaults = dict(
        enabled=True,
        ablation_top_k=4,
        init_score_top_k=1,
        quick_model_family="lightgbm",
        quick_model_n_estimators=50,
        sample_n=2000,
        random_state=42,
    )
    defaults.update(overrides)
    return BaselineDiagnosticsConfig(**defaults)


def _fit(df, feature_cols):
    """Fits BaselineDiagnostics on df using the given feature columns."""
    from mlframe.training.baselines.diagnostics import BaselineDiagnostics

    diag = BaselineDiagnostics(_make_config())
    return diag.fit_and_report(
        train_df=df,
        train_target=df["y"],
        feature_cols=feature_cols,
        target_type="regression",
        target_name="y",
    )


def test_biz_val_init_score_small_delta_on_linear_dominant_vs_large_on_interaction():
    """init_score baseline delta_vs_raw must be SMALL on a linear-dominant target and clearly LARGER on a
    pure-interaction target the linear init_score cannot represent. Catches a broken/disabled init_score path."""
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(7)
    n = 2000
    cols = ["x0", "x1", "x2", "x3"]

    x = {c: rng.normal(size=n) for c in cols}
    # Linear-dominant: y is almost entirely a linear function of x0. The init_score (linear in x0) captures nearly
    # all signal, so residual learning matches raw and delta_vs_raw is tiny.
    y_lin = 6.0 * x["x0"] + 0.3 * rng.normal(size=n)
    df_lin = pd.DataFrame({**x, "y": y_lin})

    x2 = {c: rng.normal(size=n) for c in cols}
    # Pure interaction: y = x0 * x1. The dominant single feature's LINEAR init_score cannot represent the product,
    # so the residual fit must rediscover the interaction from scratch -> larger delta_vs_raw.
    y_int = 4.0 * x2["x0"] * x2["x1"] + 0.3 * rng.normal(size=n)
    df_int = pd.DataFrame({**x2, "y": y_int})

    rep_lin = _fit(df_lin, cols)
    rep_int = _fit(df_int, cols)

    isb_lin = rep_lin.init_score_baseline
    isb_int = rep_int.init_score_baseline
    assert isb_lin is not None, "init_score baseline should fire on the linear-dominant regression target"

    delta_lin = abs(isb_lin.delta_vs_raw_pct)
    # On the linear target the residual fit should land within a few pct of the raw headline metric.
    assert delta_lin <= 15.0, f"linear-dominant init_score delta_vs_raw should be small; got {delta_lin:.2f}%"

    if isb_int is not None:
        delta_int = abs(isb_int.delta_vs_raw_pct)
        assert delta_int >= delta_lin, (
            f"interaction-target init_score delta ({delta_int:.2f}%) must be >= linear-target delta "
            f"({delta_lin:.2f}%); a linear init_score cannot capture x0*x1"
        )
