"""Continuous engineered-feature transform output -> downstream linear models can
USE the magnitude of a synthesized feature (regression + biz-value pin, 2026-06-12).

Root cause this file pins
-------------------------
MRMR correctly synthesizes the dominant engineered feature on the user's CASE2
target ``y = 0.2*a**2/b + f/5 + log(2c)*sin(d/3)`` (``f`` hidden) -- it picks
``div(sqr(a),abs(b))`` = ``a**2/b``. But ``transform()`` USED to deliver that
``unary_binary`` column as the internal MI **quantile code** (10 integer bins),
not its continuous value. Binning a heavy-tailed product to 10 codes preserves
RANK but discards MAGNITUDE: measured Pearson 0.03 between the 10-bin code and
the true ``a**2/b`` (rank-corr 0.99). A linear model can recover ``0.2*a**2/b``
EXACTLY from the continuous feature (the relationship is perfectly linear) but is
helpless on a 10-level rank code -- its predictions cap far below the heavy tail
(target reaches ~3133, code-model predictions ~2.7) and test-R2 collapses to
~0.002 despite the perfect feature being selected.

The fix mirrors the pre-existing ``prewarp`` / ``hermite_pair`` recipe siblings,
which already skip quantization at replay for the same reason: ``transform()`` now
emits the CONTINUOUS engineered value for ``unary_binary`` and ``cluster_aggregate``
recipes. The downstream MRMR fit still discretises the fit-time column for its OWN
MI matrix via ``_mrmr_fe_step`` (a separate path, unaffected) -- this is a
replay-only change.

Two pins
--------
* ``test_engineered_unary_binary_transform_is_continuous`` -- the DETERMINISTIC
  regression guard. Whenever MRMR produces a numeric pair-FE column on CASE2, that
  column must be continuous (high cardinality, float, |Pearson| ~ 1 against the
  true ``a**2/b``), never a low-cardinality integer code. Fast, no model training.
* ``test_suite_linear_reaches_r2_099_on_case2_via_mrmr`` -- the biz-value proof the
  user asked for: ``train_mlframe_models_suite(model="linear")`` with MRMR feature
  selection reaches test-R2 >= 0.99 on the user's exact CASE2. Pre-fix this scored
  ~0.002 (the quantized feature is unusable by a linear model); post-fix ~0.9999.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.conftest import is_fast_mode


# ---------------------------------------------------------------------------
# CASE2 -- the user's exact reproduction (uniform inputs; ``f`` is a HIDDEN noise
# variable present in y but NOT in X; ``e`` is a pure-noise feature in X).
# ---------------------------------------------------------------------------


def _build_case2(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.random(n)
    b = rng.random(n)
    c = rng.random(n)
    d = rng.random(n)
    e = rng.random(n)
    f = rng.random(n)
    y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2.0) * np.sin(d / 3.0)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    return df, y, (a, b)


def _is_unary_binary_pair_name(name: str) -> bool:
    """A synthesized numeric pair-FE column names a binary op over (possibly
    transformed) operands, e.g. ``div(sqr(a),abs(b))`` / ``div(sqr(a),neg(b))``.
    The fit may sanitise ``,`` -> ``_`` in the materialised column name."""
    s = str(name)
    return s.startswith("div(") or s.startswith("mul(") or s.startswith("sub(") or s.startswith("add(")


# ---------------------------------------------------------------------------
# DETERMINISTIC regression pin -- engineered transform output is continuous.
# ---------------------------------------------------------------------------


def test_engineered_unary_binary_transform_is_continuous():
    """The exact bug: a synthesized ``unary_binary`` column (``div(sqr(a),abs(b))``
    = ``a**2/b``) must come out of ``transform()`` CONTINUOUS -- high cardinality,
    floating dtype, near-perfect |Pearson| with the true ``a**2/b`` -- not as a
    low-cardinality integer quantile code (|Pearson| ~ 0 although rank-corr ~ 1).

    Pre-fix this column was a 10-bin int32 code (nunique==10, Pearson |0.03|);
    post-fix it is the continuous value (nunique==n, |Pearson| ~ 1.0).
    """
    from mlframe.feature_selection.filters import MRMR

    n = 40_000 if is_fast_mode() else 100_000
    df, y, (a, b) = _build_case2(n=n, seed=0)
    true_ab = a**2 / b  # the structural feature the engineered column should equal (up to sign)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(verbose=0, random_seed=0).fit(X=df, y=y)
        out = fs.transform(df)

    names = list(out.columns)
    pair_cols = [c for c in names if _is_unary_binary_pair_name(c)]
    assert pair_cols, (
        f"MRMR did not synthesize a numeric pair-FE feature on CASE2; selected={names}. "
        f"This test pins the CONTINUITY of such a feature when produced -- if FE stopped "
        f"producing it entirely that is a separate regression worth investigating."
    )

    for col in pair_cols:
        vals = np.asarray(out[col], dtype=np.float64)
        nuniq = int(np.unique(vals).size)
        # CONTINUITY: a quantile code has exactly quantization_nbins (~10) distinct
        # values; the continuous feature has ~n. Require far more than any plausible
        # bin count.
        assert nuniq > 1000, (
            f"engineered column {col!r} looks QUANTIZED (nunique={nuniq}); transform must "
            f"emit the continuous value, not the MI bin code. vals[:5]={vals[:5]}"
        )
        assert np.issubdtype(out[col].to_numpy().dtype, np.floating), (
            f"engineered column {col!r} dtype is {out[col].dtype}; expected a floating "
            f"continuous feature, not an integer code."
        )
        # MAGNITUDE preserved: the continuous a**2/b column correlates ~ +-1 with the
        # true a**2/b. The 10-bin code had |Pearson| ~ 0.03 (rank preserved, magnitude
        # destroyed) -- this is the precise discriminator between the bug and the fix.
        pear = abs(float(np.corrcoef(vals, true_ab)[0, 1]))
        assert pear > 0.95, (
            f"engineered column {col!r} has |Pearson|={pear:.3f} with the true a**2/b; "
            f"a quantized rank code scores ~0.03 here. Magnitude was not preserved."
        )


# ---------------------------------------------------------------------------
# Biz-value proof -- the user's headline ask: linear suite R2 >= 0.99 via MRMR.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_suite_linear_reaches_r2_099_on_case2_via_mrmr():
    """``train_mlframe_models_suite(model="linear")`` with MRMR feature selection
    must reach test-R2 >= 0.99 on the user's exact CASE2.

    Why this is the right floor: ``0.2*a**2/b`` has a heavy tail (``1/b``, b->0) and
    accounts for 99.9997% of Var(y) (Var(0.2*a**2/b) ~ 1367 vs Var(log*sin) ~ 0.04,
    Var(f/5) ~ 0.003). MRMR synthesizes ``div(sqr(a),abs(b))`` = ``a**2/b`` exactly,
    so a linear model recovers ``0.2*a**2/b`` essentially perfectly -- IF the feature
    reaches the model with its magnitude intact. The continuous-transform-output fix
    is exactly what makes that true; pre-fix the quantized 10-bin code dropped R2 to
    ~0.002.

    Pinned at the user's n=100k (deterministic for a fixed seed). The companion
    deterministic test above guards the underlying continuity invariant for fast mode.
    """
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import (
        BaselineDiagnosticsConfig,
        CompositeTargetDiscoveryConfig,
        DummyBaselinesConfig,
        OutputConfig,
        ReportingConfig,
    )
    from mlframe.training._feature_selection_config import FeatureSelectionConfig
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    n = 100_000
    df, y, _ = _build_case2(n=n, seed=0)
    df = df.copy()
    df["y"] = y.astype("float64")

    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models, meta = train_mlframe_models_suite(
            df=df,
            target_name="y",
            model_name="case2_linear_r2",
            features_and_targets_extractor=fte,
            mlframe_models=["linear"],
            verbose=0,
            use_mlframe_ensembles=False,
            feature_selection_config=FeatureSelectionConfig(
                use_mrmr_fs=True, mrmr_kwargs=dict(verbose=0, random_seed=0)
            ),
            output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
            composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
            baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
            dummy_baselines_config=DummyBaselinesConfig(enabled=False),
            reporting_config=ReportingConfig(
                show_perf_chart=False, show_fi=False, plot_inline_display=False
            ),
        )

    target_type_key = next(iter(models))
    entries = models[target_type_key]["y"]
    r2s = [float(e.metrics.get("test", {}).get("R2", float("-inf"))) for e in entries]
    best = max(r2s)
    selected = meta.get("selected_features")
    assert best >= 0.99, (
        f"linear suite test-R2 should be >= 0.99 on CASE2 via MRMR; got best={best:.6f} "
        f"(per-entry {[round(r, 6) for r in r2s]}). selected_features={selected}. "
        f"A collapse here means the synthesized a**2/b feature is reaching the linear "
        f"model magnitude-stripped (quantized) again, or MRMR failed to synthesize it."
    )
