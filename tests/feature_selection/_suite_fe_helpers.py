"""Shared scaffold for END-TO-END ``train_mlframe_models_suite`` feature-engineering
quality tests (2026-06-12).

``train_mlframe_models_suite`` is the library's main entry point, and the
quantized-transform-output bug (engineered features handed to a downstream model as
~10-level rank codes instead of their continuous value) only surfaced as a
FAILED test-split R2 there -- every isolated unit test stayed green. This module
provides the proven building blocks so the suite-entrypoint test files stay terse
and reliable:

* ``make_*`` generators for targets whose signal is a SYNTHESIZABLE feature MRMR's
  FE can build (a ratio, a product, a polynomial, a trig product, a cluster
  aggregate), each in a chosen input ``distribution``. Every generator returns
  ``(df, meta)`` where ``df`` includes the target column ``y`` and ``meta`` records
  the recoverable structure + a SAFE n (heavy-tailed ratios need ~100k for stable
  FE synthesis; bounded products are stable smaller -- see
  ``MRMR_FE_TEST_GAP_ANALYSIS.md``).
* ``run_suite`` -- one call that runs the suite with a chosen model + optional MRMR
  feature selection and returns the fitted entries + the per-split metrics.
* ``best_test_metric`` / ``prediction_span_fraction`` -- the assertions today's bug
  needed: a LINEAR model on a magnitude-carrying engineered feature must reach high
  test-R2, and its predictions must SPAN the target (the bug capped predictions far
  below the heavy tail).

Keep n modest in fast mode; the generators expose ``safe_n`` so callers can scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# input distributions
# ---------------------------------------------------------------------------
def draw(rng: np.random.Generator, distribution: str, n: int) -> np.ndarray:
    if distribution == "normal":
        return rng.standard_normal(n)
    if distribution == "lognormal":
        return rng.lognormal(0.0, 1.0, n)
    if distribution == "heavytail":
        return rng.standard_t(3, n)
    return rng.random(n)  # uniform [0,1)


def _pos(x: np.ndarray) -> np.ndarray:
    return np.abs(x) + 1e-3


@dataclass
class SuiteCase:
    df: pd.DataFrame
    target: str
    # the recoverable structure, for diagnostics / docstrings.
    structure: str
    # a model family for which FE should make the signal accessible.
    safe_n: int
    distribution: str
    # rough fraction of Var(y) that lives in the dominant synthesizable term.
    dominant_var_frac: float = field(default=0.0)


def _frame(cols: dict, sig: np.ndarray, rng, distribution: str, noise_scale: float = 0.05):
    sig = np.asarray(sig, dtype=float)
    sd = np.nanstd(sig)
    sd = sd if (np.isfinite(sd) and sd > 0) else 1.0
    y = sig + rng.normal(0.0, noise_scale * sd, len(sig))
    data = dict(cols)
    data["e"] = draw(rng, distribution, len(sig))  # pure-noise feature IN X
    data["y"] = y.astype("float64")
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# magnitude-carrying / synthesizable targets
# ---------------------------------------------------------------------------
def make_ratio_heavytail(seed: int, distribution: str = "uniform", n: int | None = None) -> SuiteCase:
    """y = 0.2*a**2/b + f/5 + log(2c)*sin(d/3). The user's CASE2: a HEAVY-TAILED ratio
    (1/b, b->0) dominates Var(y) ~99.99%, so a LINEAR model needs the continuous
    magnitude of ``a**2/b`` (the bug capped predictions and collapsed R2). f hidden."""
    n = n or 100_000
    rng = np.random.default_rng(seed)
    a, b, c, d, f = (draw(rng, distribution, n) for _ in range(5))
    a, b, c, d = _pos(a), _pos(b), _pos(c), d
    sig = 0.2 * a ** 2 / b + f / 5.0 + np.log(c * 2.0) * np.sin(d / 3.0)
    df = _frame({"a": a, "b": b, "c": c, "d": d}, sig, rng, distribution)
    return SuiteCase(df, "y", "0.2*a**2/b (heavy-tail ratio)", n, distribution, 0.999)


def make_bilinear(seed: int, distribution: str = "uniform", n: int | None = None) -> SuiteCase:
    """y = 1.5*a*b + 0.5*g/k. A clean bilinear PRODUCT + a ratio term -- the smooth
    low-raw-MI interaction FE must synthesize (a linear model on raws cannot)."""
    n = n or 40_000
    rng = np.random.default_rng(seed)
    a, b, g, k = (draw(rng, distribution, n) for _ in range(4))
    sig = 1.5 * a * b + 0.5 * g / (_pos(k) + 0.3)
    df = _frame({"a": a, "b": b, "g": g, "k": k}, sig, rng, distribution)
    return SuiteCase(df, "y", "1.5*a*b (bilinear product)", n, distribution, 0.7)


def make_poly(seed: int, distribution: str = "normal", n: int | None = None) -> SuiteCase:
    """y = a**3 - 3a + 0.5*b**2. A pure polynomial: FE's orth-poly / sqr / qubed must
    make it linearly accessible. Defaults to normal inputs (orth-poly territory)."""
    n = n or 40_000
    rng = np.random.default_rng(seed)
    a, b = draw(rng, distribution, n), draw(rng, distribution, n)
    sig = a ** 3 - 3.0 * a + 0.5 * b ** 2
    df = _frame({"a": a, "b": b}, sig, rng, distribution)
    return SuiteCase(df, "y", "a**3-3a + 0.5*b**2 (polynomial)", n, distribution, 0.9)


def make_trig_product(seed: int, distribution: str = "uniform", n: int | None = None) -> SuiteCase:
    """y = log(2c)*sin(d) -- a separable trig/log product (the (c,d) term in isolation)."""
    n = n or 40_000
    rng = np.random.default_rng(seed)
    c, d = _pos(draw(rng, distribution, n)), draw(rng, distribution, n)
    sig = np.log(c * 2.0) * np.sin(d)
    df = _frame({"c": c, "d": d}, sig, rng, distribution)
    return SuiteCase(df, "y", "log(2c)*sin(d) (trig product)", n, distribution, 0.9)


def make_cluster_linear(seed: int, distribution: str = "normal", n: int | None = None) -> SuiteCase:
    """y = 3*mean(cluster) + noise, where m1..m3 are tight copies of one latent. The
    cluster-aggregate FE must deliver the CONTINUOUS aggregate magnitude."""
    n = n or 20_000
    rng = np.random.default_rng(seed)
    latent = draw(rng, distribution, n)
    m1 = latent + 0.1 * rng.standard_normal(n)
    m2 = latent + 0.1 * rng.standard_normal(n)
    m3 = latent + 0.1 * rng.standard_normal(n)
    sig = 3.0 * (m1 + m2 + m3) / 3.0
    df = _frame({"m1": m1, "m2": m2, "m3": m3}, sig, rng, distribution)
    return SuiteCase(df, "y", "3*mean_z(cluster) (cluster aggregate)", n, distribution, 0.95)


GENERATORS = {
    "ratio_heavytail": make_ratio_heavytail,
    "bilinear": make_bilinear,
    "poly": make_poly,
    "trig_product": make_trig_product,
    "cluster_linear": make_cluster_linear,
}


# ---------------------------------------------------------------------------
# suite runner
# ---------------------------------------------------------------------------
def run_suite(df: pd.DataFrame, target: str = "y", *, model: str = "linear",
              use_mrmr: bool = True, mrmr_kwargs: dict | None = None,
              classification: bool = False, random_seed: int = 0,
              behavior_kwargs: dict | None = None):
    """Run ``train_mlframe_models_suite`` with ``model`` + optional MRMR FS. Returns
    ``(entries, metadata)`` where ``entries`` is the list of fitted model entries for
    the (single) target. All the noisy optional stages are off for speed."""
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import (
        BaselineDiagnosticsConfig, CompositeTargetDiscoveryConfig,
        DummyBaselinesConfig, OutputConfig, ReportingConfig, TrainingBehaviorConfig,
    )
    from mlframe.training._feature_selection_config import FeatureSelectionConfig
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    if classification:
        fte = SimpleFeaturesAndTargetsExtractor(classification_targets=[target])
    else:
        fte = SimpleFeaturesAndTargetsExtractor(regression_targets=[target])

    fs_cfg = FeatureSelectionConfig(
        use_mrmr_fs=use_mrmr,
        mrmr_kwargs=dict(verbose=0, random_seed=random_seed, **(mrmr_kwargs or {})) if use_mrmr else None,
    )
    behavior = TrainingBehaviorConfig(**(behavior_kwargs or {}))
    models, metadata = train_mlframe_models_suite(
        df=df, target_name=target, model_name="suite_fe", features_and_targets_extractor=fte,
        mlframe_models=[model], verbose=0, use_mlframe_ensembles=False,
        feature_selection_config=fs_cfg, behavior_config=behavior,
        output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(show_perf_chart=False, show_fi=False, plot_inline_display=False),
    )
    tt_key = next(iter(models))
    entries = models[tt_key][target]
    return entries, metadata


def best_test_metric(entries, key: str = "R2") -> float:
    """Best test-split metric across the fitted entries (R2 for regression, roc_auc
    for classification). Returns -inf if absent."""
    best = float("-inf")
    for e in entries:
        m = getattr(e, "metrics", None) or {}
        bag = m.get("test") if isinstance(m, dict) else None
        if isinstance(bag, dict):
            if key in bag and isinstance(bag[key], (int, float)):
                best = max(best, float(bag[key]))
            else:  # nested {class_label: {metric: v}}
                for md in bag.values():
                    if isinstance(md, dict) and key in md and isinstance(md[key], (int, float)):
                        best = max(best, float(md[key]))
    return best


def prediction_span_fraction(entries) -> float:
    """Max over entries of: (range of test predictions) / (range of test target). The
    quantized-output bug capped a heavy-tail target's predictions at a tiny fraction
    (~0.001) of its true range; a healthy linear fit spans most of it."""
    best = 0.0
    for e in entries:
        tt = getattr(e, "test_target", None)
        tp = getattr(e, "test_preds", None)
        if tt is None or tp is None:
            continue
        tt = np.asarray(tt, dtype=float).ravel()
        tp = np.asarray(tp, dtype=float).ravel()
        tr = np.ptp(tt) if tt.size else 0.0
        if tr <= 0:
            continue
        best = max(best, float(np.ptp(tp) / tr))
    return best
