"""cProfile harness for the composite-target pipeline + the world-class
extensions (discovery time-aware screen, conformal / CQR intervals, base-margin
classification). Run as:

    python -m mlframe.training.composite._profile_pipeline [n]

Prints the top cumulative-time hotspots per stage so a maintainer can re-profile
after a change. Per CLAUDE.md: cProfile inflates pandas/sklearn deep-stack call
timings ~10-13x vs standalone wall-time -- treat sub-millisecond wall hotspots as
attribution noise, not real cost.

Findings (n=15k, 2026-06-11) -- no actionable speedup in the new code:
  * conformal calibrate + interval (incl. CQR): ~0.007s total; the radius
    quantile + band arithmetic are sub-millisecond. NOT a hotspot.
  * base-margin classification: ~1.75s, of which ~1.67s is the inner LightGBM
    fit (``_fit_inner_with_init_score``); the wrapper's margin add + softmax are
    negligible. NOT a wrapper hotspot.
  * regression fit+predict: ~0.006s; predict / _predict_unclipped sub-ms.
  * discovery.fit: ~3.45s, entirely the (already perf-mature) MI-screen + tiny
    rerank, not the M6 time-ordering add (a single argsort).
The cost lives in the underlying model fits, as designed; the composite layer is
thin. Re-run after any change to the predict / inverse / conformal hot path.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys


def _synth_regression(n, seed=0):
    """Build a small synthetic regression frame (base column ``b``, extra feature ``feat``, monotone timestamp ``ts``) whose target is a noisy linear combination -- deterministic given ``seed`` so profile runs are reproducible."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 1.0, n)
    feat = rng.normal(0.0, 1.0, n)
    ts = np.arange(n)
    y = b + 0.5 * feat + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"b": b, "feat": feat, "ts": ts}), y


def _profile(label, fn, top=12):
    """Run ``fn`` once under cProfile, print its top-``top`` cumulative-time frames labelled with ``label``."""
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative").print_stats(top)
    print(f"\n===== {label} =====")
    print("\n".join(s.getvalue().splitlines()[: top + 6]))


def main(argv):
    """CLI entrypoint: build the synthetic dataset (size from ``argv[1]``, default 20k) and profile the four composite-pipeline stages -- discovery, plain regression fit+predict, conformal calibration, and base-margin classification."""
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from .estimator import CompositeTargetEstimator
    from .classification import CompositeClassificationEstimator
    from .discovery import CompositeTargetDiscovery
    from ..configs import CompositeTargetDiscoveryConfig

    n = int(argv[1]) if len(argv) > 1 else 20_000
    X, y = _synth_regression(n)

    Xy = X.assign(y=y)  # discovery expects the target as a column of the frame

    def disco():
        """Profile target-discovery's time-aware MI screen + rerank on the synthetic frame."""
        cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=min(4000, n), base_candidates=["b"], time_column="ts")
        CompositeTargetDiscovery(cfg).fit(Xy, "y", ["b", "feat"], np.arange(n), time_ordering=X["ts"].to_numpy())

    def reg_fit_predict():
        """Profile a plain composite regression fit + predict (no conformal/interval machinery)."""
        est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="linear_residual", base_column="b").fit(X, y)
        est.predict(X)

    def conformal():
        """Profile a fit/calibrate/predict-interval cycle exercising the CQR conformal path on a held-out calibration half."""
        est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="linear_residual", base_column="b").fit(X.iloc[: n // 2], y[: n // 2])
        est.calibrate_conformal(X.iloc[n // 2 :], y[n // 2 :], 0.1)
        est.predict_interval(X, 0.1)

    def classify():
        """Profile base-margin classification (LightGBM init-score fit + softmax); no-ops if lightgbm isn't installed."""
        try:
            import lightgbm as lgb
        except Exception:
            return
        yc = (y > np.median(y)).astype(int)
        est = CompositeClassificationEstimator(base_estimator=lgb.LGBMClassifier(n_estimators=80, verbose=-1)).fit(X, yc)
        est.predict_proba(X)

    print(f"composite pipeline profile @ n={n}")
    _profile("discovery.fit (time-aware screen)", disco)
    _profile("regression fit+predict", reg_fit_predict)
    _profile("conformal calibrate+interval", conformal)
    _profile("base-margin classification", classify)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv)
