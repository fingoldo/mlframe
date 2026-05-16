"""End-to-end ``train_mlframe_models_suite`` profile on 1M-row inputs.

Acknowledges the explicit gap noted in the dummy_baselines wave: smoke
tests + unit tests run on 5K rows; nothing exercised the suite end-to-
end at production shapes (1M+ rows). Per the mlframe CLAUDE.md profile
rule.

Builds a synthetic 1M-row frame per target_type (regression / binary /
multiclass) and runs ``train_mlframe_models_suite`` under cProfile.
Reports per-combo wall time + cProfile top-N cumulative.

Usage::

    python -m mlframe.training._profile_fuzz_1m
    python -m mlframe.training._profile_fuzz_1m --target regression
    python -m mlframe.training._profile_fuzz_1m --n-rows 500000 --top 50
    python -m mlframe.training._profile_fuzz_1m --models cb,xgb,lgb
    python -m mlframe.training._profile_fuzz_1m --save-charts  # surface kaleido cost

== Findings (2026-05-10, n_rows=1M, regression x lgb) ==

Without chart saving (--save-charts NOT set):

  Total wall: 21.6s
  - LGB engine.train + Booster.update:    14.3s  (library bound)
  - numba JIT compilation cold-start:      4.6s  (one-time per process;
                                                  amortized across targets)
  - report_regression_model_perf:          0.1s  (standalone direct-bench
                                                  on 1M rows; the e2e attribution
                                                  noise inflates this to ~2s
                                                  via cProfile deep-stack overhead)
  - All other suite overhead:              2.6s  (split / preprocess /
                                                  pipeline / save / metadata)

With chart saving (--save-charts, plot_outputs=plotly[html,png] default):

  Total wall: 98.4s  (76s extra over the no-chart baseline)
  Dominant cost: kaleido PNG export -- each plotly figure triggers a
  Chromium ``page.reload()`` that takes 12-15s while plotly.js re-mounts
  in the headless browser. On a 4-model x val+test x N-ensemble suite
  this can balloon to MINUTES of pure chart-export wall-time.

  matplotlib backend init (one-time): 20.5s  (Qt backend)

== Mitigation ==

  plot_outputs='matplotlib[png]'  -- 10-20x faster PNG (no Chromium)
  plot_outputs='plotly[html]'     -- HTML only, no PNG, no kaleido,
                                     instant; HTML is interactive in
                                     jupyter and shareable as a file
  plot_inline_display=False       -- skip inline render in jupyter (env
                                     var or ReportingConfig knob)

The suite warns at startup when ``save_charts=True AND
plot_outputs`` contains both ``plotly`` and ``png`` (see core.py
``[reporting] plot_outputs=...`` warning).

== cProfile attribution noise calibration ==

cProfile inflates pandas / sklearn / matplotlib / plotly deep-stack call
timings ~10-13x vs standalone wall-time microbench. When this harness
flags a function as a hotspot at 2-3s cumtime, cross-check by isolating
the function in a direct microbench (mostly the cumulative time turns
out to be 100-300ms standalone, not seconds).

The HONEST mlframe-side hotspots on 1M-row regression are: numba JIT
cold-start (one-time, amortized) + the chart-export path (only when
plot_outputs has plotly+png). Everything else is library-bounded.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pstats
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def _make_synthetic_frame(target_type: str, n_rows: int, seed: int = 42):
    """Build a synthetic frame whose shape varies by seed to exercise
    diverse mlframe code paths:

    - frame_type: 50/50 pandas vs polars (seeded). Polars input
      activates the polars-fastpath in CB/LGB/XGB and the
      get_pandas_view_of_polars_df bridge for sklearn consumers.
    - cat_columns: 1-2 string-categorical columns (low + mid card)
      added with 50% probability each. Exercises CatBoostEncoder /
      ordinal / one-hot paths.
    - text_column: free-form short-string column added with 30%
      probability. Exercises the TF-IDF path when downstream config
      enables it (off by default in this harness).
    - embedding_column: pl.List(Float32) / pd.Series-of-arrays added
      with 20% probability. Exercises the embedding-column auto-
      detect path.

    Always keeps 6 numeric features + a low/mid card int column so the
    suite always has enough usable features for the model fit. The
    seed-derived choices are deterministic, so re-runs with the same
    seed reproduce the same frame shape.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    use_polars = bool(rng.integers(0, 2))
    add_cat_low = bool(rng.integers(0, 2))
    add_cat_mid = bool(rng.integers(0, 2))
    add_text = rng.random() < 0.30
    add_embedding = rng.random() < 0.20
    # Data-quality variability: exercise the NaN-handling, drop-constant-
    # columns and correlation-aware paths. Probabilities deliberately
    # high enough to fire in most iterations.
    nan_fraction = float(rng.choice([0.0, 0.0, 0.05, 0.2]))  # 50% no-NaN, 25% sparse, 25% heavy
    add_constant_col = rng.random() < 0.30
    add_correlated_col = rng.random() < 0.30

    cols = {
        f"x{i}": rng.normal(size=n_rows).astype("float32")
        for i in range(6)
    }
    if add_constant_col:
        # A constant column - exercise remove_constant_columns. Using a
        # single value forces both min==max and (with eq_missing) the
        # all-null edge case to be reachable through the same code path.
        cols["x_const"] = np.full(n_rows, 7.0, dtype="float32")
    if add_correlated_col:
        # Heavily correlated with x0 (rho ~0.99). Stresses the MRMR /
        # RFECV redundancy detection and any covariance-based path.
        _noise = rng.normal(scale=0.05, size=n_rows).astype("float32")
        cols["x_corr_x0"] = (cols["x0"] + _noise).astype("float32")
    # Always keep two int "id-like" columns so the suite has enough
    # usable features regardless of the cat / text / embedding axes.
    cols["c_low"] = rng.integers(0, 5, n_rows).astype("int32")
    cols["c_mid"] = rng.integers(0, 50, n_rows).astype("int32")
    if add_cat_low:
        # String-categorical low-card column (5 levels) - the canonical
        # CatBoost native-cat + OneHot/Ordinal-encoder path.
        _labels = np.array(["A", "B", "C", "D", "E"], dtype=object)
        cols["cat_low"] = _labels[rng.integers(0, 5, n_rows)]
    if add_cat_mid:
        # Higher-card string-categorical column (50 levels).
        _labels = np.array([f"M{j:02d}" for j in range(50)], dtype=object)
        cols["cat_mid"] = _labels[rng.integers(0, 50, n_rows)]
    if add_text:
        # Short free-form text column; the suite ignores it for tree
        # models unless tfidf_columns lists it explicitly. Mostly
        # exercises auto-detect-feature-types text-promotion logic.
        _vocab = np.array(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa".split(),
            dtype=object,
        )
        # 3-5 words per row; np.choice is slow for 1M rows so build via index.
        _idx = rng.integers(0, len(_vocab), (n_rows, 4))
        cols["text_col"] = np.array([" ".join(_vocab[r]) for r in _idx], dtype=object)
    if add_embedding:
        # Per-row 8-dim embedding vector. Stored as object-of-ndarray on
        # pandas (the auto-detect path) or as pl.List(pl.Float32) on polars.
        cols["emb"] = [rng.normal(size=8).astype("float32") for _ in range(n_rows)]

    if target_type == "regression":
        y = (
            2.0 * cols["x0"]
            - 1.5 * cols["x1"]
            + 0.5 * cols["x2"] * cols["x3"]
            + rng.normal(0, 0.5, n_rows).astype("float32")
        )
        cols["y"] = y.astype("float32")
    elif target_type == "binary_classification":
        logit = 1.5 * cols["x0"] - 0.8 * cols["x1"] + 0.3 * cols["x2"]
        prob = 1.0 / (1.0 + np.exp(-logit))
        cols["y"] = (rng.uniform(0, 1, n_rows) < prob).astype("int32")
    elif target_type == "multiclass_classification":
        scores = np.column_stack([
            1.5 * cols["x0"] + rng.normal(0, 0.3, n_rows),
            -1.0 * cols["x1"] + rng.normal(0, 0.3, n_rows),
            0.5 * cols["x2"] + rng.normal(0, 0.3, n_rows),
            -0.5 * cols["x3"] + rng.normal(0, 0.3, n_rows),
        ])
        cols["y"] = scores.argmax(axis=1).astype("int32")
    elif target_type == "multilabel_classification":
        K = 4
        for k in range(K):
            logit = (
                rng.uniform(-1, 1) * cols["x0"]
                + rng.uniform(-1, 1) * cols["x1"]
                + rng.normal(0, 0.3, n_rows)
            )
            prob = 1.0 / (1.0 + np.exp(-logit))
            cols[f"y_{k}"] = (rng.uniform(0, 1, n_rows) < prob).astype("int32")
    else:
        raise ValueError(f"unsupported target_type {target_type!r}")
    if nan_fraction > 0:
        # Inject NaN AFTER target build so the target stays clean (any
        # column referenced in the target formula would propagate NaN
        # into y otherwise, crashing process_model with "train target
        # contains N NaN values"). x4, x5 are unreferenced in every
        # target_type formula above; safe to perturb.
        # Skip x_const so remove_constant_columns still flags it.
        for _c in ("x4", "x5"):
            if _c in cols:
                _mask = rng.random(n_rows) < nan_fraction
                cols[_c] = np.where(_mask, np.float32("nan"), cols[_c]).astype("float32")

    if use_polars:
        try:
            import polars as pl
            # Polars rejects object-dtype mixed columns; cast text/emb
            # explicitly to a polars-friendly representation.
            pl_cols: dict = {}
            for _k, _v in cols.items():
                if isinstance(_v, list) and _v and hasattr(_v[0], "shape"):
                    # Embedding column: list-of-ndarray -> pl.List(Float32)
                    pl_cols[_k] = pl.Series(
                        name=_k, values=_v, dtype=pl.List(pl.Float32)
                    )
                elif isinstance(_v, np.ndarray) and _v.dtype == object:
                    # String column (cat or text)
                    pl_cols[_k] = pl.Series(name=_k, values=_v.tolist(), dtype=pl.String)
                else:
                    pl_cols[_k] = pl.Series(name=_k, values=_v)
            print(f"  frame type: POLARS  cols: {list(pl_cols.keys())}")
            return pl.DataFrame(pl_cols)
        except Exception as _exc:
            print(f"  polars construction failed ({type(_exc).__name__}); falling back to pandas")
    print(f"  frame type: PANDAS  cols: {list(cols.keys())}")
    return pd.DataFrame(cols)


def _run_suite_profiled(
    target_type: str,
    n_rows: int,
    models: tuple[str, ...],
    seed: int,
    top_n: int,
    save_charts: bool = False,
    profile_predict: bool = True,
) -> tuple[float, bool, str, str, float, str]:
    """Returns ``(train_wall, ok, status, train_profile, predict_wall, predict_profile)``.

    ``predict_wall`` is 0.0 and ``predict_profile`` is "" when training did
    not return usable models (training crash, empty model dict, or
    ``profile_predict=False``).
    """
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.core.predict import predict_from_models
    from mlframe.training.configs import (
        TargetTypes, BaselineDiagnosticsConfig, DummyBaselinesConfig,
        OutputConfig, ReportingConfig, CompositeTargetDiscoveryConfig,
        FeatureSelectionConfig, OutlierDetectionConfig,
        PreprocessingBackendConfig, PreprocessingExtensionsConfig,
    )
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    df = _make_synthetic_frame(target_type, n_rows, seed=seed)
    print(f"  built frame: {len(df):_} rows x {len(df.columns)} cols")

    # Seed-derived axis variations drawn from ``_3WAY_AXES`` in
    # ``tests/training/_fuzz_combo.py``. Previously the harness only varied
    # frame_type/cat/text/emb/NaN/const/correlated; outlier detection,
    # MRMR, categorical_encoding, scaler, dim_reducer, ensembles paths
    # were never exercised at 1M scale. OCSVM intentionally omitted from
    # the outlier menu - O(n^2) fit dominates at n>=1200 (mirrors
    # test_fuzz_suite._outlier_detector_for_combo canonicalization).
    _axis_rng = np.random.default_rng(seed ^ 0xA11CE)
    _use_mrmr_fs = bool(_axis_rng.random() < 0.25)
    _outlier_method = str(_axis_rng.choice(
        ["none", "isolation_forest", "lof"],
        p=[0.60, 0.25, 0.15],
    ))
    _use_ensembles = bool(_axis_rng.random() < 0.40)
    _categorical_encoding = str(_axis_rng.choice(["ordinal", "onehot"], p=[0.80, 0.20]))
    _scaler_name = str(_axis_rng.choice(["standard", "robust", "none"], p=[0.50, 0.30, 0.20]))
    _dim_reducer = str(_axis_rng.choice(["none", "PCA", "TruncatedSVD"], p=[0.80, 0.13, 0.07]))
    print(
        f"  axes: mrmr_fs={_use_mrmr_fs} outlier={_outlier_method} "
        f"ensembles={_use_ensembles} cat_enc={_categorical_encoding} "
        f"scaler={_scaler_name} dim_reducer={_dim_reducer}"
    )

    if target_type == "regression":
        target_col = "y"
        fte_kwargs = dict(regression_targets=["y"])
        _tt = TargetTypes.REGRESSION
    elif target_type == "binary_classification":
        target_col = "y"
        # ``classification_exact_values`` alone is silently ignored by
        # SimpleFeaturesAndTargetsExtractor.build_targets — that branch is
        # gated on ``classification_targets`` being truthy. Pass both so the
        # binary suite actually trains end-to-end instead of suite-returning
        # empty target_by_type and looking like a 0.1s "OK" run.
        fte_kwargs = dict(
            classification_targets=["y"],
            classification_exact_values={"y": 1},
        )
        _tt = TargetTypes.BINARY_CLASSIFICATION
    elif target_type == "multiclass_classification":
        target_col = "y"
        fte_kwargs = dict(classification_targets=["y"])
        _tt = TargetTypes.MULTICLASS_CLASSIFICATION
    elif target_type == "multilabel_classification":
        # Multilabel needs a different FTE setup; out of scope for
        # SimpleFeaturesAndTargetsExtractor's defaults — skip from
        # this profile harness for now (TODO: extend when needed).
        return 0.0, False, "MULTILABEL_FTE_SETUP_OOS", "", 0.0, ""
    else:
        return 0.0, False, "UNSUPPORTED_TARGET_TYPE", "", 0.0, ""

    fte = SimpleFeaturesAndTargetsExtractor(**fte_kwargs)

    train_profiler = cProfile.Profile()
    t0 = time.perf_counter()
    train_profiler.enable()
    status = "OK"
    trained_models: dict | None = None
    trained_metadata: dict | None = None
    try:
        # MRMR at 1M is expensive; the kwargs below mirror the fast
        # settings used by test_fuzz_3way_suite (verbose=0, hard 1-min
        # cap, simple_mode, low quantization).
        _fs_cfg = (
            FeatureSelectionConfig(
                use_mrmr_fs=True,
                mrmr_kwargs={
                    "verbose": 0,
                    "max_runtime_mins": 1,
                    "n_workers": 1,
                    "quantization_nbins": 5,
                    "use_simple_mode": True,
                    "min_nonzero_confidence": 0.9,
                    "max_consec_unconfirmed": 3,
                    "full_npermutations": 3,
                },
            )
            if _use_mrmr_fs
            else FeatureSelectionConfig()
        )
        _od_detector = None
        if _outlier_method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            _od_detector = IsolationForest(
                contamination=0.05, random_state=int(seed) & 0xFFFFFFFF, n_estimators=20,
            )
        elif _outlier_method == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            _od_detector = LocalOutlierFactor(novelty=True, n_neighbors=20)
        _od_cfg = OutlierDetectionConfig(detector=_od_detector)
        _pp_cfg = PreprocessingBackendConfig(
            categorical_encoding=_categorical_encoding,
            scaler_name=(None if _scaler_name == "none" else _scaler_name),
        )
        # PreprocessingExtensionsConfig.scaler uses verbose sklearn names
        # ("StandardScaler", "RobustScaler", ...), not the polars-ds
        # short names that PreprocessingBackendConfig.scaler_name accepts.
        # Skip wiring scaler twice; vary dim_reducer instead.
        _ext_cfg = (
            PreprocessingExtensionsConfig(dim_reducer=_dim_reducer, dim_n_components=10)
            if _dim_reducer != "none"
            else None
        )
        trained_models, trained_metadata = train_mlframe_models_suite(
            df=df,
            target_name=target_col,
            model_name="prof",
            features_and_targets_extractor=fte,
            mlframe_models=list(models),
            use_mlframe_ensembles=_use_ensembles,
            feature_selection_config=_fs_cfg,
            outlier_detection_config=_od_cfg,
            pipeline_config=_pp_cfg,
            preprocessing_extensions=_ext_cfg,
            verbose=0,
            output_config=OutputConfig(
                data_dir=("data" if save_charts else ""),
                models_dir=("models" if save_charts else ""),
                save_charts=save_charts,
            ),
            # Disable expensive auxiliary diagnostics so we measure
            # the SUITE path proper, not the addons:
            composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
            baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
            dummy_baselines_config=DummyBaselinesConfig(enabled=False),
            reporting_config=ReportingConfig(
                # plotly[html,png] is the prod default that tripped the
                # user's 5M run with kaleido cycles dominating. Use that
                # when --save-charts is set; otherwise matplotlib (no
                # kaleido cost).
                plot_outputs=("plotly[html,png]" if save_charts else "matplotlib[png]"),
                plot_inline_display=False,
            ),
        )
    except Exception as e:
        status = f"{type(e).__name__}: {e}"[:120]
    finally:
        train_profiler.disable()
    train_wall = time.perf_counter() - t0

    s = io.StringIO()
    ps = pstats.Stats(train_profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(top_n)
    train_profile_text = s.getvalue()

    # Predict pass — exercise the full predict path (preprocess, per-model
    # predict, ensemble average) on the SAME input frame. Surfaces hot
    # spots in pipeline.transform / model.predict / ensemble averaging
    # that the training-only profile misses entirely.
    predict_wall = 0.0
    predict_profile_text = ""
    if (
        profile_predict
        and status == "OK"
        and trained_models is not None
        and trained_metadata is not None
        and any(trained_models.values())
    ):
        # Fresh FTE: the training FTE captures fit-time state; predict
        # path should reuse the public API the way a real downstream
        # would (load suite, run predict on raw input).
        predict_fte = SimpleFeaturesAndTargetsExtractor(**fte_kwargs)
        predict_profiler = cProfile.Profile()
        p0 = time.perf_counter()
        predict_profiler.enable()
        try:
            predict_from_models(
                df=df,
                models=trained_models,
                metadata=trained_metadata,
                features_and_targets_extractor=predict_fte,
                return_probabilities=(target_type != "regression"),
                verbose=0,
            )
        except Exception as e:
            # Don't clobber the training status; surface predict-only failure separately.
            status = f"{status} | PREDICT:{type(e).__name__}: {e}"[:200]
        finally:
            predict_profiler.disable()
        predict_wall = time.perf_counter() - p0
        sp = io.StringIO()
        psp = pstats.Stats(predict_profiler, stream=sp).sort_stats("cumulative")
        psp.print_stats(top_n)
        predict_profile_text = sp.getvalue()

    return train_wall, status.startswith("OK"), status, train_profile_text, predict_wall, predict_profile_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-rows", type=int, default=1_000_000)
    p.add_argument("--target", default="all",
                   choices=("all", "regression", "binary_classification",
                            "multiclass_classification"))
    p.add_argument("--models", default="cb",
                   help="Comma-separated model list (cb,xgb,lgb,linear). "
                        "Default 'cb' to bound per-combo wall time.")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-charts", action="store_true",
                   help="Enable chart saving (default off — measures core suite, "
                        "not chart export). Use to surface plotly+kaleido cost.")
    p.add_argument("--no-predict", action="store_true",
                   help="Skip the post-training predict-on-full-frame profile "
                        "pass. Default ON because predict-path hotspots "
                        "(pipeline.transform, per-model predict, ensemble averaging) "
                        "are invisible to the training-only profile.")
    args = p.parse_args()

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())

    targets: list[str] = (
        ["regression", "binary_classification", "multiclass_classification"]
        if args.target == "all" else [args.target]
    )

    print(f"# 1M-row e2e profile (n_rows={args.n_rows:_}, models={models}, "
          f"save_charts={args.save_charts}, profile_predict={not args.no_predict})")
    summary: list[tuple[str, float, float, str]] = []
    for tt in targets:
        label = f"{tt} x {','.join(models)}"
        print(f"\n=== {label} ===")
        train_wall, ok, status, train_prof, predict_wall, predict_prof = _run_suite_profiled(
            tt, args.n_rows, models, args.seed, args.top,
            save_charts=args.save_charts,
            profile_predict=not args.no_predict,
        )
        summary.append((label, train_wall, predict_wall, status))
        print(f"  train wall: {train_wall:.1f}s  status: {status}")
        print(train_prof[:6000])
        if predict_wall > 0 or predict_prof:
            print(f"\n--- PREDICT phase (same input frame, predict_from_models) ---")
            print(f"  predict wall: {predict_wall:.1f}s")
            print(predict_prof[:6000])

    print("\n# Wall-time summary:")
    for label, t_train, t_pred, status in summary:
        if t_pred > 0:
            print(f"  {label:<55} train={t_train:>7.1f}s  predict={t_pred:>6.1f}s  {status}")
        else:
            print(f"  {label:<55} train={t_train:>7.1f}s  predict=---     {status}")


if __name__ == "__main__":
    main()
