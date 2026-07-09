"""End-to-end ``train_mlframe_models_suite`` profile on 1M-row inputs.

Acknowledges the explicit gap noted in the dummy_baselines wave: smoke
tests + unit tests run on 5K rows; nothing exercised the suite end-to-
end at production shapes (1M+ rows). Per the mlframe CLAUDE.md profile
rule.

Builds a synthetic 1M-row frame per target_type (regression / binary /
multiclass) and runs ``train_mlframe_models_suite`` under cProfile.
Reports per-combo wall time + cProfile top-N cumulative.

Usage::

    python -m mlframe.training._benchmarks._profile_fuzz_1m
    python -m mlframe.training._benchmarks._profile_fuzz_1m --target regression
    python -m mlframe.training._benchmarks._profile_fuzz_1m --n-rows 500000 --top 50
    python -m mlframe.training._benchmarks._profile_fuzz_1m --models cb,xgb,lgb
    python -m mlframe.training._benchmarks._profile_fuzz_1m --save-charts  # surface kaleido cost

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
  plot_outputs='plotly[html]'  -- HTML only, no PNG, no kaleido,
                                     instant; HTML is interactive in
                                     jupyter and shareable as a file
  plot_inline_display=False  -- skip inline render in jupyter (env
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def _make_synthetic_frame(
    target_type: str, n_rows: int, seed: int = 42,
    *,
    extra_targets: list | None = None,
    add_ts: bool = False,
):
    """Build a synthetic frame whose shape varies by seed to exercise
    diverse mlframe code paths.

    Additional knobs (added post 390-finding-audit harness extension):
      - ``extra_targets``: list of ``(col_name, kind)`` pairs where ``kind in {"reg", "bin"}``.
        Generates sibling target columns alongside the primary ``y`` so the FTE can route
        multiple targets / mixed target types through the suite (per-target loop hoists,
        per-weight-schema FS cache).
      - ``add_ts``: when True, emit a strictly-monotonic-increasing ``ts`` column the FTE
        consumes via ``ts_field="ts"`` for recency-based weight schema generation.

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

    cols = {f"x{i}": rng.normal(size=n_rows).astype("float32") for i in range(6)}
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
        y = 2.0 * cols["x0"] - 1.5 * cols["x1"] + 0.5 * cols["x2"] * cols["x3"] + rng.normal(0, 0.5, n_rows).astype("float32")
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
            logit = rng.uniform(-1, 1) * cols["x0"] + rng.uniform(-1, 1) * cols["x1"] + rng.normal(0, 0.3, n_rows)
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

    # Sibling targets (multi-target / mixed-type fuzz extension). The base ``y`` was
    # already built per ``target_type`` above; here we add ``y2`` (and any future names)
    # using the same numeric features so the model has a chance to learn each one.
    # iter-141 fix: use only NaN-clean columns (x0/x1/x3) -- x4 and x5 may have NaN
    # injected at the lines above, which propagates into y2 and trips the
    # production NaN-target guard at process_model. Observed many times
    # (iter-77/81/108/136/137/140); see commit log.
    for _name, _kind in extra_targets or []:
        if _kind == "reg":
            _y2 = 1.2 * cols["x1"] - 0.7 * cols["x0"] + 0.4 * cols["x3"] + rng.normal(0, 0.5, n_rows).astype("float32")
            cols[_name] = _y2.astype("float32")
        elif _kind == "bin":
            _logit = -1.0 * cols["x1"] + 0.5 * cols["x3"] + 0.3 * cols["x0"]
            _prob = 1.0 / (1.0 + np.exp(-_logit))
            cols[_name] = (rng.uniform(0, 1, n_rows) < _prob).astype("int32")
        else:
            raise ValueError(f"unsupported extra_target kind {_kind!r}")

    if add_ts:
        # Strictly monotonic-increasing seconds-since-epoch column. The FTE's recency
        # weighting needs a comparable (numeric/datetime) sequence; a plain int64 second
        # count is the simplest form the polars + pandas paths both accept.
        _start = int(1_700_000_000)  # Nov 2023 epoch baseline
        cols["ts"] = _start + np.arange(n_rows, dtype=np.int64)

    if use_polars:
        try:
            import polars as pl
            # Polars rejects object-dtype mixed columns; cast text/emb
            # explicitly to a polars-friendly representation.
            pl_cols: dict = {}
            for _k, _v in cols.items():
                if isinstance(_v, list) and _v and hasattr(_v[0], "shape"):
                    # Embedding column: list-of-ndarray -> pl.List(Float32)
                    pl_cols[_k] = pl.Series(name=_k, values=_v, dtype=pl.List(pl.Float32))
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


def _build_composite_discovery_config_for_1m_inline(*, enabled: bool, transforms_mode: str):
    """Build a CompositeTargetDiscoveryConfig honoring the iter-23.5
    fuzz axes. Disabled-config returns the legacy fast path.

    Note (2026-05-18 user feedback): adding a new axis currently
    requires edits in (a) _fuzz_combo.AXES, (b) FuzzCombo dataclass,
    (c) canonical_key, (d) _build_combo, (e) test_fuzz_suite
    _configs_for_combo, (f) here. Follow-up: extract a shared
    "FuzzCombo -> suite_kwargs" builder so the 1M harness sources
    its axes from a FuzzCombo instance and consumes the same
    config-building helpers as the pytest suite. See loop iter log.
    """
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    if not enabled:
        return CompositeTargetDiscoveryConfig(enabled=False)
    if transforms_mode == "unary_only":
        _transforms = ["cbrt_y", "log_y", "yeo_johnson_y", "quantile_normal_y"]
    elif transforms_mode == "chain_only":
        _transforms = ["chain_linres_cbrt", "chain_linres_yj", "chain_monres_cbrt", "chain_monres_yj"]
    elif transforms_mode == "legacy":
        _transforms = ["diff", "ratio", "logratio", "linear_residual", "quantile_residual", "monotonic_residual"]
    else:  # "all"
        _transforms = None  # library default (all 14)
    _kw: dict = {
        "enabled": True,
        "base_candidates": "auto",
        "auto_base_top_k": 3,
        "multi_base_enabled": True,
        "multi_base_max_k": 2,
    }
    if _transforms is not None:
        _kw["transforms"] = _transforms
    return CompositeTargetDiscoveryConfig(**_kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-rows", type=int, default=1_000_000)
    p.add_argument("--target", default="all", choices=("all", "regression", "binary_classification", "multiclass_classification"))
    p.add_argument("--models", default="cb", help="Comma-separated model list (cb,xgb,lgb,linear). " "Default 'cb' to bound per-combo wall time.")
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
    p.add_argument("--no-save", action="store_true",
                   help="Skip the model-save-to-disk profile pass. Default ON: "
                        "production callers always save trained models, and the "
                        "dill + zstd compression path (especially for multi-MB "
                        "tree-ensemble pickles at 1M rows) is a real production "
                        "hot spot that the train profile alone does not surface.")
    args = p.parse_args()

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())

    targets: list[str] = ["regression", "binary_classification", "multiclass_classification"] if args.target == "all" else [args.target]

    print(
        f"# 1M-row e2e profile (n_rows={args.n_rows:_}, models={models}, "
        f"save_charts={args.save_charts}, "
        f"profile_predict={not args.no_predict}, "
        f"profile_save={not args.no_save})"
    )
    summary: list[tuple] = []
    for tt in targets:
        label = f"{tt} x {','.join(models)}"
        print(f"\n=== {label} ===")
        (
            train_wall, ok, status, train_prof,
            predict_wall, predict_prof,
            save_wall, save_prof, save_n_models, save_total_bytes,
            load_wall, load_prof, load_n_models,
            predict_loaded_wall, predict_loaded_prof,
            parity_status,
        ) = _run_suite_profiled(
            tt, args.n_rows, models, args.seed, args.top,
            save_charts=args.save_charts,
            profile_predict=not args.no_predict,
            profile_save=not args.no_save,
        )
        summary.append((
            label, train_wall, predict_wall, save_wall, save_n_models, save_total_bytes,
            load_wall, load_n_models, predict_loaded_wall, parity_status, status,
        ))
        print(f"  train wall: {train_wall:.1f}s  status: {status}")
        print(train_prof[:6000])
        if predict_wall > 0 or predict_prof:
            print(f"\n--- PREDICT phase (same input frame, predict_from_models) ---")
            print(f"  predict wall: {predict_wall:.1f}s")
            print(predict_prof[:6000])
        if save_wall > 0 or save_prof:
            _mb = save_total_bytes / (1024.0 * 1024.0)
            print(f"\n--- SAVE phase (dill + zstd, tempdir) ---")
            print(f"  save wall: {save_wall:.2f}s  models_saved={save_n_models}  total_bytes_on_disk={_mb:.2f} MB")
            print(save_prof[:6000])
        if load_wall > 0 or load_prof:
            print(f"\n--- LOAD phase (load_mlframe_suite from tempdir) ---")
            print(f"  load wall: {load_wall:.2f}s  models_loaded={load_n_models}")
            print(load_prof[:6000])
        if predict_loaded_wall > 0 or predict_loaded_prof:
            print(f"\n--- PREDICT-LOADED phase (predict_from_models on disk-roundtripped suite) ---")
            print(f"  predict_loaded wall: {predict_loaded_wall:.2f}s  parity={parity_status}")
            print(predict_loaded_prof[:6000])

    print("\n# Wall-time summary:")
    for (
        label, t_train, t_pred, t_save, n_save, b_save,
        t_load, n_load, t_pred_loaded, parity, status,
    ) in summary:
        _mb = b_save / (1024.0 * 1024.0)
        _pred_str = f"predict={t_pred:>6.1f}s" if t_pred > 0 else "predict=---   "
        _save_str = f"save={t_save:>5.2f}s ({n_save}m,{_mb:.1f}MB)" if t_save > 0 else "save=---           "
        _load_str = f"load={t_load:>5.2f}s ({n_load}m)" if t_load > 0 else "load=---       "
        _pl_str = f"predict_loaded={t_pred_loaded:>5.2f}s parity={parity}" if t_pred_loaded > 0 else f"predict_loaded=---   parity={parity}"
        print(f"  {label:<55} train={t_train:>7.1f}s  {_pred_str}  {_save_str}  " f"{_load_str}  {_pl_str}  {status}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# Sibling-module re-export. The 869-LOC ``_run_suite_profiled`` body
# lives in ``_profile_fuzz_1m_run_suite.py`` so this file stays below
# the 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._profile_fuzz_1m_run_suite import _run_suite_profiled  # noqa: E402,F401
