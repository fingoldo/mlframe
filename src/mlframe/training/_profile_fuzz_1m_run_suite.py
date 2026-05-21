"""``_run_suite_profiled`` carved out of
``mlframe.training._profile_fuzz_1m``.

Re-imported at the parent's module bottom so historical
``from mlframe.training._profile_fuzz_1m import _run_suite_profiled``
resolves transparently.
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




def _run_suite_profiled(
    target_type: str,
    n_rows: int,
    models: tuple[str, ...],
    seed: int,
    top_n: int,
    save_charts: bool = False,
    profile_predict: bool = True,
    profile_save: bool = True,
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

    # n_rows is the user-supplied UPPER bound. Heavy-axis combinations downstream may shrink
    # the frame to keep RAM bounded on commodity hardware (boruta_shap surrogate + SHAP
    # TreeExplainer materialises ~5 GB on 1M rows; LOF k-d tree fits + queries the entire
    # training set; polynomial expansion explodes feature count). The shrinkage decision lives
    # below at the axis-sampling block (search for ``_effective_n_rows``); the FULL frame is
    # built once anyway because the synthetic builder is O(n_rows) cheap and we slice via
    # ``df.head(_effective_n_rows)`` after the axes are known.
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
    # Post 390-finding-audit surface: previously these toggles were unexercised by the harness.
    # ``boruta_shap`` probability lowered 0.10 -> 0.04 after fuzz iter#196 (seed 2026051710)
    # peaked ~5 GB RSS on a 16 GB dev box: BorutaShap doubles the feature matrix with shadow
    # permutations, fits a CatBoost surrogate, and materialises a SHAP TreeExplainer
    # intermediate ~4-6x the doubled-feature size. Combined with the n_rows cap below this
    # keeps single-iteration peak under ~2 GB.
    _use_boruta_shap = bool(_axis_rng.random() < 0.04)
    # polynomial_degree=None (no expansion) / 2 (squared+cross) / 3 (cubed). degree=3 with
    # only 6 numeric cols stays well under polynomial_max_features=10000; the auto-tune chain
    # (interaction_only -> degree-- -> skip) is reachable when the synthetic frame grows wide.
    _polynomial_degree = int(_axis_rng.choice([0, 2, 3], p=[0.85, 0.10, 0.05]))
    # 0 = single-pass predict; >0 chunks the predict path through predict_batch_rows. At 1M
    # rows on a multiclass HGB the unchunked path can peak >2 GB RSS; chunked stays bounded.
    _predict_batch_rows = int(_axis_rng.choice([0, 50_000, 250_000], p=[0.70, 0.15, 0.15]))
    # Multi-target / mixed-target-type / multi-weight-schema axes: the previous harness only
    # ever passed a single ``y`` column with a single weight schema (uniform implicit). The
    # 390-finding audit's per-weight-schema FS-cache + per-target loop hoists never fired.
    # ``_n_targets``: 1 (legacy) or 2 (sibling target ``y2`` of the SAME base target_type).
    # ``_target_mix``: when 2 targets and the base type is classification or regression, sometimes
    #   inject a sibling of the OTHER type so target_by_type carries two distinct TargetTypes
    #   (e.g. binary y + regression y2). Forces _train_one_target to iterate >1 (target_type, name).
    # ``_weight_schema``: "uniform_only" (legacy) / "recency_only" / "both". When != uniform_only
    #   we synthesise a monotonic ts column the FTE uses for recency weighting; the suite's
    #   weight-aware FS / per-weight-schema model fit loop fires.
    _n_targets = int(_axis_rng.choice([1, 2], p=[0.70, 0.30]))
    _mix_target_types = bool(_n_targets == 2 and _axis_rng.random() < 0.50)
    _weight_schema = str(_axis_rng.choice(
        ["uniform_only", "recency_only", "both"],
        p=[0.60, 0.20, 0.20],
    ))
    # 2026-05-18 -- iter-23.5 fuzz axes (Packs J + K composite discovery)
    # and iter-32.5 fuzz axes (MRMR FE-search knobs) extended to the
    # 1M-row harness too. Without these the production-scale profile
    # ran the COMPOSITE_DISCOVERY=False / fe_npermutations=0 /
    # smart_polynom=0 path and never exercised the new code in the
    # post-Pack-H/J/K wave at production shapes. Defaults are
    # OFF-weighted because each axis adds cost (composite discovery
    # ~1-5s, MRMR FE-pollination ~2-10s for the n_total=1M case).
    _composite_discovery_enabled = (
        target_type == "regression"
        and bool(_axis_rng.random() < 0.30)
    )
    _composite_transforms_mode = (
        str(_axis_rng.choice(["all", "unary_only", "chain_only", "legacy"]))
        if _composite_discovery_enabled else "off"
    )
    # MRMR FE knobs only when MRMR is on (otherwise no-op). Probabilities
    # weighted to give meaningful coverage without exploding wall time.
    _mrmr_fe_ntop = int(_axis_rng.choice([0, 5], p=[0.60, 0.40])) if _use_mrmr_fs else 0
    _mrmr_fe_npermutations = (
        int(_axis_rng.choice([0, 10], p=[0.50, 0.50]))
        if (_use_mrmr_fs and _mrmr_fe_ntop > 0) else 0
    )
    _mrmr_fe_unary_preset = str(
        _axis_rng.choice(["minimal", "medium"], p=[0.50, 0.50])
    ) if _use_mrmr_fs else "minimal"
    _mrmr_fe_binary_preset = str(
        _axis_rng.choice(["minimal", "medium"], p=[0.50, 0.50])
    ) if _use_mrmr_fs else "minimal"
    _mrmr_fe_smart_polynom_iters = (
        int(_axis_rng.choice([0, 1], p=[0.70, 0.30]))
        if _use_mrmr_fs else 0
    )
    _mrmr_cat_fe_include_numeric = (
        bool(_axis_rng.random() < 0.30)
        if _use_mrmr_fs else False
    )
    print(
        f"  axes: mrmr_fs={_use_mrmr_fs} outlier={_outlier_method} "
        f"ensembles={_use_ensembles} cat_enc={_categorical_encoding} "
        f"scaler={_scaler_name} dim_reducer={_dim_reducer} "
        f"boruta_shap={_use_boruta_shap} poly_deg={_polynomial_degree} "
        f"predict_batch_rows={_predict_batch_rows} "
        f"n_targets={_n_targets} mix_target_types={_mix_target_types} weight={_weight_schema}"
    )
    print(
        f"  axes2: composite_disc={_composite_discovery_enabled} "
        f"transforms_mode={_composite_transforms_mode} "
        f"mrmr_fe_ntop={_mrmr_fe_ntop} mrmr_fe_nperm={_mrmr_fe_npermutations} "
        f"unary={_mrmr_fe_unary_preset} binary={_mrmr_fe_binary_preset} "
        f"smart_polynom={_mrmr_fe_smart_polynom_iters} "
        f"cat_fe_include_numeric={_mrmr_cat_fe_include_numeric}"
    )

    # n_rows shrinkage on heavy-axis combinations. Each cap is calibrated to keep the
    # affected code path's peak working set under ~1.5 GB on commodity 16 GB hardware
    # while still exercising the full pipeline. The strictest cap wins when several
    # heavy axes co-fire. Default (no heavy axis) keeps the user-supplied n_rows.
    _effective_n_rows = n_rows
    _row_cap_reasons: list[str] = []
    if _use_boruta_shap:
        # BorutaShap: feature-matrix doubling + CatBoost surrogate + SHAP TreeExplainer
        # all stack into a working set ~4-6x the feature matrix at peak. Iter#196 confirmed
        # ~5 GB RSS at 1M rows with default budget; 200k brings the peak into safe range.
        _effective_n_rows = min(_effective_n_rows, 200_000)
        _row_cap_reasons.append("boruta_shap")
    if _outlier_method == "lof":
        # LOF novelty=True fits a k-d tree on the full training set, then queries
        # n_neighbors per row at predict. At 1M rows the predict-side queries dominate
        # (~O(N log N x M)); 250k keeps the kd-tree build + queries under ~1 GB.
        _effective_n_rows = min(_effective_n_rows, 250_000)
        _row_cap_reasons.append("outlier=lof")
    if _outlier_method == "isolation_forest":
        # IsolationForest with sklearn default ``max_samples='auto'`` caps each tree's
        # working set at min(256, n_samples); fit RAM is bounded. The predict-side cost
        # is O(n_estimators x N) tree walks which is ~negligible at our n_estimators=20.
        # No cap needed but keep the flag visible so future iterations know IF was used.
        _row_cap_reasons.append("outlier=isolation_forest (no cap)")
    if _polynomial_degree >= 3:
        # Degree-3 polynomial on the 6-numeric synthetic features expands to ~84 cols
        # (interaction_only=True) or ~210 cols (full). Materialising the dense intermediate
        # ndarray at 1M rows is ~3-7 GB. Cap to 300k so the expansion stays under ~1 GB.
        _effective_n_rows = min(_effective_n_rows, 300_000)
        _row_cap_reasons.append("polynomial_degree=3")
    if _use_boruta_shap and _use_mrmr_fs:
        # Both heavy FS paths in one iteration is rare (0.04 x 0.25 = 1% probability) but
        # combines the BorutaShap surrogate + MRMR's MI matrix; cap to 150k to stay safe.
        _effective_n_rows = min(_effective_n_rows, 150_000)
        _row_cap_reasons.append("boruta_shap+mrmr_fs")
    if _effective_n_rows < n_rows:
        print(f"  row cap: n_rows={n_rows:_} -> {_effective_n_rows:_} (reasons: {', '.join(_row_cap_reasons)})")

    # Build sibling-target / weight-schema injections that the synthetic frame builder
    # downstream will pick up (frame is rebuilt below with these knobs threaded through).
    _frame_extra_targets: list[tuple[str, str]] = []  # [(col_name, kind=reg|bin)]
    if _n_targets == 2:
        # Sibling of the SAME or DIFFERENT type, depending on _mix_target_types.
        if _mix_target_types:
            # Pair a regression with a binary or vice versa, so target_by_type has 2 keys.
            if target_type == "regression":
                _frame_extra_targets.append(("y2", "bin"))
            elif target_type == "binary_classification":
                _frame_extra_targets.append(("y2", "reg"))
            elif target_type == "multiclass_classification":
                _frame_extra_targets.append(("y2", "reg"))
            else:
                _frame_extra_targets.append(("y2", "reg"))
        else:
            # Same-type sibling (most common multi-target setup).
            _frame_extra_targets.append(("y2", "reg" if target_type == "regression" else "bin"))
    _frame_add_ts = bool(_weight_schema != "uniform_only")
    # Re-build the frame WITH the extra-target / ts knobs at the EFFECTIVE row count
    # (post heavy-axis row cap above). The first build above used full ``n_rows`` and
    # gets replaced when extras are needed; otherwise we slice the pre-built frame.
    if _frame_extra_targets or _frame_add_ts:
        df = _make_synthetic_frame(
            target_type, _effective_n_rows, seed=seed,
            extra_targets=_frame_extra_targets, add_ts=_frame_add_ts,
        )
        print(f"  rebuilt frame with extras: targets={[t[0] for t in _frame_extra_targets]} add_ts={_frame_add_ts}")
    elif _effective_n_rows < len(df):
        # No extras to add, just trim. ``head`` works for both pandas + polars and is O(1)
        # (zero-copy view; the un-referenced lower portion is GC'd on the next collection).
        df = df.head(_effective_n_rows)
        print(f"  trimmed frame to {_effective_n_rows:_} rows for heavy-axis cap")

    # FTE kwargs: list the requested targets per type. The legacy single-target path is
    # preserved when _n_targets=1 and _weight_schema=uniform_only (no ts).
    _reg_targets: list[str] = []
    _cls_targets: list[str] = []
    _cls_exact: dict[str, Any] = {}
    if target_type == "regression":
        _reg_targets.append("y")
    elif target_type in ("binary_classification", "multiclass_classification"):
        _cls_targets.append("y")
        if target_type == "binary_classification":
            _cls_exact["y"] = 1
    # Append sibling targets to the right lists.
    for _name, _kind in _frame_extra_targets:
        if _kind == "reg":
            _reg_targets.append(_name)
        else:
            _cls_targets.append(_name)
            _cls_exact[_name] = 1

    if target_type == "regression":
        target_col = "y"
        fte_kwargs = dict(regression_targets=_reg_targets or ["y"])
        if _cls_targets:
            fte_kwargs["classification_targets"] = _cls_targets
            fte_kwargs["classification_exact_values"] = _cls_exact
        _tt = TargetTypes.REGRESSION
    elif target_type == "binary_classification":
        target_col = "y"
        # ``classification_exact_values`` alone is silently ignored by
        # SimpleFeaturesAndTargetsExtractor.build_targets — that branch is
        # gated on ``classification_targets`` being truthy. Pass both so the
        # binary suite actually trains end-to-end instead of suite-returning
        # empty target_by_type and looking like a 0.1s "OK" run.
        fte_kwargs = dict(
            classification_targets=_cls_targets or ["y"],
            classification_exact_values=_cls_exact or {"y": 1},
        )
        if _reg_targets:
            fte_kwargs["regression_targets"] = _reg_targets
        _tt = TargetTypes.BINARY_CLASSIFICATION
    elif target_type == "multiclass_classification":
        target_col = "y"
        fte_kwargs = dict(classification_targets=_cls_targets or ["y"])
        if _reg_targets:
            fte_kwargs["regression_targets"] = _reg_targets
        _tt = TargetTypes.MULTICLASS_CLASSIFICATION
    elif target_type == "multilabel_classification":
        # Multilabel needs a different FTE setup; out of scope for
        # SimpleFeaturesAndTargetsExtractor's defaults — skip from
        # this profile harness for now (TODO: extend when needed).
        return 0.0, False, "MULTILABEL_FTE_SETUP_OOS", "", 0.0, ""
    else:
        return 0.0, False, "UNSUPPORTED_TARGET_TYPE", "", 0.0, ""

    # Weight-schema knobs: SimpleFeaturesAndTargetsExtractor.get_sample_weights returns the
    # subset of {"uniform", "recency"} controlled by use_uniform_weighting / use_recency_weighting.
    # Recency requires ``ts_field`` so the FTE can read timestamps off the frame. When the
    # ``_frame_add_ts`` knob fired, ``_make_synthetic_frame`` injected a monotonic ``ts`` column.
    if _weight_schema == "recency_only":
        fte_kwargs["use_uniform_weighting"] = False
        fte_kwargs["use_recency_weighting"] = True
        fte_kwargs["ts_field"] = "ts"
    elif _weight_schema == "both":
        fte_kwargs["use_uniform_weighting"] = True
        fte_kwargs["use_recency_weighting"] = True
        fte_kwargs["ts_field"] = "ts"
    # else uniform_only: leave defaults (use_uniform_weighting=True, use_recency_weighting=True
    # but no ts_field so recency degrades to no-op).
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
        # 2026-05-18 -- iter-32.5 fuzz axes propagated. cat_fe_config
        # respects the include_numeric flag; the FE knobs respect the
        # per-iter ntop / npermutations / unary / binary / smart_polynom
        # axes drawn above. When ntop=0 the FE-pollination step is
        # short-circuited inside MRMR (same as the unaxed legacy default).
        _mrmr_kwargs_extra: dict = {}
        if _use_mrmr_fs:
            if _mrmr_cat_fe_include_numeric:
                from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
                _mrmr_kwargs_extra["cat_fe_config"] = CatFEConfig(
                    enable=True, include_numeric=True,
                )
            _mrmr_kwargs_extra.update({
                "fe_ntop_features": _mrmr_fe_ntop,
                "fe_npermutations": _mrmr_fe_npermutations,
                "fe_unary_preset": _mrmr_fe_unary_preset,
                "fe_binary_preset": _mrmr_fe_binary_preset,
                "fe_smart_polynom_iters": _mrmr_fe_smart_polynom_iters,
                "fe_smart_polynom_optimization_steps": (
                    10 if _mrmr_fe_smart_polynom_iters > 0 else 1000
                ),
                "fe_min_polynom_degree": 3,
                "fe_max_polynom_degree": 5 if _mrmr_fe_smart_polynom_iters > 0 else 3,
            })
        _fs_cfg = (
            FeatureSelectionConfig(
                use_mrmr_fs=_use_mrmr_fs,
                use_boruta_shap=_use_boruta_shap,
                mrmr_kwargs=({
                    "verbose": 0,
                    "max_runtime_mins": 1,
                    "n_workers": 1,
                    "quantization_nbins": 5,
                    "use_simple_mode": True,
                    "min_nonzero_confidence": 0.9,
                    "max_consec_unconfirmed": 3,
                    "full_npermutations": 3,
                    **_mrmr_kwargs_extra,
                } if _use_mrmr_fs else None),
                # BorutaShap budget tightened after iter#196 OOM (~5 GB on 1M rows): the
                # 200k row cap above bounds the matrix size; here we cap n_trials so
                # SHAP TreeExplainer stays small. iter-75 fix: ``max_iter`` and
                # ``surrogate_model_params`` were rejected by the
                # ``FeatureSelectionConfig.boruta_shap_kwargs`` validator because they
                # aren't in ``BorutaShap.__init__``'s signature -- removed. The
                # surrogate-model iteration count is controlled via the suite-level
                # ``model_kwargs`` (CatBoost iterations) when needed.
                # iter-99 fix: BorutaShap default ``classification=True`` paired with
                # a regression target raises ``ValueError: Unknown label type:
                # continuous``. Thread the target_type-driven flag so the surrogate
                # picks the right RandomForest{Classifier,Regressor}.
                boruta_shap_kwargs=({
                    "n_trials": 5,
                    "verbose": 0,
                    "classification": target_type != "regression",
                } if _use_boruta_shap else None),
            )
        )
        _od_detector = None
        if _outlier_method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            # n_estimators=10 (default 100, was 20 in earlier harness), max_samples=256
            # (sklearn 'auto' default already does min(256, n_samples); pinning it explicit
            # makes the cap visible and unaffected by future sklearn-default changes).
            # n_jobs=1 prevents joblib spawning multiple worker processes on a single-suite
            # fuzz iteration where the model itself is small.
            _od_detector = IsolationForest(
                contamination=0.05, random_state=int(seed) & 0xFFFFFFFF,
                n_estimators=10, max_samples=256, n_jobs=1,
            )
        elif _outlier_method == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            # LOF k-d tree at 250k rows (post heavy-axis cap above) + n_neighbors=20 fits
            # in ~150 MB; algorithm='kd_tree' is the explicit fast path (auto picks ball_tree
            # past a high-dim threshold). n_jobs=1 mirrors the IF rationale.
            _od_detector = LocalOutlierFactor(
                novelty=True, n_neighbors=20, algorithm="kd_tree", n_jobs=1,
            )
        _od_cfg = OutlierDetectionConfig(detector=_od_detector)
        _pp_cfg = PreprocessingBackendConfig(
            categorical_encoding=_categorical_encoding,
            scaler_name=(None if _scaler_name == "none" else _scaler_name),
        )
        # PreprocessingExtensionsConfig.scaler uses verbose sklearn names
        # ("StandardScaler", "RobustScaler", ...), not the polars-ds
        # short names that PreprocessingBackendConfig.scaler_name accepts.
        # Skip wiring scaler twice; vary dim_reducer + polynomial_degree instead.
        _ext_kwargs: dict = {}
        if _dim_reducer != "none":
            _ext_kwargs["dim_reducer"] = _dim_reducer
            _ext_kwargs["dim_n_components"] = 10
        if _polynomial_degree >= 2:
            _ext_kwargs["polynomial_degree"] = _polynomial_degree
            # polynomial_interaction_only=True keeps cross-products only (no x_i^2 etc.);
            # combined with the 390-finding-audit auto-tune chain (interaction_only ->
            # degree-- -> skip) gated by polynomial_max_features=10000 default, this is
            # the surface the new hardening should defend.
            _ext_kwargs["polynomial_interaction_only"] = True
        _ext_cfg = PreprocessingExtensionsConfig(**_ext_kwargs) if _ext_kwargs else None
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
            # 2026-05-18 -- iter-23.5 composite-discovery axis wired.
            # When the per-iter random draw picked discovery_enabled=True,
            # CompositeTargetDiscoveryConfig fires with the transform
            # palette from the iter-32.5 mode selector. Otherwise the
            # disabled config matches the legacy fast path. The
            # auxiliary diagnostics (baseline_diagnostics +
            # dummy_baselines) stay off in the 1M harness because they
            # add ~5-30 s of bootstrap compute per profile run that
            # masks the actual suite path.
            composite_target_discovery_config=(
                _build_composite_discovery_config_for_1m_inline(
                    enabled=_composite_discovery_enabled,
                    transforms_mode=_composite_transforms_mode,
                )
            ),
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
    _predict_results = None
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
            _predict_kwargs: dict = dict(
                df=df,
                models=trained_models,
                metadata=trained_metadata,
                features_and_targets_extractor=predict_fte,
                return_probabilities=(target_type != "regression"),
                verbose=0,
            )
            # predict_batch_rows is a 390-finding-audit addition that chunks the predict
            # path through bounded RSS. 0 = single-pass (legacy); >0 routes via the
            # chunked entry. The harness samples per-iteration to exercise both code paths.
            if _predict_batch_rows > 0:
                _predict_kwargs["predict_batch_rows"] = _predict_batch_rows
            _predict_results = predict_from_models(**_predict_kwargs)
            # Shape / dimensionality validation -- surfaces silent failures where
            # predict_from_models returns an empty/None/wrong-shape result while the
            # broad except-Exception block above would otherwise let the iteration
            # appear "OK". A model can fail at the per-model loop (logged + skipped
            # via _phase_helpers's continue) leaving results["predictions"] empty
            # without raising; we want THAT to surface as PREDICT:EMPTY here.
            _n_input = len(df)
            _preds_map = (_predict_results or {}).get("predictions") or {}
            _probs_map = (_predict_results or {}).get("probabilities") or {}
            _per_target_probs = (_predict_results or {}).get("per_target_probabilities") or {}
            if not _preds_map and not _probs_map:
                raise RuntimeError(
                    f"predict_from_models returned empty predictions+probabilities "
                    f"(models trained: {sum(len(v) for tt in trained_models.values() for v in tt.values())}; "
                    f"per_target_probs keys: {list(_per_target_probs)})"
                )
            for _mn, _p in _preds_map.items():
                _arr = np.asarray(_p) if _p is not None else None
                if _arr is None or _arr.shape[0] != _n_input:
                    raise RuntimeError(
                        f"predict_from_models[{_mn}] prediction len mismatch: "
                        f"got shape={None if _arr is None else _arr.shape}, expected first-dim={_n_input}"
                    )
            for _mn, _p in _probs_map.items():
                if _mn in ("ensemble",):
                    # The ensemble key is a 2-D aggregate; checked separately if present.
                    continue
                _arr = np.asarray(_p) if _p is not None else None
                if _arr is None or _arr.shape[0] != _n_input:
                    raise RuntimeError(
                        f"predict_from_models[{_mn}] probability shape mismatch: "
                        f"got shape={None if _arr is None else _arr.shape}, expected first-dim={_n_input}"
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

    # Save-to-disk pass -- exercise the model-serialization path (dill +
    # zstd compression at level=4 with threads=-1) on every trained model
    # the suite produced. iter-60: the SAVE tempdir is now kept alive
    # across two additional phases (LOAD + PREDICT-LOADED) so the harness
    # can detect bugs that only surface after a full save/load round trip
    # (e.g. fitted-state lost in serialization, base-column lists not
    # persisted, pre_pipeline.feature_names_in_ mismatch on reload, etc.).
    # Layout matches what ``load_mlframe_suite`` expects:
    #     <tmpdir>/metadata.pkl.zst
    #     <tmpdir>/<tt_slug>/<name_slug>/model.dump
    save_wall = 0.0
    save_profile_text = ""
    save_n_models = 0
    save_total_bytes = 0
    load_wall = 0.0
    load_profile_text = ""
    load_n_models = 0
    predict_loaded_wall = 0.0
    predict_loaded_profile_text = ""
    parity_status = "skip"  # "skip" / "ok" / "diff:<summary>"
    _save_tmpdir_obj = None
    _save_root: Optional[Path] = None
    if (
        profile_save
        and status.startswith("OK")
        and trained_models is not None
        and any(trained_models.values())
    ):
        import tempfile
        import pickle as _pickle
        import zstandard as _zstd
        from pyutilz.strings import slugify as _slugify
        from mlframe.training.io import save_mlframe_model

        save_profiler = cProfile.Profile()
        s0 = time.perf_counter()
        save_profiler.enable()
        try:
            _save_tmpdir_obj = tempfile.mkdtemp(prefix="mlframe_prof_save_")
            _save_root = Path(_save_tmpdir_obj)
            # Mirror production's _persist_ct_ensemble_entries slug-map
            # registration. Production only runs that finalize step when
            # output_config (data_dir/models_dir) is set; the harness
            # trains WITHOUT output_config, so trained_metadata's
            # slug_to_original_target_name lacks entries for the auto-
            # generated ``_CT_ENSEMBLE__*`` target_names. Without these,
            # load_mlframe_suite reads slug "CT_ENSEMBLE__y" (slugify
            # stripped the leading underscore) and falls back to the
            # slug itself -- the in-memory dict key becomes
            # "CT_ENSEMBLE__y" instead of "_CT_ENSEMBLE__y", and the
            # predict-from-loaded result keys diverge from the
            # predict-from-memory keys. iter-66 surfaced this as a
            # parity diff:
            #   preds_missing_after_load=['regression__CT_ENSEMBLE__y']
            #   preds_extra_after_load=['regression_CT_ENSEMBLE__y']
            _meta_for_save = dict(trained_metadata) if isinstance(trained_metadata, dict) else trained_metadata
            try:
                _sttn = dict(_meta_for_save.get("slug_to_original_target_name") or {})
                for _tt_key, _by_name in (trained_models or {}).items():
                    if not isinstance(_by_name, dict):
                        continue
                    for _tname in _by_name.keys():
                        if not isinstance(_tname, str):
                            continue
                        if _tname.startswith("_CT_ENSEMBLE__"):
                            _sttn[_tname] = _tname
                            _sttn[_slugify(_tname)] = _tname
                _meta_for_save["slug_to_original_target_name"] = _sttn
                # Also stamp slug_to_original_target_type defensively.
                _sttt = dict(_meta_for_save.get("slug_to_original_target_type") or {})
                for _tt_key in (trained_models or {}).keys():
                    if isinstance(_tt_key, str):
                        _sttt[_slugify(str(_tt_key).lower())] = _tt_key
                _meta_for_save["slug_to_original_target_type"] = _sttt
            except Exception:
                # Best-effort -- if metadata is non-dict for some exotic
                # reason, fall through and save as-is.
                pass
            # Suite-level metadata. load_mlframe_suite reads metadata.pkl.zst
            # first; zstd-compressed pickle for compatibility with the loader.
            try:
                # Wrap polars-ds Pipeline so pickle goes via to_json/from_json
                # rather than per-PyExpr Rust deserialization (avoided the
                # 100-200ms per expr observed at seed=20260522 / 100k binary;
                # see _setup_helpers._PolarsDsPipelineJsonProxy docstring).
                # Validate roundtrip BEFORE wrapping -- some step types
                # raise ComputeError on from_json (encoder variants in
                # particular). Fall back to plain pickle if so.
                try:
                    from mlframe.training.core._setup_helpers import _PolarsDsPipelineJsonProxy
                    from polars_ds.pipeline import Pipeline as _PdsPipeline
                    _pl_pipeline = _meta_for_save.get("pipeline")
                    if _pl_pipeline is not None and isinstance(_pl_pipeline, _PdsPipeline):
                        try:
                            _PdsPipeline.from_json(_pl_pipeline.to_json())
                            _meta_for_save = dict(_meta_for_save)
                            _meta_for_save["pipeline"] = _PolarsDsPipelineJsonProxy(_pl_pipeline)
                        except Exception:
                            pass  # roundtrip failed; ship as plain pickle
                except ImportError:
                    pass  # polars-ds unavailable; nothing to wrap
                _meta_path = _save_root / "metadata.pkl.zst"
                _cctx = _zstd.ZstdCompressor(level=4, write_checksum=True, write_content_size=True, threads=-1)
                with open(_meta_path, "wb") as _mf:
                    _mf.write(_cctx.compress(_pickle.dumps(_meta_for_save)))
                if _meta_path.exists():
                    save_total_bytes += _meta_path.stat().st_size
            except Exception as _meta_err:
                status = f"{status} | SAVE_META:{type(_meta_err).__name__}: {_meta_err}"[:200]
            # Per-model dump files under <tt_slug>/<name_slug>/<idx>.dump
            for _tt_key, _by_name in trained_models.items():
                if not isinstance(_by_name, dict):
                    continue
                _tt_slug = _slugify(str(_tt_key))
                for _model_name, _entries in _by_name.items():
                    if not isinstance(_entries, list):
                        continue
                    _name_slug = _slugify(str(_model_name))
                    _dir = _save_root / _tt_slug / _name_slug
                    _dir.mkdir(parents=True, exist_ok=True)
                    for _idx, _entry in enumerate(_entries):
                        _path = _dir / f"{_idx}.dump"
                        try:
                            # durable default is False (2026-05-20); harness keeps the
                            # explicit pass for clarity (bench writes to a tempdir that
                            # gets removed afterwards -- never needed durability).
                            #
                            # 2026-05-20: ONLY increment save_n_models when the call
                            # returns truthy AND the file actually landed on disk.
                            # save_mlframe_model returns False (logs error) on internal
                            # failures like zstd Allocation error / dill TypeError; the
                            # prior code blindly incremented save_n_models which then
                            # said "1 saved" but glob found 0 .dump files at load time,
                            # creating a confusing "save=1m/load=0m" report. With the
                            # check the report honestly shows 0 saved on memory-pressure
                            # combos and the load=0m result becomes consistent.
                            _save_ok = save_mlframe_model(_entry, str(_path), verbose=0, lean=True, durable=False)
                            if _save_ok and _path.exists():
                                save_n_models += 1
                                save_total_bytes += _path.stat().st_size
                            else:
                                status = f"{status} | SAVE_ONE_FAIL:returned={_save_ok!r} file_exists={_path.exists()}"[:200]
                        except Exception as _save_one_err:
                            status = f"{status} | SAVE_ONE:{type(_save_one_err).__name__}: {_save_one_err}"[:200]
        except Exception as e:
            status = f"{status} | SAVE:{type(e).__name__}: {e}"[:200]
        finally:
            save_profiler.disable()
        save_wall = time.perf_counter() - s0
        ss = io.StringIO()
        pss = pstats.Stats(save_profiler, stream=ss).sort_stats("cumulative")
        pss.print_stats(top_n)
        save_profile_text = ss.getvalue()

    # LOAD phase -- read the persisted suite back via the public
    # ``load_mlframe_suite`` API. Surfaces bugs in dill/zstd round trips,
    # pickle protocol drift, missing-dependency import-time errors during
    # de-serialization, and metadata schema changes between save and load.
    loaded_models = None
    loaded_metadata = None
    if (
        _save_root is not None
        and (_save_root / "metadata.pkl.zst").exists()
        and save_n_models > 0
    ):
        from mlframe.training.core.predict import load_mlframe_suite as _load_suite
        load_profiler = cProfile.Profile()
        l0 = time.perf_counter()
        load_profiler.enable()
        try:
            loaded_models, loaded_metadata = _load_suite(str(_save_root))
            load_n_models = sum(
                len(_entries)
                for _by_name in (loaded_models or {}).values()
                if isinstance(_by_name, dict)
                for _entries in _by_name.values()
                if isinstance(_entries, list)
            )
        except Exception as e:
            status = f"{status} | LOAD:{type(e).__name__}: {e}"[:200]
        finally:
            load_profiler.disable()
        load_wall = time.perf_counter() - l0
        ls = io.StringIO()
        pls = pstats.Stats(load_profiler, stream=ls).sort_stats("cumulative")
        pls.print_stats(top_n)
        load_profile_text = ls.getvalue()

    # PREDICT-LOADED phase -- run predict_from_models again against the
    # disk-roundtripped suite on the SAME input frame. Surfaces bugs that
    # only fire on the fresh-load path (fitted-state lost on dill round
    # trip, base_columns serialised wrong, cached attributes not pickled,
    # FTE-replay drift, etc.) and provides the substrate for the parity
    # check below.
    _predict_loaded_results = None
    if (
        loaded_models is not None
        and loaded_metadata is not None
        and any(loaded_models.values())
    ):
        from mlframe.training.core.predict import predict_from_models as _predict_from_models_loaded
        predict_loaded_profiler = cProfile.Profile()
        lp0 = time.perf_counter()
        predict_loaded_profiler.enable()
        try:
            predict_loaded_fte = SimpleFeaturesAndTargetsExtractor(**fte_kwargs)
            _pl_kwargs: dict = dict(
                df=df,
                models=loaded_models,
                metadata=loaded_metadata,
                features_and_targets_extractor=predict_loaded_fte,
                return_probabilities=(target_type != "regression"),
                verbose=0,
            )
            if _predict_batch_rows > 0:
                _pl_kwargs["predict_batch_rows"] = _predict_batch_rows
            _predict_loaded_results = _predict_from_models_loaded(**_pl_kwargs)
        except Exception as e:
            status = f"{status} | PREDICT_LOADED:{type(e).__name__}: {e}"[:200]
        finally:
            predict_loaded_profiler.disable()
        predict_loaded_wall = time.perf_counter() - lp0
        plps = io.StringIO()
        ppls = pstats.Stats(predict_loaded_profiler, stream=plps).sort_stats("cumulative")
        ppls.print_stats(top_n)
        predict_loaded_profile_text = plps.getvalue()

    # PARITY check -- compare every comparable output BEFORE save vs
    # AFTER load. Covers per-model dicts (``predictions`` /
    # ``probabilities``) AND ensemble-level aggregates
    # (``ensemble_predictions`` / ``ensemble_probabilities`` /
    # ``per_target_predictions`` / ``per_target_probabilities``). Every
    # shared key must yield bit-close output (rtol=1e-7); anything else
    # points to a serialization-induced silent drift that would corrupt
    # production downstream. Reports ``ok`` / ``diff:<summary>`` /
    # ``skip`` (insufficient data either side). Missing-key surface
    # (pre-only / post-only) is also reported as a diff -- a model key
    # disappearing on reload is a real bug class.
    if (
        _predict_results is not None
        and _predict_loaded_results is not None
    ):
        _diffs: list[str] = []

        def _cmp_array(_label: str, _a: Any, _b: Any) -> None:
            try:
                _a = np.asarray(_a)
                _b = np.asarray(_b)
                if _a.shape != _b.shape:
                    _diffs.append(f"{_label}:shape {_a.shape}!={_b.shape}")
                    return
                if not np.allclose(_a, _b, rtol=1e-7, atol=1e-7, equal_nan=True):
                    _max_abs = float(np.nanmax(np.abs(_a - _b))) if _a.size else 0.0
                    _diffs.append(f"{_label}:maxabs={_max_abs:.3g}")
            except Exception as _pcerr:
                _diffs.append(f"{_label}:cmp_err={_pcerr}")

        def _cmp_dict(_kind: str, _pre: dict, _post: dict) -> None:
            _shared = set(_pre) & set(_post)
            for _k in _shared:
                _cmp_array(f"{_kind}[{_k}]", _pre[_k], _post[_k])
            _missing_post = set(_pre) - set(_post)
            _missing_pre = set(_post) - set(_pre)
            if _missing_post:
                _diffs.append(f"{_kind}_missing_after_load={sorted(_missing_post)[:5]}")
            if _missing_pre:
                _diffs.append(f"{_kind}_extra_after_load={sorted(_missing_pre)[:5]}")

        _pre = _predict_results or {}
        _post = _predict_loaded_results or {}
        # Per-model dicts (includes the suite-wide "ensemble" key when
        # the suite trained >1 component; the inner key is kept as part
        # of the dict comparison so any ensemble drift surfaces here).
        _cmp_dict("preds", _pre.get("predictions") or {}, _post.get("predictions") or {})
        _cmp_dict("probs", _pre.get("probabilities") or {}, _post.get("probabilities") or {})
        # Top-level ensemble aggregates.
        if _pre.get("ensemble_predictions") is not None or _post.get("ensemble_predictions") is not None:
            if _pre.get("ensemble_predictions") is None or _post.get("ensemble_predictions") is None:
                _diffs.append(
                    "ensemble_predictions_one_sided="
                    f"pre={_pre.get('ensemble_predictions') is not None}/"
                    f"post={_post.get('ensemble_predictions') is not None}"
                )
            else:
                _cmp_array(
                    "ensemble_predictions",
                    _pre.get("ensemble_predictions"),
                    _post.get("ensemble_predictions"),
                )
        if _pre.get("ensemble_probabilities") is not None or _post.get("ensemble_probabilities") is not None:
            if _pre.get("ensemble_probabilities") is None or _post.get("ensemble_probabilities") is None:
                _diffs.append(
                    "ensemble_probabilities_one_sided="
                    f"pre={_pre.get('ensemble_probabilities') is not None}/"
                    f"post={_post.get('ensemble_probabilities') is not None}"
                )
            else:
                _cmp_array(
                    "ensemble_probabilities",
                    _pre.get("ensemble_probabilities"),
                    _post.get("ensemble_probabilities"),
                )
        # Per-target ensemble dicts (chosen-flavour aggregates).
        _cmp_dict(
            "per_target_preds",
            _pre.get("per_target_predictions") or {},
            _post.get("per_target_predictions") or {},
        )
        _cmp_dict(
            "per_target_probs",
            _pre.get("per_target_probabilities") or {},
            _post.get("per_target_probabilities") or {},
        )

        _any_shared = bool(
            (set(_pre.get("predictions") or {}) & set(_post.get("predictions") or {}))
            or (set(_pre.get("probabilities") or {}) & set(_post.get("probabilities") or {}))
            or (_pre.get("ensemble_predictions") is not None
                and _post.get("ensemble_predictions") is not None)
            or (_pre.get("ensemble_probabilities") is not None
                and _post.get("ensemble_probabilities") is not None)
            or (set(_pre.get("per_target_predictions") or {})
                & set(_post.get("per_target_predictions") or {}))
            or (set(_pre.get("per_target_probabilities") or {})
                & set(_post.get("per_target_probabilities") or {}))
        )
        if _diffs:
            parity_status = "diff:" + "; ".join(_diffs[:8])
        elif _any_shared:
            parity_status = "ok"

    # Best-effort cleanup of the save tempdir now that LOAD + PARITY are done.
    if _save_tmpdir_obj is not None:
        try:
            import shutil as _shutil
            _shutil.rmtree(_save_tmpdir_obj, ignore_errors=True)
        except Exception:
            pass

    return (
        train_wall, status.startswith("OK"), status, train_profile_text,
        predict_wall, predict_profile_text,
        save_wall, save_profile_text, save_n_models, save_total_bytes,
        load_wall, load_profile_text, load_n_models,
        predict_loaded_wall, predict_loaded_profile_text,
        parity_status,
    )
