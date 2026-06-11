"""Reporting + naming + fairness + container configs for ``mlframe.training.configs``.

Split out from ``configs.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every class is re-exported from
``configs`` so existing ``from mlframe.training.configs import ReportingConfig``
(and the other moved names) imports continue to resolve.

What lives here:
  - ``ReportingConfig`` (calibration / training-performance report appearance)
  - ``ConfidenceAnalysisConfig``
  - ``NamingConfig``
  - ``PredictionsContainer``
  - ``FairnessConfig``
"""
from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, List, Literal, Optional, Tuple, Union

from pydantic import Field, field_validator, model_validator

from ._configs_base import BaseConfig
# ``ReportingConfig.feature_importance_config`` is typed as
# ``Optional[FeatureImportanceConfig]``; the latter lives in
# ``_training_runtime_configs.py``.
from ._training_runtime_configs import FeatureImportanceConfig
# ``training.diagnostics`` is leaf (does not import configs), so importing the
# opt-in learning-curve config here introduces no cycle.
from .diagnostics import LearningCurveConfig


# Title-metrics token grammar - mirrors metrics.TITLE_METRIC_TOKENS but kept
# duplicated here to avoid importing from metrics.py at config-class
# definition time (import-cost concern, plus configs is a foundational
# module). Keep these two sets in sync; the validator in ReportingConfig
# falls back gracefully if a new token is added in metrics.py without here.
_REPORTING_ALLOWED_TITLE_TOKENS: FrozenSet[str] = frozenset({
    "ICE", "BR", "BR_DECOMP", "ECE", "CMAEW",
    "COV", "LL", "ROC_AUC", "PR_AUC", "DENS",
    # 2026-05-28 audit batch additions (binary classification title tokens).
    # Gini deliberately NOT a token: =2*AUC-1, redundant with ROC_AUC for
    # chart-title use; available in metrics dict as "Gini" anyway.
    "KS", "MCC", "BSS",
})


class ReportingConfig(BaseConfig):
    """Look of the calibration / training performance report.

    Scope: report appearance + per-metric title composition + histogram
    subplot toggles. Filesystem paths live on ``OutputConfig``;
    feature-importance plot parameters live on ``FeatureImportanceConfig``
    (referenced via ``feature_importance_config``).

    Title-metric composition is an ordered string template
    ``title_metrics_template``. The grammar is validated at config
    construction time so an invalid template fails before training
    starts, not mid-figure.

    Token grammar (closed set, case-insensitive on input):
      - ``ICE``: integral calibration error
      - ``BR``: Brier loss (bare)
      - ``BR_DECOMP``: Brier with REL/RES/UNC decomposition parenthetical
        (mutually exclusive with ``BR``)
      - ``ECE``: standard expected calibration error
      - ``CMAEW``: mlframe-native power-weighted calibration MAE
      - ``COV``: bin coverage
      - ``LL``: log loss
      - ``ROC_AUC``: ROC AUC (with grouped variant in brackets when
        group_ids supplied)
      - ``PR_AUC``: PR AUC (followed by PR/RE/F1 trailing)
      - ``DENS``: bin density [max;min]

    Tokens render in the order given. Whitespace-separated. Duplicates
    rejected. Unknown tokens rejected. Empty string is legal (title gets
    only the user-supplied prefix).

    Histogram subplot (``show_prob_histogram``, default True) draws a
    predicted-probability histogram under the reliability scatter, sharing
    the X axis. Y-scale auto-picks log when ``max(hits)/max(min(hits),1) >
    100`` and linear otherwise; override via
    ``prob_histogram_yscale="log" | "linear"``. Inline per-bin population
    text labels next to scatter points are independently controlled by
    ``show_inline_population_labels`` so users can keep both, drop both, or
    keep only one.
    """

    # Floats not ints: every chart layer downstream (FigureSpec.figsize) already takes
    # Tuple[float, float]; an int-only annotation rejected fractional sizes via pydantic.
    figsize: Tuple[float, float] = (15.0, 5.0)
    print_report: bool = True
    show_perf_chart: bool = True
    show_fi: bool = True
    feature_importance_config: Optional[FeatureImportanceConfig] = None
    display_sample_size: int = 0
    show_feature_names: bool = False

    # Per-split metric computation gates (lifted from the trainer-internal
    # TrainingControlConfig so suite users can disable train-set metrics for
    # speed, or enable them for overfit diagnostics). Defaults match the
    # historical trainer-internal hardcoded defaults.
    compute_trainset_metrics: bool = False
    compute_valset_metrics: bool = True
    compute_testset_metrics: bool = True

    # Custom ICE / RICE metric callables (signature: (y_true, y_score) -> float).
    # When None, the trainer falls back to the mlframe-native ICE built from
    # compute_probabilistic_multiclass_error.
    custom_ice_metric: Optional[Callable] = None
    custom_rice_metric: Optional[Callable] = None

    # Histogram subplot - independent toggles for the histogram itself and
    # for the inline population annotations on the scatter plot.
    show_prob_histogram: bool = True
    prob_histogram_yscale: Literal["auto", "log", "linear"] = "auto"
    show_inline_population_labels: bool = True

    # Title-metrics template. Validator parses + populates title_metrics_tokens.
    # 2026-05-28 audit: added KS / MCC / BSS to the default per user
    # preference - the most informative single-number summaries beyond
    # the calibration / AUC family. Gini is available as a token but
    # not in default (it's algebraically derivable from ROC_AUC).
    title_metrics_template: str = "ICE BR_DECOMP ECE CMAEW LL ROC_AUC PR_AUC KS MCC BSS"
    # Populated by the model_validator after title_metrics_template is validated.
    # Stored as a tuple so downstream hot-path code (fast_calibration_report)
    # never has to re-parse the string. Do not set directly - it is overwritten
    # at construction.
    title_metrics_tokens: Tuple[str, ...] = ()

    # 2026-05-28 audit: token-based regression chart title. Default keeps
    # the historical 4 tokens (MAE/RMSE/MaxError/R2) and adds RMSLE,
    # Spearman, MBE per user feedback. Empty tokens (e.g. RMSLE on a
    # signed target) gracefully render as empty fragments.
    regression_title_metrics_tokens: Tuple[str, ...] = (
        "MAE", "RMSE", "MaxError", "R2",
        "RMSLE", "Spearman", "MBE",
    )

    # 2026-05-28 audit batch: MASE seasonality (Hyndman & Koehler 2006).
    # The MASE *value* is only computed when the caller plumbs the
    # precomputed train-fold naive-MAE scale into the regression-report
    # signature (``mase_naive_mae=``); this knob sets the seasonality the
    # caller used so it can be stamped alongside the metric.
    # Common values: 1 (simple naive), 7 (daily->weekly), 12 (monthly->yearly),
    # 24 (hourly->daily). MUST match the seasonality used by the caller.
    mase_seasonality: int = 1

    # backend x output-format DSL. See ``mlframe.reporting.output.parse_plot_output_dsl`` for grammar.
    #
    # Default keeps interactive plotly HTML (for sharing / jupyter) + matplotlib PNG (10-20x faster, no Chromium). Routing PNG export through kaleido spends 12-15s per figure on a Chromium ``page.reload()``; on a 4-model x VAL+TEST x N-ensemble suite this ballooned to MINUTES of pure chart-export wall-time. Users who need plotly PNG explicitly set ``"plotly[html,png]"``.
    plot_outputs: str = "plotly[html] + matplotlib[png]"

    # Opt-out for jupyter inline plot display.
    # ``None`` (default): auto-detect via ``__IPYTHON__`` / ``sys.ps1`` in ``render_and_save`` - inside a notebook kernel, figures render inline in the cell output AFTER on-disk save (the saved file is the artifact; the inline render is the operator-feedback path).
    # ``True``: force inline display (useful for non-standard runtimes where auto-detection misfires).
    # ``False``: save-to-disk only - skips ``renderer.show(fig)`` even when running inside a kernel. Use for batch jupyter runs (papermill, nbconvert, scheduled notebooks) that don't need cell-output renders AND want to skip the per-figure inline render cost (~50-200ms / figure for plotly, ~20-50ms for matplotlib; accumulates to seconds on a 4-model x VAL+TEST x 6-ensemble suite). Also useful when the inline backend is broken (eg plotly.io renderer misconfigured) and the operator wants the on-disk PNG/HTML without cell-output errors.
    plot_inline_display: Optional[bool] = None

    # Matplotlib style + rcParams override.
    #
    # Use cases:
    # - ``matplotlib_style="ggplot"`` -> use the "ggplot" style sheet for all charts the suite emits. Accepts any name resolvable by ``plt.style.use(...)`` (eg ``"seaborn-v0_8-darkgrid"``, ``"dark_background"``, ``"fivethirtyeight"``, ``"_classic_test_patch"``, or a path to a user-written ``.mplstyle`` file).
    # - ``matplotlib_style=["seaborn-v0_8", "dark_background"]`` - list to layer multiple styles (matplotlib stacks them; later wins on conflict).
    # - ``matplotlib_rcparams={"font.size": 12, "axes.grid": True, ...}`` - direct rcParams dict; merged ON TOP of any style sheet so the user can fine-tune specific keys without writing a full .mplstyle file.
    #
    # Application: both fields are applied to the PROCESS-WIDE matplotlib state at suite entry (mirrors the existing ``plot_inline_display`` plumbing). When ``None`` (default), the user's script-level ``plt.style.use(...)`` / ``plt.rcParams`` settings are preserved untouched - so a one-line ``plt.style.use("ggplot")`` before the suite invocation also works for callers who don't want to thread the field through a config object.
    #
    # The fields are NOT reverted on suite exit; matches the ``plot_inline_display`` semantics (operators expect "set once, see everywhere" for plot styling in a long-running notebook session).
    matplotlib_style: Optional[Union[str, List[str]]] = None
    matplotlib_rcparams: Optional[Dict[str, Any]] = None

    # Plotly template override - separate from the matplotlib style because plotly has its own template system. Common values: ``"plotly"`` (default), ``"plotly_white"``, ``"plotly_dark"``, ``"ggplot2"``, ``"seaborn"``, ``"simple_white"``, ``"presentation"``. Applied via ``plotly.io.templates.default = ...`` at suite entry, process-wide (mirrors matplotlib_style semantics). ``None`` (default) keeps the user's pre-suite plotly setting.
    #
    # Ergonomic note: to unify the look across both backends, set BOTH ``matplotlib_style`` and ``plotly_template`` to matching themes, eg ``matplotlib_style="ggplot"`` + ``plotly_template="ggplot2"``. There is no single "theme" knob that targets both because the available style names and rcParams keys differ between backends.
    plotly_template: Optional[str] = None

    # Per-figure DPI for saved PNG / inline rendering. matplotlib's default is 100. Lowering to 80 cuts savefig wall-time ~30% linearly (verified on a 6-panel multiclass figure: 1330ms -> ~900ms) at a visible-but-acceptable resolution loss; raising to 150 sharpens for publication / slides at a ~2.25x cost. ``None`` (default) defers to matplotlib's global default. Honoured by the matplotlib renderer (``MatplotlibRenderer.save``) and by the legacy ``show_calibration_plot`` save path; plotly path (``plot_outputs`` with ``[png]``) routes through kaleido which has its own DPI knob - when both plotly+matplotlib are emitted, only the matplotlib PNG honours this flag.
    plot_dpi: Optional[int] = None

    # Honest-estimator diagnostics aggregator: ON by default. When True, ``finalize_suite`` invokes ``training.honest_diagnostics.run_honest_diagnostics(ctx, models, metadata)`` so every run emits bootstrap CI per top-line metric, categorical PSI drift summary, reliability/calibration plot, and the provenance disposition table. Set False on hot loops or batch runs where the ~1-3s aggregator wall time matters more than the audit trail.
    honest_estimator_diagnostics: bool = True

    # Per-target_type panel templates. Same DSL grammar as ``title_metrics_template`` (space-separated tokens, validator checks against the chart modules' ALLOWED_*_PANEL_TOKENS frozensets, no duplicates). All-by-default; operator removes tokens to skip individual panels.
    # Binary classification previously had no curve charts (only a reliability diagram); these render ROC/PR/SCORE_DIST/KS/THRESHOLD/GAIN by default.
    binary_panels: str = "ROC PR SCORE_DIST KS THRESHOLD GAIN PIT"
    # Opt-in data-aware panel emphasis for the binary report. ``"all"`` (default) keeps the full binary_panels set in order (back-compat, no surprise). ``"data_aware"`` derives the positive base rate from y_true at dispatch and emphasizes the panels that matter for that skew: imbalanced (base rate < emphasis_imbalance_lo or > _hi) leads with PR / THRESHOLD and drops the optimistic-under-imbalance ROC; balanced leads with ROC. Only applies when binary_panels is left at its default; a custom binary_panels is never reordered. Single-class / tiny-n falls back to "all".
    panel_emphasis: Literal["all", "data_aware"] = "all"
    emphasis_imbalance_lo: float = 0.2
    emphasis_imbalance_hi: float = 0.8
    # CONFUSION_MARGINS flanks the confusion heatmap with per-true-class support + per-pred-class precision bars, so a class imbalance that explains a confusion block is visible in one panel.
    multiclass_panels: str = "CONFUSION CONFUSION_MARGINS CONFUSED_PAIRS PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC"
    # THRESHOLD_SWEEP adds the per-label argmax-F1 cutoff heatmap (per-label, not a global threshold) -- the operating-point picker for multilabel.
    multilabel_panels: str = "PR_F1 CALIB_GRID COOCCURRENCE CARDINALITY JACCARD_DIST THRESHOLD_SWEEP"
    ltr_panels: str = "NDCG_K NDCG_DIST NDCG_BY_QSIZE LIFT MRR_DIST SCORE_BY_REL"
    # Kept in lockstep with the composer's ``DEFAULT_QUANTILE_PANELS`` so QUANTILE_RELIABILITY / PINBALL_DECOMP / QUANTILE_CROSSING render on a default suite run (the suite passes this string to the dispatcher, not the composer default). FAN_CHART overlays the nested forecast-uncertainty bands over the horizon.
    quantile_panels: str = (
        "RELIABILITY COVERAGE PINBALL_BY_ALPHA INTERVAL_BAND WIDTH_DIST PIT_HIST "
        "QUANTILE_RELIABILITY PINBALL_DECOMP QUANTILE_CROSSING FAN_CHART"
    )
    # Regression report panels rendered by ``compose_regression_figure``; all-by-default. WORM (de-trended residual QQ) + RESID_ACF (residual autocorrelation, Bartlett band) surface tail-misfit and serial dependence the scatter/hist hide.
    regression_panels: str = "SCATTER RESID_HIST RESID_VS_PRED ERR_BY_DECILE WORM RESID_ACF"

    # Calibration binning strategy for the reliability diagram + ECE bins. ``"auto"`` (default) picks quantile (equal-population) bins under a rare-event base rate where uniform bins collapse to <=2 populated bins, uniform otherwise. ``"uniform"`` / ``"quantile"`` force one strategy.
    calibration_binning: Literal["auto", "uniform", "quantile"] = "auto"
    # Render Wilson 95% binomial confidence bands on the per-bin empirical frequencies in the reliability diagram. ON by default -- the band tells the operator which deviations from the diagonal are sampling noise vs real miscalibration.
    reliability_show_ci: bool = True

    # Per-model train/val metric-vs-iteration curves (from evals_result_ / ES callback history), early-stop point marked. ON by default; the renderer only emits when charts are saved AND iteration history is available, so it is a no-op for models without boosting history.
    training_curves: bool = True

    # Retain the pure-data ``FigureSpec`` objects (no live matplotlib/plotly handle, so they stay pickle-safe) in ``metrics["figure_specs"]`` so callers can re-render or post-tweak panels after the suite. OFF by default -- the on-disk artifact is the primary output; this only matters for programmatic post-processing.
    keep_figure_handles: bool = False

    # Row cap for the regression pred-vs-actual scatter (extremes-preserving subsample keeps the worst-error points). 5000 is dense enough to read the residual structure while bounding render cost; raise for finer detail, lower for faster figures.
    regression_scatter_sample_size: int = 5000

    # Standalone post-fit diagnostics wired into the per-(model, split) report path. All are cheap (reuse the
    # already-computed preds / errors / importances on a bounded subsample) so they default ON; each can be disabled
    # individually for hot loops. The single genuinely-expensive path (non-tree SHAP, learning curve) is opt-in below.
    #
    # PDP/ICE for the top feature-importance features when a fitted model + feature frame are present. Rows are
    # subsampled to ``pdp_sample`` before any predict, so cost is ``pdp_grid`` predicts independent of n.
    pdp_ice: bool = True
    pdp_top_features: int = 4
    pdp_sample: int = 2000
    pdp_grid: int = 20
    # Multi-model leaderboard, assembled per-target from the per-model preds/metrics the suite already collected; only
    # rendered when >=2 models were trained on the same task (single-model runs skip cheaply).
    model_comparison: bool = True
    # Multi-dim weak-slice search on the precomputed per-row error (no model calls); finds the worst feature-value regions.
    slice_finder: bool = True
    # Decision-curve net-benefit analysis for binary targets (needs only y_true + positive-class score).
    decision_curve: bool = True
    # Calibration-drift-over-time for binary targets that carry a split timestamp; reuses the warmed njit ECE kernel.
    calibration_drift: bool = True
    # Target ACF/PACF when the split carries timestamps (serial-dependence diagnostic on the target series).
    target_acf: bool = True
    # SHAP beeswarm + top-K dependence. Default-ON for TREE models (exact fast TreeExplainer, cost scales with the
    # explained-row cap not n); for NON-tree models the slow KernelExplainer path is OFF unless ``shap_allow_kernel``.
    shap_panels: bool = True
    shap_max_rows: int = 20000
    shap_top_k: int = 6
    shap_allow_kernel: bool = False
    # Per-instance SHAP attribution for the top-K most-confident-wrong predictions (tree models only). Niche +
    # adds render cost beyond the global beeswarm, so OPT-IN; explains WHY a specific costly error happened.
    shap_per_instance: bool = False
    # Combined single-page HTML index per (model, split) stitching the rendered chart artifacts (PNG refs + plotly
    # fragments); assembly-only, default-on when charts are saved.
    combined_html: bool = True
    # Decile gain/lift/KS table figure for binary targets (the tabular complement to the GAIN curve). Default-ON; reuses the already-computed score, single O(n log n) sort.
    decile_table: bool = True
    # One-glance per-(model, split) model card: header metrics + traffic-light verdict + mini sparklines. Default-ON when charts saved; reuses the split's y_true + scores/preds.
    model_card: bool = True
    # Cross-split overfit panel per model (grouped headline-metric bars + delta table + verdict), rendered once all splits exist. Default-ON when >=2 usable splits.
    split_comparison_charts: bool = True
    # CUSUM change-point on standardized regression residuals; catches a sustained mean shift per-bucket residual_vs_time misses. Default-ON for regression when timestamps cover the split.
    cusum_drift: bool = True
    # Ensemble member-disagreement panels (spread histogram + spread-vs-mean + uncertainty calibration). Default-ON when an ensemble exposes >=2 member predictions, else skipped.
    prediction_stability: bool = True
    # Per-subgroup reliability small-multiples + per-group ECE for binary targets when fairness subgroups are present: surfaces whether the model is calibrated EQUALLY across groups (equal accuracy != equal calibration) via the max-min ECE disparity gap + traffic-light. Default-ON when charts saved AND subgroups configured; a no-op otherwise.
    fairness_calibration_charts: bool = True
    # Per-feature calibration for binary targets: reliability + ECE conditioned on quantile bins of the top-importance continuous feature(s), surfacing whether calibration degrades across a feature's range (the max-min "calibration heterogeneity" metric) -- a miscalibration a single pooled reliability curve hides. Default-ON when charts saved AND a feature frame is present; a no-op otherwise.
    calibration_by_feature_charts: bool = True
    # 2D calibration-ECE heatmap for binary targets: quantile-bin the TOP-2-importance features into a grid and show per-cell |mean(score)-mean(true)| on an RdYlGn_r heatmap, surfacing a LOCALIZED miscalibration pocket at a joint corner (high f0 AND high f1) that the pooled / 1D per-feature views average away. Headline = worst-cell ECE + its (f_x bin, f_y bin) location. Default-ON when charts saved AND a feature frame is present; a no-op otherwise.
    calibration_heatmap_2d_charts: bool = True
    # Opt-in learning curve (holdout score vs train size). ``None`` / ``enabled=False`` skips it: it is K full refits
    # by construction, the documented cost-gated exception to "cheap diagnostics default on". Set
    # ``LearningCurveConfig(enabled=True)`` (optionally ``warm_start=True`` / ``time_budget_s=...``) to turn it on.
    learning_curve: Optional[LearningCurveConfig] = None

    @field_validator("title_metrics_template")
    @classmethod
    def _validate_title_template(cls, v: str) -> str:
        toks = [t.strip().upper() for t in v.split() if t.strip()]
        unknown = [t for t in toks if t not in _REPORTING_ALLOWED_TITLE_TOKENS]
        if unknown:
            raise ValueError(
                f"Unknown title-metrics tokens {unknown}. "
                f"Allowed: {sorted(_REPORTING_ALLOWED_TITLE_TOKENS)}"
            )
        if len(toks) != len(set(toks)):
            dupes = sorted({t for t in toks if toks.count(t) > 1})
            raise ValueError(f"Duplicate title-metrics tokens: {dupes}")
        if "BR" in toks and "BR_DECOMP" in toks:
            raise ValueError(
                "BR and BR_DECOMP are mutually exclusive in title_metrics_template"
            )
        return v

    @field_validator("plot_outputs")
    @classmethod
    def _validate_plot_outputs(cls, v: str) -> str:
        # Defer to the DSL parser; it raises ValueError on any malformed
        # / unsupported / duplicate clause. We don't store the parsed
        # spec on the config -- callers re-parse at render time (cheap;
        # parser is regex-based and runs once per chart).
        from mlframe.reporting.output import parse_plot_output_dsl
        parse_plot_output_dsl(v)
        return v

    @field_validator(
        "binary_panels", "multiclass_panels", "multilabel_panels", "ltr_panels",
        "quantile_panels", "regression_panels",
    )
    @classmethod
    def _validate_panel_template(cls, v: str, info) -> str:
        # Source the allowed token sets from the chart modules' own
        # ALLOWED_*_PANEL_TOKENS frozensets (the single source of truth that
        # the composers also key off), so any new builder token is valid here
        # without a duplicated literal that can drift. Imported lazily: this
        # validator runs at config CONSTRUCTION, not module import, so there
        # is no import cycle with the reporting layer.
        from mlframe.reporting.charts import (
            ALLOWED_BINARY_PANEL_TOKENS,
            ALLOWED_LTR_PANEL_TOKENS,
            ALLOWED_MULTICLASS_PANEL_TOKENS,
            ALLOWED_MULTILABEL_PANEL_TOKENS,
            ALLOWED_QUANTILE_PANEL_TOKENS,
            ALLOWED_REGRESSION_PANEL_TOKENS,
        )
        _ALLOWED = {
            "binary": ALLOWED_BINARY_PANEL_TOKENS,
            "multiclass": ALLOWED_MULTICLASS_PANEL_TOKENS,
            "multilabel": ALLOWED_MULTILABEL_PANEL_TOKENS,
            "ltr": ALLOWED_LTR_PANEL_TOKENS,
            "quantile": ALLOWED_QUANTILE_PANEL_TOKENS,
            "regression": ALLOWED_REGRESSION_PANEL_TOKENS,
        }
        target_key = info.field_name.replace("_panels", "")
        allowed = _ALLOWED[target_key]
        toks = [t.strip().upper() for t in v.split() if t.strip()]
        unknown = [t for t in toks if t not in allowed]
        if unknown:
            raise ValueError(
                f"Unknown {target_key} panel tokens {unknown}. "
                f"Allowed: {sorted(allowed)}"
            )
        if len(toks) != len(set(toks)):
            dupes = sorted({t for t in toks if toks.count(t) > 1})
            raise ValueError(
                f"Duplicate {target_key} panel tokens: {dupes}"
            )
        return v

    @field_validator("feature_importance_config", mode="before")
    @classmethod
    def _coerce_feature_importance_config(cls, v):
        """Accept ``FeatureImportanceConfig`` instances even when the Python
        class identity has diverged.

        Pydantic v2's ``model_type`` validator strictly checks
        ``type(instance) is FeatureImportanceConfig``. Two practical scenarios
        break that without any code bug on either side:
          1) ``%autoreload 2`` in a Jupyter session re-imports ``configs.py``
             after a code edit -- new ``FeatureImportanceConfig`` class is
             defined, but ``trainer.py`` (already imported earlier) still
             references the OLD class and instantiates from it.
          2) Two separate working copies of mlframe sit on ``sys.path`` (e.g.
             a recovery checkout + the canonical one) and import resolution
             picks one for ``configs`` and the other for ``trainer``.

        Both produce ``input_value=FeatureImportanceConfig(...),
        input_type=FeatureImportanceConfig`` errors that are confusing
        because the names match. Round-tripping through ``model_dump()``
        rebuilds the instance against THIS module's class identity and
        recovers transparently. Same-class instances pass through.
        """
        if v is None:
            return None
        if isinstance(v, FeatureImportanceConfig):
            return v
        # Stale-class shim: anything pydantic-shaped with the right name.
        if hasattr(v, "model_dump") and type(v).__name__ == "FeatureImportanceConfig":
            return FeatureImportanceConfig(**v.model_dump())
        # Dicts pass through normal pydantic validation (handled by the
        # default validator after this one returns a dict).
        return v

    @model_validator(mode="after")
    def _populate_title_tokens(self) -> "ReportingConfig":
        toks = tuple(t.strip().upper() for t in self.title_metrics_template.split() if t.strip())
        # Bypass validate_assignment for this derived field so we don't recurse.
        object.__setattr__(self, "title_metrics_tokens", toks)
        return self


class ConfidenceAnalysisConfig(BaseConfig):
    """Confidence analysis configuration for train_and_evaluate_model.

    Controls SHAP-based confidence analysis of model predictions.

    Parameters
    ----------
    include : bool
        Whether to include confidence analysis (default: False).
    use_shap : bool
        Whether to use SHAP for explanations (default: True).
    max_features : int
        Maximum features to show in plots (default: 6).
    cmap : str
        Colormap for plots (default: "bwr").
    alpha : float
        Transparency for plot points (default: 0.9).
    ylabel : str
        Y-axis label for plots.
    title : str
        Plot title.
    model_kwargs : dict
        Additional kwargs for confidence model. Keys: n_estimators, max_depth.
    """

    include: bool = False
    use_shap: bool = True
    max_features: int = 6
    cmap: str = "bwr"
    alpha: float = 0.9
    ylabel: str = "Feature value"
    title: str = "Confidence of correct Test set predictions"
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)  # keys: n_estimators, max_depth


class NamingConfig(BaseConfig):
    """Model naming configuration for train_and_evaluate_model.

    Controls model naming for reports and saved files.

    Parameters
    ----------
    model_name : str
        Name of the model for reports.
    model_name_prefix : str
        Prefix to add before model type in names.
    """

    model_name: str = ""
    model_name_prefix: str = ""


class PredictionsContainer(BaseConfig):
    """Container for pre-computed predictions (used in just_evaluate mode).

    Holds predictions and probabilities for train/val/test splits.

    Parameters
    ----------
    train_preds : np.ndarray, optional
        Training set predictions.
    train_probs : np.ndarray, optional
        Training set probabilities.
    val_preds : np.ndarray, optional
        Validation set predictions.
    val_probs : np.ndarray, optional
        Validation set probabilities.
    test_preds : np.ndarray, optional
        Test set predictions.
    test_probs : np.ndarray, optional
        Test set probabilities.
    """

    train_preds: Optional[Any] = None  # np.ndarray
    train_probs: Optional[Any] = None  # np.ndarray
    val_preds: Optional[Any] = None  # np.ndarray
    val_probs: Optional[Any] = None  # np.ndarray
    test_preds: Optional[Any] = None  # np.ndarray
    test_probs: Optional[Any] = None  # np.ndarray


class FairnessConfig(BaseConfig):
    """Fairness analysis configuration.

    Controls fairness metric computation across demographic subgroups.

    Parameters
    ----------
    enabled : bool
        Whether to enable fairness analysis (default: False).
    protected_attributes : list of str, optional
        Column names of protected attributes.
    fairness_metrics : list of str, optional
        Fairness metrics to compute (e.g., "demographic_parity", "equalized_odds").
    """

    enabled: bool = False
    protected_attributes: Optional[List[str]] = None
    fairness_metrics: Optional[List[str]] = None


