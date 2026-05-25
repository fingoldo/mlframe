"""Feature-distribution drift report -- per-feature train/val/test shift signal.

OBSERVATIONAL ONLY. The 2026-05-22 review explicitly flagged 3 weaknesses
in any "feature drift -> auto-action" coupling:

1. **There is no operator**. The mlframe suite runs autonomously; "operator
   reads the WARN and decides" is the wrong framing for an autonomous
   pipeline. The sensor must produce signal the SYSTEM can use, not text
   a human will read.

2. **Drift does NOT prove model harm.** A 5-sigma drift on a feature with
   FI=0 is harmless; a 2-sigma drift on a feature with FI=0.8 (dominant)
   is potentially catastrophic. Per-feature z-score alone is NOT a
   grounded "the model will fail" signal -- pairing with feature
   importance is required for any actionable claim, and even then it's
   a correlation not a guarantee.

3. **Feature selection runs inside per-model pre_pipeline.** MRMR / RFECV
   drop features before the model sees them; this sensor flags features
   the model NEVER receives. Drift on a dropped feature is observationally
   uninteresting.

So the actionable layer in the TVT-2026-05-21 protection stack is the K=2
ensemble catastrophic-dropout (target-aware, measures actual MAE
divergence after training). This sensor is COMPLEMENTARY: it stamps
per-feature drift stats into metadata for post-mortem correlation
("the K=2 dropout fired -- here are the drift candidates that MIGHT
explain it") and surfaces an INFO line on extreme drift (>= 10x default
threshold) where the correlation is strong enough to log without false-
positive risk.

Implementation notes
--------------------
- Per-numeric-feature drift = (other_mean - train_mean) / train_std (z-score).
- We only consider numeric columns; categorical drift via PSI is a
  separate (more expensive) pass.
- Default threshold 3.0 sigma for the metadata stamp; INFO-log fires at
  3.0 sigma, WARN-log only at >= 10x sigma (extreme outliers where the
  signal is strong enough to act on independently).
- Report stamps into ``metadata["feature_distribution_drift"]`` so
  observability tooling can plot per-feature drift trends.

Public API
----------
``compute_feature_distribution_drift(train_df, val_df, test_df, *,
warn_threshold_z=3.0) -> dict``
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_FEATURE_DRIFT_WARN_THRESHOLD_Z: float = 3.0
"""Default warn threshold: emit if any feature's val or test mean is more
than this many train-std away from the train mean."""


ROBUST_RECURRENT_OVERRIDES_UNDER_DRIFT: Dict[str, Any] = {}
"""Placeholder for a future recurrent-model (LSTM / GRU / RNN /
Transformer) override under feature drift.

The MLP override family below was grounded by sweeps over a tabular
sklearn-MLP. The recurrent failure mode under drift is mechanically
different: feature/sequence shift evolves the hidden state through
unfamiliar regions, and the cure is more about input gating / robust
attention / sequence normalisation than the "collapse to linear"
strategy that wins on tabular MLP. Until a recurrent-sequence-DGP
sweep grounds equivalent values, this dict stays EMPTY and the wire-in
in ``_phase_recurrent`` does not auto-apply any recurrent override.
"""


ROBUST_MLP_OVERRIDES_UNDER_DRIFT_CLASSIFICATION: Dict[str, Any] = {
    "alpha": 1.0,
    "hidden_layer_sizes": (32, 16),
    "activation": "identity",
}
"""Classification counterpart to ``ROBUST_MLP_OVERRIDES_UNDER_DRIFT``.

Grounded by ``profiling/bench_mlp_robustness_sweep_classification.py``:
4 DGPs (linear_binary / interaction_binary / sinusoidal_binary /
linear_multiclass_3) x 48 configs x 15 seeds = 2880 trials phase-1
+ 1344 trials phase-2 = 4224 trials. Per-metric cross-DGP min-max:
  log_loss  : alpha=1.0  hidden=(32,16) identity   (worst-case 0.0191)
  accuracy  : alpha=10   hidden=(16,)   identity   (worst-case 0.0000)
  ROC-AUC   : alpha=1e-4 hidden=(16,)  identity    (worst-case 0.0013)

All three winners use ``activation=identity`` -- same mechanism as
regression (collapse to a linear head). The alpha differs (regression
prefers 1e-4 because the underlying score IS linear; classification
prefers higher L2 because the sigmoid + softmax compresses small
weight differences into similar probabilities). We pick the log_loss
winner because log_loss is the primary classification metric (the only
one of the three that the suite optimizes for via early-stopping in
calibrated classifiers).

The classification grounding regime used drift_z=5 (sigmoid still in
transition) instead of regression's drift_z=10 (which would saturate
the sigmoid and degenerate one class).
"""


ROBUST_MLP_OVERRIDES_UNDER_DRIFT: Dict[str, Any] = {
    "alpha": 1e-4,
    "hidden_layer_sizes": (32, 16),
    "activation": "identity",
}
"""HPT overrides applied to MLPConfig for a target whose FI-weighted feature
drift score exceeds ``WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLD``.

Grounded empirically by the 2026-05-22 sweep stack:

A. ``profiling/bench_mlp_robustness_sweep.py`` (1,440 sklearn-MLP trials,
   LINEAR DGP). Baseline sklearn defaults (alpha=1e-4, hidden=(100,),
   activation=relu) suffered mean MLP_excess_harm = 6.455 R^2 at
   drift_z=10 (catastrophic, matches the TVT-2026-05-21 prod-log
   collapse). Identity-activation configs reduce harm by ~10000x.

B. ``profiling/bench_mlp_robustness_sweep_nonlinear.py`` (3,520 trials
   across 4 DGPs: linear / quadratic_dominant / interaction (x_dom *
   x_2) / sinusoidal (5*sin(x_dom) + 3*x_dom)). Per-trial metrics: R^2
   gap, RMSE/y_std gap, MAE/y_std gap (mlframe.metrics.core fast_*
   variants -- 15-17x faster than sklearn equivalents). Cross-DGP min-
   max winner under EACH metric:

     metric                          winner
     R^2 gap                         alpha=1e-4 hidden=(32,16) identity
     RMSE/y_std gap                  alpha=1e-4 hidden=(32,16) identity
     MAE/y_std gap                   alpha=1e-4 hidden=(32,16) identity

   All three metrics agree, worst-case 0.0023 in RMSE/y_std / 0.0023 in
   MAE/y_std / 0.0108 in R^2 (the R^2 worst-case is the quadratic
   curvature MLP-identity cannot capture; in absolute error terms that's
   still 0.2% relative). Per-DGP for the winning config:
       linear  : R^2gap=+0.000  RMSE/y=+0.002  MAE/y=+0.002
       quadr   : R^2gap=+0.011  RMSE/y=+0.002  MAE/y=+0.002
       inter   : R^2gap=-0.000  RMSE/y=-0.000  MAE/y=-0.000
       sinus   : R^2gap=-5.157  RMSE/y=-0.163  MAE/y=-0.163  (MLP beats Ridge)

The alpha=1e-4 (default sklearn weight decay) winning is non-obvious but
mechanically sound: MLPRegressor has early_stopping=True which already
regularizes; an additional heavy L2 pushes the linear head off the
optimal OLS solution and makes the gap WORSE under RMSE/MAE.

Sklearn-shape keys (alpha / hidden_layer_sizes / activation). The
torch-backed mlframe MLP uses different field names; the consumer at
the wire-in site translates via
``translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs``.

Tradeoff (documented, not a bug): on nonlinear targets with strong
interactions or smooth nonlinearity, applying this override forfeits
some MLP-ReLU nonlinear-capture capacity that the unconstrained model
would have had. The min-max framing prefers this because that
unconstrained capacity comes with worst-case R^2=6+ gap (and 2+ RMSE/
y_std) when drift hits a linear-shaped target -- a real production
incident pattern. The override is gated by drift detection (fires only
when weighted_drift_score >= 3.0), so nonlinear-rich targets WITHOUT
drift keep the original ReLU config and its nonlinear capacity.
"""


WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLDS: Dict[str, Optional[float]] = {
    "regression": 3.0,
    "classification": 3.0,
}
"""Per-target-type FI-weighted drift score threshold above which the
sensor recommends applying robustness overrides to neural-model HPT.

Threshold = None means the trigger is DISABLED for that target-type
family (no auto-apply ever); the override family is still documented in
the relevant ``ROBUST_MLP_OVERRIDES_UNDER_DRIFT*`` constant for manual
use. Threshold = float means "fire above this value, subject to extra
gates (e.g. linear-shape detector for classification)".

Per-type empirical grounding (2026-05-22):

  regression -> 3.0
    Paired study ``profiling/bench_drift_fi_vs_model_harm.py``
    (570 trials, 9 drift_z levels x 3 drift_target modes x 30 seeds).
    Pearson(weighted_drift_score, MLP_excess_R^2_harm) = +0.834 overall.
    At threshold=3.0: precision=1.000, recall=0.883, zero false
    positives. Trigger is well-grounded; no shape gate needed because
    Ridge wins universally on drifted linear / quadratic / interaction /
    sinusoidal regression DGPs.

  classification -> 3.0 + linear-shape gate (see
    ``CLASSIFICATION_LINEAR_SHAPE_MAX_DELTA_VS_RAW_PCT``)
    Paired study ``profiling/bench_drift_fi_vs_model_harm_classification.py``
    (810 trials, 3 binary DGPs x 9 drift_z x 30 seeds).
    Pearson(weighted_drift_score, MLP_excess_logloss) overall: r=-0.101
       per-DGP: linear=+0.233  interaction=-0.227  sinusoidal=-0.041
    The negative per-DGP correlation on interaction_binary is the
    diagnostic: when the underlying signal is genuinely nonlinear
    (x_dom * x_2), MLP-with-relu OUTPERFORMS LogReg under drift -- the
    identity-collapse override would HURT. So we gate classification on
    ``init_score_baseline.delta_vs_raw_pct`` from baseline_diagnostics:
    linear captures within 10% of LightGBM -> linear-shape -> apply.
    Outside that band -> nonlinear-shape -> WARN only.
"""


CLASSIFICATION_LINEAR_SHAPE_MAX_DELTA_VS_RAW_PCT: float = 10.0
"""Threshold on ``abs(init_score_baseline.delta_vs_raw_pct)`` (from
baseline_diagnostics) below which a classification target is treated
as 'linear-shape' for the purposes of auto-applying the MLP HPT
override under drift.

``delta_vs_raw_pct`` measures the relative gap between the linear
baseline (LogisticRegression on top-FI features) and the LightGBM raw
metric (AUC for binary). |delta| <= 10% means LogReg captures
essentially what LightGBM does -- i.e. the target IS linear in feature
space. Per the per-DGP correlation breakdown in
``bench_drift_fi_vs_model_harm_classification.py``, on linear-shape
binary classification the drift-vs-harm correlation is r=+0.233
(weak but positive) -- safe direction. Interaction-rich targets show
delta_vs_raw_pct of 30-50%+ and r=-0.227, where the override hurts.

10% is conservative: linear AUC within 0.9*LGBM_AUC is a tight bound.
Adjust if production data calibrates differently.
"""


# Back-compat alias for callers that consumed the pre-2026-05-22 flat constant.
WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLD: float = WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLDS["regression"]
"""Deprecated -- use ``WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLDS[target_type_group]``
which is per-target-type. Kept for back-compat with callers that
imported the flat constant before per-type thresholds landed."""


def translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs(
    sklearn_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Translate a sklearn-MLPRegressor-shape override dict (produced by the
    robustness bench) into the nested ``mlp_kwargs`` shape consumed by the
    mlframe MLP builder in ``training/helpers.py:get_training_configs``.

    Keys recognised on input (sklearn naming):
      - ``alpha`` (float): L2 weight decay -> mlframe routes via AdamW
        optimizer with ``weight_decay``.
      - ``hidden_layer_sizes`` (tuple[int, ...]): output topology ->
        ``network_params`` ``nlayers`` + neuron sizes.
      - ``activation`` (str): {"relu", "tanh", "identity", "leaky_relu"}.
        ``identity`` collapses to ``nlayers=0`` (a literal linear head).

    Anything else is passed through verbatim under
    ``mlp_kwargs["model_params"]`` so the caller can extend the bench
    without changing this translator.

    The translation is informed but not exhaustive: the bench validates
    the MECHANISM (regularization + capacity + activation), not specific
    field names; the torch-backed mlframe MLP is a different
    implementation. If a bench winner uses a key we cannot translate, we
    skip it and surface the skipped keys in the returned dict's
    ``__untranslated__`` entry for caller-side logging.
    """
    if not sklearn_overrides:
        return {}

    out: Dict[str, Any] = {}
    untranslated: List[str] = []

    if "alpha" in sklearn_overrides:
        try:
            import torch  # local import: torch is an MLP-only optional dep.
            out.setdefault("model_params", {})
            out["model_params"]["optimizer"] = torch.optim.AdamW
            out["model_params"].setdefault("optimizer_kwargs", {})
            out["model_params"]["optimizer_kwargs"]["weight_decay"] = float(
                sklearn_overrides["alpha"]
            )
        except Exception:
            untranslated.append("alpha")

    if "hidden_layer_sizes" in sklearn_overrides:
        hls = sklearn_overrides["hidden_layer_sizes"]
        if isinstance(hls, (tuple, list)) and len(hls) >= 1:
            first = int(hls[0])
            minl = int(hls[-1])
            consec = (first / minl) if minl > 0 else 1.0
            out.setdefault("network_params", {})
            out["network_params"]["nlayers"] = int(len(hls))
            out["network_params"]["first_layer_num_neurons"] = first
            out["network_params"]["min_layer_neurons"] = minl
            out["network_params"]["consec_layers_neurons_ratio"] = float(consec)
        else:
            untranslated.append("hidden_layer_sizes")

    if "activation" in sklearn_overrides:
        try:
            import torch
            _act = str(sklearn_overrides["activation"]).lower()
            out.setdefault("network_params", {})
            if _act == "relu":
                out["network_params"]["activation_function"] = torch.nn.ReLU
            elif _act == "tanh":
                out["network_params"]["activation_function"] = torch.nn.Tanh
            elif _act in ("leaky_relu", "leakyrelu"):
                out["network_params"]["activation_function"] = torch.nn.LeakyReLU
            elif _act == "identity":
                # Identity activation = single linear transform on the
                # input. ``generate_mlp`` requires ``nlayers >= 1``;
                # force ``nlayers=1`` so the network is an honest
                # ``Linear(in, out) -> Identity`` instead of a multi-
                # layer stack that mathematically collapses but
                # is redundantly parameterised, optimises poorly,
                # and catastrophically OOD-extrapolates under
                # covariate shift (prod TVT 2026-05-22: 25->32->16->1
                # Identity stack went to ~-17 sigma on test split,
                # R^2=-326 while Ridge R^2=1.00 on the same data).
                out["network_params"]["activation_function"] = torch.nn.Identity
                out["network_params"]["nlayers"] = 1
                # Aux dropout sources also break linearity if non-zero --
                # zero them so the network really IS linear end-to-end.
                out["network_params"]["dropout_prob"] = 0.0
                out["network_params"]["inputs_dropout_prob"] = 0.0
            else:
                untranslated.append(f"activation={_act}")
        except Exception:
            untranslated.append("activation")

    for k, v in sklearn_overrides.items():
        if k in {"alpha", "hidden_layer_sizes", "activation"}:
            continue
        out.setdefault("model_params", {})[k] = v

    if untranslated:
        out["__untranslated__"] = untranslated
    return out


def translate_sklearn_mlp_overrides_to_recurrent_config_kwargs(
    sklearn_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Translate the same sklearn-MLP-shape override into the
    ``RecurrentConfig`` field shape used by mlframe's LSTM / GRU / RNN /
    Transformer wrappers.

    Mapping (BEST-EFFORT, not empirically grounded for recurrent yet):
      - ``alpha`` -> ``weight_decay`` (RecurrentConfig has it directly).
      - ``hidden_layer_sizes`` -> ``mlp_hidden_sizes`` (the post-recurrent
        MLP head; the recurrent layers themselves are uncontrolled here).
      - ``activation`` -> no direct mapping. ``identity`` cannot collapse
        the recurrent cell to linear, so this key is recorded as
        untranslated rather than silently dropped.

    This translator is NOT used at runtime until
    ``ROBUST_RECURRENT_OVERRIDES_UNDER_DRIFT`` is populated (currently
    empty). The empirical grounding for recurrent will require a
    sequence-DGP sweep -- the MLP sweep does not transfer because the
    recurrent failure mode (hidden-state extrapolation) is mechanically
    different from MLP feature-range extrapolation.
    """
    if not sklearn_overrides:
        return {}
    out: Dict[str, Any] = {}
    untranslated: List[str] = []
    if "alpha" in sklearn_overrides:
        out["weight_decay"] = float(sklearn_overrides["alpha"])
    if "hidden_layer_sizes" in sklearn_overrides:
        hls = sklearn_overrides["hidden_layer_sizes"]
        if isinstance(hls, (tuple, list)) and len(hls) >= 1:
            out["mlp_hidden_sizes"] = tuple(int(h) for h in hls)
        else:
            untranslated.append("hidden_layer_sizes")
    if "activation" in sklearn_overrides:
        # Recurrent cells (LSTM/GRU/RNN) have hard-coded gate activations;
        # transformer uses softmax+MLP. None of them have a single
        # "activation" knob the way sklearn MLPRegressor does. Skip rather
        # than silently force an incompatible value.
        untranslated.append(f"activation={sklearn_overrides['activation']} (no direct recurrent equivalent)")
    if untranslated:
        out["__untranslated__"] = untranslated
    return out


def _numeric_columns(df: Any) -> List[str]:
    """Return the list of numeric column names from a pandas/polars DataFrame."""
    if df is None:
        return []
    # pandas path
    if hasattr(df, "select_dtypes"):
        try:
            return list(df.select_dtypes(include="number").columns)
        except Exception:
            return []
    # polars path -- duck-type the schema walk
    if hasattr(df, "schema") and hasattr(df, "columns"):
        try:
            import polars as pl
            return [name for name, dt in df.schema.items() if dt.is_numeric()]
        except Exception:
            return []
    return []


DEFAULT_CATEGORICAL_PSI_WARN_MODERATE: float = 0.20
"""PSI moderate-warn threshold for categorical features (credit-risk convention: 0.10-0.25 = moderate shift, >=0.25 = high)."""

DEFAULT_CATEGORICAL_PSI_WARN_HIGH: float = 0.25
"""PSI high-warn threshold for categorical features (credit-risk convention)."""


def _categorical_columns(df: Any) -> List[str]:
    """Return the list of categorical / string column names from a pandas/polars DataFrame.

    Pandas: ``category``, ``object``, ``string`` dtypes. Polars: ``Categorical``, ``Enum``, ``Utf8``/``String``.
    """
    if df is None:
        return []
    if hasattr(df, "select_dtypes"):
        try:
            return list(df.select_dtypes(include=["category", "object", "string"]).columns)
        except Exception:
            return []
    if hasattr(df, "schema") and hasattr(df, "columns"):
        try:
            import polars as pl
            _string_types = (pl.Utf8, pl.String) if hasattr(pl, "String") else (pl.Utf8,)
            out: List[str] = []
            for name, dt in df.schema.items():
                if dt == pl.Categorical or dt in _string_types or (hasattr(pl, "Enum") and isinstance(dt, pl.Enum)):
                    out.append(name)
            return out
        except Exception:
            return []
    return []


def _col_value_counts(df: Any, col: str) -> Optional[Dict[Any, int]]:
    """Per-value count for a single column across pandas / polars; returns ``None`` on failure."""
    if df is None:
        return None
    try:
        if hasattr(df, "loc") and not (hasattr(df, "schema") and hasattr(df, "columns") and not hasattr(df, "iloc")):
            # pandas Series.value_counts(dropna=False) keeps NaN as its own bucket; convert via list-of-tuples then dict because pandas may use NaN keys that hash inconsistently.
            try:
                _vc = df[col].value_counts(dropna=False)
                return {k: int(v) for k, v in _vc.items()}
            except Exception:
                return None
        # polars path
        if hasattr(df, "schema") and hasattr(df, "columns"):
            try:
                import polars as pl
                _ser = df[col]
                _vc = _ser.value_counts()
                # polars value_counts returns a 2-column frame (col_name, "count"); column names vary across versions.
                _cols = _vc.columns
                _val_col = _cols[0]
                _cnt_col = _cols[1] if len(_cols) > 1 else "count"
                _vals = _vc[_val_col].to_list()
                _cnts = _vc[_cnt_col].to_list()
                return {v: int(c) for v, c in zip(_vals, _cnts)}
            except Exception:
                return None
    except Exception:
        return None
    return None


def _compute_categorical_psi(
    train_counts: Dict[Any, int],
    other_counts: Dict[Any, int],
    bin_min_count: int = 5,
) -> float:
    """PSI on category-bucket counts between train and another split.

    PSI = sum( (p_other - p_train) * ln(p_other / p_train) ), summed over the union of categories.
    Tiny / zero-bucket categories are clipped to ``bin_min_count / total_n`` to keep ``ln(0)`` and divide-by-zero finite (credit-risk convention; same trick used in ``optbinning``).

    A category that appears in val/test but is absent from train contributes a positive PSI bucket (the new-category case) -- this is the desired behaviour for shift detection, since a new categorical level is exactly the kind of silent prod failure PSI is designed to surface.
    """
    n_train = sum(train_counts.values())
    n_other = sum(other_counts.values())
    if n_train <= 0 or n_other <= 0:
        return float("nan")
    cats = set(train_counts.keys()) | set(other_counts.keys())
    if not cats:
        return 0.0
    floor_train = float(bin_min_count) / float(n_train) if n_train > 0 else 0.0
    floor_other = float(bin_min_count) / float(n_other) if n_other > 0 else 0.0
    psi = 0.0
    for c in cats:
        p_t = max(float(train_counts.get(c, 0)) / n_train, floor_train)
        p_o = max(float(other_counts.get(c, 0)) / n_other, floor_other)
        psi += (p_o - p_t) * math.log(p_o / p_t)
    return float(psi)


def compute_categorical_drift_psi(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    *,
    feature_names: Optional[List[str]] = None,
    bin_min_count: int = 5,
    moderate_threshold: float = DEFAULT_CATEGORICAL_PSI_WARN_MODERATE,
    high_threshold: float = DEFAULT_CATEGORICAL_PSI_WARN_HIGH,
    max_features_in_log: int = 10,
) -> Dict[str, Any]:
    """Per-categorical-feature PSI drift across train / val / test.

    Returns ``{per_feature: {col: {val_psi, test_psi}}, drift_candidates: [(col, max_psi)], moderate_threshold, high_threshold, n_categorical_features}``.
    ``drift_candidates`` is sorted descending by max(val_psi, test_psi); only features whose max exceeds ``moderate_threshold`` are listed.
    New-category cases (a value present in val/test but missing from train) are surfaced as a positive PSI contribution -- see ``_compute_categorical_psi`` docstring.
    """
    cols = feature_names or _categorical_columns(train_df)
    per_feature: Dict[str, Dict[str, float]] = {}
    candidates: List[tuple[str, float]] = []
    for col in cols:
        train_counts = _col_value_counts(train_df, col)
        if not train_counts:
            continue
        val_counts = _col_value_counts(val_df, col) if val_df is not None else None
        test_counts = _col_value_counts(test_df, col) if test_df is not None else None
        val_psi = (
            _compute_categorical_psi(train_counts, val_counts, bin_min_count=bin_min_count)
            if val_counts is not None else float("nan")
        )
        test_psi = (
            _compute_categorical_psi(train_counts, test_counts, bin_min_count=bin_min_count)
            if test_counts is not None else float("nan")
        )
        per_feature[col] = {"val_psi": val_psi, "test_psi": test_psi}
        max_psi = max(
            val_psi if math.isfinite(val_psi) else 0.0,
            test_psi if math.isfinite(test_psi) else 0.0,
        )
        if max_psi >= moderate_threshold:
            candidates.append((col, max_psi))
    candidates.sort(key=lambda pair: -pair[1])
    if candidates:
        _shown = candidates[:max_features_in_log]
        _detail = ", ".join(f"{c}(PSI={p:.3f})" for c, p in _shown)
        if len(candidates) > max_features_in_log:
            _detail += f", +{len(candidates) - max_features_in_log} more"
        _max_psi = candidates[0][1]
        _level = "warning" if _max_psi >= high_threshold else "info"
        getattr(logger, _level)(
            "[categorical-distribution-drift] %d categorical feature(s) PSI >= %.2f between train and val/test (max PSI=%.3f). Categorical PSI: 0.10-0.25 moderate, >=0.25 high (credit-risk convention). Top drifters: %s",
            len(candidates), moderate_threshold, _max_psi, _detail,
        )
    return {
        "per_feature": per_feature,
        "drift_candidates": [(c, p) for c, p in candidates],
        "moderate_threshold": moderate_threshold,
        "high_threshold": high_threshold,
        "n_categorical_features": len(per_feature),
    }


def _col_to_numpy(df: Any, col: str) -> Optional[np.ndarray]:
    """Best-effort 1-D numpy array for a single column across pandas / polars."""
    try:
        if hasattr(df, "loc"):
            return df[col].to_numpy()
        if hasattr(df, "to_numpy") and hasattr(df, "columns"):
            # polars frame
            return df[col].to_numpy()
    except Exception:
        return None
    return None


def compute_feature_distribution_drift(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    *,
    warn_threshold_z: float = DEFAULT_FEATURE_DRIFT_WARN_THRESHOLD_Z,
    feature_names: Optional[List[str]] = None,
    feature_importance: Optional[Dict[str, float]] = None,
    max_features_in_log: int = 10,
    target_type: Optional[str] = None,
    linear_shape_delta_vs_raw_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute per-feature mean drift across train / val / test.

    Returns a dict with:
      - ``per_feature``: ``{col: {"train_mean", "train_std", "val_z",
        "test_z"}}`` for each numeric feature.
      - ``drift_candidates``: list of (col, max_abs_z) where max_abs_z
        exceeds ``warn_threshold_z``, sorted descending.
      - ``weighted_drift_score``: optional importance-weighted aggregate
        = sum(|z_i| * |fi_i|) / sum(|fi_i|). Populated only when
        ``feature_importance`` is supplied. Higher = more risk because
        the IMPORTANT features (high FI) are drifting; a non-FI-weighted
        high-z on irrelevant features is much less worrying.
      - ``recommend_neural_overrides``: dict of MLP HPT overrides to apply
        for this target when ``weighted_drift_score >=
        WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLD`` (3.0 by default), or
        ``None`` when no override is recommended. Keys are MLPConfig field
        names; values are the empirically-grounded settings from the
        2026-05-22 robustness sweep that close the Ridge-vs-MLP gap on
        drifted data. Downstream model-selection should merge this dict
        into the MLP config for the target (not drop the model -- drop
        loses stacking diversity).
      - ``threshold``: the z-threshold that fired log lines.
      - ``n_numeric_features``: count of numeric features inspected.

    The function never raises on missing val/test (returns z=NaN for the
    missing slot). Per-feature z is NaN when train_std==0 (constant
    feature -- no drift signal can be extracted).

    Log level depends on the magnitude AND the FI-weighted score:
      - <3 sigma: silent.
      - 3-10 sigma: INFO log (observational; per-model FS may drop these).
      - >=10 sigma OR weighted_drift_score >= 1.0: WARN log (correlation
        with model harm is strong enough to escalate).

    ``feature_importance`` is an optional dict ``{col: fi}`` from any
    reasonable source (sklearn coef_ magnitudes, mlframe FI top-K, MRMR
    scores). Missing columns get 0 weight. The aggregate skips features
    with NaN z (constant features).
    """
    cols = feature_names or _numeric_columns(train_df)
    per_feature: Dict[str, Dict[str, float]] = {}
    candidates: List[tuple[str, float]] = []
    # bench-attempt-rejected 2026-05-23: tried vectorising as
    # train_df[cols].mean() + .std(ddof=0) once per side. profiling bench
    # bench_feature_drift_vectorize.py shows loop=170ms vec=162ms (1.05x)
    # at 200k x K=30, and vec is SLOWER at smaller sizes (0.42x at 10k x
    # K=10) due to pandas Series-construction overhead. Per-col numpy
    # path stays.
    for col in cols:
        train_vals = _col_to_numpy(train_df, col)
        if train_vals is None:
            continue
        train_vals = np.asarray(train_vals, dtype=np.float64)
        train_vals = train_vals[np.isfinite(train_vals)]
        if train_vals.size < 2:
            continue
        train_mean = float(np.mean(train_vals))
        train_std = float(np.std(train_vals))
        if train_std <= 0.0 or not math.isfinite(train_std):
            # Constant feature -- no drift signal computable.
            per_feature[col] = {
                "train_mean": train_mean,
                "train_std": 0.0,
                "val_z": float("nan"),
                "test_z": float("nan"),
            }
            continue

        def _z_for(other_df):
            if other_df is None:
                return float("nan")
            other = _col_to_numpy(other_df, col)
            if other is None:
                return float("nan")
            other = np.asarray(other, dtype=np.float64)
            other = other[np.isfinite(other)]
            if other.size < 2:
                return float("nan")
            return float((np.mean(other) - train_mean) / train_std)

        val_z = _z_for(val_df)
        test_z = _z_for(test_df)
        per_feature[col] = {
            "train_mean": train_mean,
            "train_std": train_std,
            "val_z": val_z,
            "test_z": test_z,
        }
        max_abs_z = max(
            abs(val_z) if math.isfinite(val_z) else 0.0,
            abs(test_z) if math.isfinite(test_z) else 0.0,
        )
        if max_abs_z > warn_threshold_z:
            candidates.append((col, max_abs_z))
    candidates.sort(key=lambda pair: -pair[1])
    # Compute the FI-weighted aggregate when feature_importance is provided.
    # Without FI this is None -- the per-feature z-scores alone aren't strong
    # enough to be a grounded harm signal (drift on a fi=0 feature is
    # harmless; drift on a dominant feature is potentially catastrophic).
    weighted_score: Optional[float] = None
    if feature_importance is not None:
        _num = 0.0
        _den = 0.0
        for col, entry in per_feature.items():
            fi = float(abs(feature_importance.get(col, 0.0)))
            if fi == 0.0:
                continue
            _z = max(
                abs(entry["val_z"]) if math.isfinite(entry["val_z"]) else 0.0,
                abs(entry["test_z"]) if math.isfinite(entry["test_z"]) else 0.0,
            )
            _num += _z * fi
            _den += fi
        if _den > 0:
            weighted_score = float(_num / _den)

    # Pick the target-type-appropriate override family. Regression uses
    # ROBUST_MLP_OVERRIDES_UNDER_DRIFT (alpha=1e-4, the score-is-linear
    # winner). Classification uses ROBUST_MLP_OVERRIDES_UNDER_DRIFT_CLASSIFICATION
    # (alpha=1.0, the log-loss winner -- sigmoid + softmax compress small
    # weight differences so more L2 regularization is empirically better).
    # When target_type is not supplied (legacy callers) fall back to the
    # regression family for back-compat.
    _is_classification = (
        target_type is not None
        and str(target_type).lower() != "regression"
        and "ranking" not in str(target_type).lower()
    )
    _override_family = (
        ROBUST_MLP_OVERRIDES_UNDER_DRIFT_CLASSIFICATION
        if _is_classification
        else ROBUST_MLP_OVERRIDES_UNDER_DRIFT
    )
    # Pick the target-type-grouped threshold. Regression: 3.0 (universally
    # grounded). Classification: 3.0 BUT additionally gated by the linear-
    # shape detector below -- interaction-rich classification targets show
    # NEGATIVE drift/harm correlation (MLP-with-relu beats LogReg under
    # drift) so the identity-collapse override would actively hurt.
    _per_type_threshold = WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLDS.get(
        "classification" if _is_classification else "regression",
    )
    # Shape gate: applies to classification only. ``linear_shape_delta_vs_raw_pct``
    # comes from ``baseline_diagnostics.init_score_baseline.delta_vs_raw_pct``:
    # the relative gap between the linear baseline (LogisticRegression on
    # top-FI features) and the LightGBM raw metric. When abs(delta) <=
    # CLASSIFICATION_LINEAR_SHAPE_MAX_DELTA_VS_RAW_PCT the target is
    # linear-shape and the override is empirically safe. Outside that
    # band (or when the signal is unavailable) the override stays off.
    _shape_gate_passes = True
    if _is_classification:
        if linear_shape_delta_vs_raw_pct is None:
            _shape_gate_passes = False  # no signal -> conservatively skip
        else:
            _shape_gate_passes = (
                abs(float(linear_shape_delta_vs_raw_pct))
                <= CLASSIFICATION_LINEAR_SHAPE_MAX_DELTA_VS_RAW_PCT
            )
    recommend_neural_overrides: Optional[Dict[str, Any]] = None
    if (
        weighted_score is not None
        and _per_type_threshold is not None
        and weighted_score >= _per_type_threshold
        and _override_family
        and _shape_gate_passes
    ):
        recommend_neural_overrides = dict(_override_family)

    if candidates:
        _shown = candidates[:max_features_in_log]
        _detail = ", ".join(f"{c}(|z|={z:.2f})" for c, z in _shown)
        if len(candidates) > max_features_in_log:
            _detail += f", +{len(candidates) - max_features_in_log} more"
        # 2026-05-22: WARN only when (a) extreme drift (>=10x threshold) OR
        # (b) FI-weighted score >= 1.0 (important features are drifting). The
        # plain "drift > 3 sigma" case is INFO-only because it's not a
        # grounded harm signal -- per-model FS may drop the feature, or it
        # may be FI~0 and the drift is irrelevant.
        _max_abs_z = candidates[0][1] if candidates else 0.0
        _escalate = (
            _max_abs_z >= 10.0 * warn_threshold_z
            or (weighted_score is not None and weighted_score >= 1.0)
        )
        _level = "warning" if _escalate else "info"
        _ws_str = (
            f", weighted_drift={weighted_score:.2f}" if weighted_score is not None else ""
        )
        _override_str = (
            f", recommend_neural_overrides={recommend_neural_overrides}"
            if recommend_neural_overrides else ""
        )
        getattr(logger, _level)(
            "[feature-distribution-drift] %d numeric feature(s) drift > %.1f sigma "
            "between train and val/test (max |z|=%.2f%s%s). Observational only -- "
            "drift does NOT guarantee model harm; per-model FS may drop these "
            "before training; K=2 ensemble catastrophic-dropout is the "
            "actionable target-aware protection downstream. Top drifters: %s",
            len(candidates), warn_threshold_z, _max_abs_z, _ws_str, _override_str, _detail,
        )
    # Categorical-side PSI report woven into the same return dict so a single call surfaces both. Cheap (~ms per col on typical cardinality) and uses the same train/val/test trio already in scope; explicit feature_names list filters the numeric pass to numeric cols only, so the categorical scan walks the full schema.
    categorical_psi = compute_categorical_drift_psi(train_df, val_df, test_df)
    return {
        "per_feature": per_feature,
        "drift_candidates": [(c, z) for c, z in candidates],
        "weighted_drift_score": weighted_score,
        "recommend_neural_overrides": recommend_neural_overrides,
        "threshold": warn_threshold_z,
        "n_numeric_features": len(per_feature),
        "categorical_psi": categorical_psi,
    }
