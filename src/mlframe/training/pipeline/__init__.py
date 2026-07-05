"""
Pipeline functions for mlframe training.

Handles Polars-ds and sklearn pipeline creation, fitting, and transformation.
"""

from __future__ import annotations

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging
import os
from timeit import default_timer as timer
import subprocess

logger = logging.getLogger(__name__)

# Thread-count env vars must be set BEFORE Julia/PySR boots; we defer the set until the first
# ``_apply_pysr_fe`` call so importers who never touch PySR don't get their env mutated.
def _maybe_set_pysr_thread_env() -> None:
    """Set PYTHON_JULIACALL_THREADS=auto + JULIA_NUM_THREADS=<num> if unset.

    PYTHON_JULIACALL_THREADS gets the literal string ``"auto"`` (not a number): PySR's
    juliacall bridge auto-detects core count when this is "auto", and emits a
    ``UserWarning: PYTHON_JULIACALL_THREADS environment variable is set to something other than
    'auto', so PySR was not able to set it`` when a numeric value blocks PySR's own auto-setup
    (pysr/julia_import.py:27). JULIA_NUM_THREADS still gets a numeric value for the legacy
    manually-launched-Julia path that doesn't read the juliacall var.

    Caller may pin either var before this is called -- pre-set values are preserved.
    """
    try:
        if "PYTHON_JULIACALL_THREADS" not in os.environ:
            os.environ["PYTHON_JULIACALL_THREADS"] = "auto"
        if "JULIA_NUM_THREADS" not in os.environ:
            _ncpu = os.cpu_count() or 4
            os.environ["JULIA_NUM_THREADS"] = str(max(2, _ncpu // 2))
    except Exception:
        pass

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import polars as pl
import polars.selectors as cs
from typing import Dict, Union, Optional, List, Tuple
from collections import Counter
from pyutilz.system import clean_ram
from ..utils import maybe_clean_ram_adaptive
from pyutilz.pandaslib import ensure_dataframe_float32_convertability

from ..utils import log_ram_usage
from ..configs import PreprocessingBackendConfig, PreprocessingExtensionsConfig
from ..strategies import PANDAS_CATEGORICAL_DTYPES, get_polars_cat_columns

_SCALER_FACTORIES = {
    "StandardScaler": lambda: __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler(),
    "StandardScaler_nomean": lambda: __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler(with_mean=False),
    "RobustScaler": lambda: __import__("sklearn.preprocessing", fromlist=["RobustScaler"]).RobustScaler(),
    "MinMaxScaler": lambda: __import__("sklearn.preprocessing", fromlist=["MinMaxScaler"]).MinMaxScaler(),
    "MaxAbsScaler": lambda: __import__("sklearn.preprocessing", fromlist=["MaxAbsScaler"]).MaxAbsScaler(),
    "PowerTransformer_yj": lambda: __import__("sklearn.preprocessing", fromlist=["PowerTransformer"]).PowerTransformer(method="yeo-johnson", standardize=True),
    "PowerTransformer_yj_nostd": lambda: __import__("sklearn.preprocessing", fromlist=["PowerTransformer"]).PowerTransformer(method="yeo-johnson", standardize=False),
    "QuantileTransformer_uniform": lambda: __import__("sklearn.preprocessing", fromlist=["QuantileTransformer"]).QuantileTransformer(output_distribution="uniform"),
    "QuantileTransformer_normal": lambda: __import__("sklearn.preprocessing", fromlist=["QuantileTransformer"]).QuantileTransformer(output_distribution="normal"),
}
# Row-wise normalizers (Normalizer with norm=l2/l1/max) were previously listed
# here under "scaler". They are NOT column scalers: they project each *sample*
# onto a unit hypersphere, which silently breaks tree-based models that rely
# on absolute feature magnitudes. They are excluded here; row-wise transforms will
# get a dedicated `row_transform` slot (see README.md "Roadmap").


def _build_extension_steps(config: PreprocessingExtensionsConfig, n_features: int, random_state: int = 42) -> list:
    """Assemble the ordered list of (name, transformer) pairs for the extensions config.

    Raises ImportError for missing optional deps (UMAP) with an install hint.
    Raises ValueError when PolynomialFeatures would exceed memory_safety_max_features.
    """
    from sklearn.preprocessing import Binarizer, KBinsDiscretizer, PolynomialFeatures
    steps = []
    # NaN-imputation guard: KBinsDiscretizer,
    # PolynomialFeatures, RBFSampler, Nystroem, and most sklearn
    # decompositions (PCA, TruncatedSVD, FastICA, ...) reject NaN at
    # fit time with ``ValueError: Input X contains NaN``. The
    # mlframe upstream preprocessing handles NaN for the GBDT
    # backends (CB / HGB / XGB) which tolerate NaN natively, so
    # numeric NaN can survive into ``apply_preprocessing_extensions``
    # untouched. Prepend a SimpleImputer so
    # any active extension step sees finite values; on clean data
    # the imputer is a near-zero-cost no-op (one statistic per
    # column).
    if (
        config.scaler is not None
        or config.binarization_threshold is not None
        or config.kbins is not None
        or config.polynomial_degree is not None
        or config.nonlinear_features is not None
        or config.dim_reducer is not None
    ):
        from sklearn.impute import SimpleImputer
        # NOTE: non-numeric columns must be filtered out UPSTREAM (in
        # ``apply_preprocessing_extensions`` before pipeline construction)
        # because the downstream sklearn steps (scaler / kbins / poly /
        # nonlinear / dim_reducer) all reject object dtypes. The previous
        # attempt to wrap this imputer in a ColumnTransformer + passthrough
        # left non-numeric columns leaking into the next step, which then
        # raised ``ValueError: The truth value of an array with more than
        # one element is ambiguous`` (surfaced by 1M-harness seed=11
        # where cat_mid='M03' reached the polynomial step). Simpler:
        # let the apply_preprocessing_extensions front gate handle it.
        steps.append(("imputer", SimpleImputer(strategy="median")))
    if config.scaler is not None:
        steps.append(("scaler", _SCALER_FACTORIES[config.scaler]()))
    if config.binarization_threshold is not None:
        steps.append(("binarizer", Binarizer(threshold=config.binarization_threshold)))
    if config.kbins is not None:
        steps.append(("kbins", KBinsDiscretizer(n_bins=config.kbins, encode=config.kbins_encode, strategy="quantile", quantile_method="averaged_inverted_cdf")))
    if config.polynomial_degree is not None:
        # Two-tier projection: ``n ** degree`` is a worst-case upper bound (sufficient for the
        # legacy regression-test contract that callers can rely on a conservative trip wire). When
        # the upper bound is OK we additionally evaluate the exact combinatorial count to surface a
        # better diagnostic when the user is right at the boundary; both stay below the guard,
        # both pass.
        projected_upper = n_features**config.polynomial_degree
        if projected_upper > config.memory_safety_max_features:
            # Exact count for the diagnostic only (no behavioural change vs legacy formula).
            from mlframe.training.feature_handling.polynomial import _projected_output_cols
            projected_exact = _projected_output_cols(
                n_features,
                config.polynomial_degree,
                config.polynomial_interaction_only,
            )
            raise ValueError(
                f"PolynomialFeatures(degree={config.polynomial_degree}, "
                f"interaction_only={config.polynomial_interaction_only}) on {n_features} features "
                f"would produce up to {projected_upper} columns (exact combinatorial: {projected_exact}), "
                f"above memory_safety_max_features={config.memory_safety_max_features}. "
                f"Add dim_reducer='PCA' first or raise the guard."
            )
        steps.append(("poly", PolynomialFeatures(
            degree=config.polynomial_degree,
            interaction_only=config.polynomial_interaction_only,
            include_bias=False,
        )))
    if config.nonlinear_features is not None:
        from sklearn.kernel_approximation import RBFSampler, Nystroem, AdditiveChi2Sampler, SkewedChi2Sampler

        _nl = {"RBFSampler": RBFSampler, "Nystroem": Nystroem, "AdditiveChi2Sampler": AdditiveChi2Sampler, "SkewedChi2Sampler": SkewedChi2Sampler}
        cls = _nl[config.nonlinear_features]
        kw = {"n_components": config.nonlinear_n_components}
        if cls is AdditiveChi2Sampler:
            kw = {}
        else:
            kw["random_state"] = random_state
        steps.append(("nonlinear", cls(**kw)))
    if config.dim_reducer is not None:
        reducer = _build_dim_reducer(config.dim_reducer, config.dim_n_components, random_state)
        steps.append(("dim_reducer", reducer))
    return steps


def _build_dim_reducer(name: str, n_components: int, random_state: int):
    if name == "UMAP":
        import importlib.util as _ilu
        if _ilu.find_spec("umap") is None:
            raise ImportError("UMAP requires `pip install umap-learn`")
        import umap  # type: ignore
        return umap.UMAP(n_components=n_components, random_state=random_state)
    from sklearn.decomposition import PCA, KernelPCA, NMF, TruncatedSVD, FastICA
    from sklearn.manifold import Isomap
    from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
    from sklearn.ensemble import RandomTreesEmbedding
    from sklearn.neural_network import BernoulliRBM
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    factories = {
        "PCA": lambda: PCA(n_components=n_components, random_state=random_state),
        "KernelPCA": lambda: KernelPCA(n_components=n_components, random_state=random_state),
        "LDA": lambda: LinearDiscriminantAnalysis(n_components=n_components),
        "NMF": lambda: NMF(n_components=n_components, random_state=random_state),
        "TruncatedSVD": lambda: TruncatedSVD(n_components=n_components, random_state=random_state),
        "FastICA": lambda: FastICA(n_components=n_components, random_state=random_state),
        "Isomap": lambda: Isomap(n_components=n_components),
        "GaussianRandomProjection": lambda: GaussianRandomProjection(n_components=n_components, random_state=random_state),
        "SparseRandomProjection": lambda: SparseRandomProjection(n_components=n_components, random_state=random_state),
        # RandomTreesEmbedding exposes `n_estimators` (trees), not `n_components` -- the
        # output dim is controlled by tree leaves. Map our `n_components` knob to
        # `n_estimators` for consistency with other dim_reducer factories.
        "RandomTreesEmbedding": lambda: RandomTreesEmbedding(n_estimators=n_components, random_state=random_state),
        "BernoulliRBM": lambda: BernoulliRBM(n_components=n_components, random_state=random_state),
    }
    return factories[name]()


class PySRTransformer:
    """Persisted PySR symbolic-FE step. Holds the fitted ``PySRRegressor`` plus
    the (column_name -> equation_index) map selected at train time, and
    reproduces the same numeric columns on predict frames.

    Kept as a plain class (not sklearn ``BaseEstimator``) because joblib
    pickling of the underlying ``PySRRegressor`` handles Julia-side state via
    PySR's own ``__getstate__``; subclassing BaseEstimator would force a
    get_params/set_params contract we don't need here.
    """

    def __init__(self, model, col_to_index: Dict[str, int], equations: Optional[Dict[str, str]] = None):
        self.model = model
        self.col_to_index = dict(col_to_index)
        self.equations = dict(equations) if equations else {}

    def transform(self, df):
        import numpy as _np
        if df is None:
            return df
        for _col, _idx in self.col_to_index.items():
            if _col in df.columns:
                continue
            df[_col] = _np.asarray(self.model.predict(df, index=int(_idx)), dtype=_np.float32)
        return df

    # Mirror sklearn-pipeline-ish access so ``get_feature_names_out`` callers
    # downstream see the added column names.
    def get_feature_names_out(self):
        return list(self.col_to_index.keys())


class PreprocessingExtensionsBundle:
    """Composite extensions object: PySR transformer + TFIDF dict + sklearn
    pipeline, applied in that order. Mirrors the contract that
    ``apply_preprocessing_extensions`` runs (PySR -> TF-IDF -> sklearn) so the
    persisted object is enough to replay every step.

    Attributes are optional; absent stages are ``None``. ``_apply_extensions_pipeline``
    detects this type and dispatches per-stage; legacy persisted shapes
    (raw dict / raw sklearn Pipeline) are unchanged for backward compatibility.
    """

    def __init__(self, pysr=None, tfidf=None, sklearn_pipe=None):
        self.pysr = pysr
        self.tfidf = tfidf
        self.sklearn_pipe = sklearn_pipe

    @property
    def feature_names_in_(self):
        return getattr(self.sklearn_pipe, "feature_names_in_", None)


def prepare_df_for_catboost(df: pd.DataFrame, cat_features: List[str]) -> None:
    """
    Prepare categorical features for CatBoost.

    Args:
        df: DataFrame (modified in-place)
        cat_features: List of categorical feature names

    Notes:
        CatBoost's Pool rejects NaN in cat_features with "Invalid type for
        cat_feature[object_idx=X,feature_idx=Y]=NaN : cat_features must be
        integer or string, real number values and NaN values should be
        converted to string." Fuzz c0036/c0038 hit this when
        ``skip_categorical_encoding=True`` + pandas input + 10-30% null_frac
        in cat columns. Fill NaN with a sentinel "__MISSING__" BEFORE the
        category cast so the sentinel lands as a valid category level.

        Single-frame variant: each call builds an independent Categorical
        dtype from the frame's own visible values, so codes can drift
        between train/val/test splits ("A" -> code 0 in train but code 1
        in val if val's first row is "B"). When preparing all three splits
        of one training run, prefer ``prepare_dfs_for_catboost_joint`` which
        builds the dtype once from train+val and reuses it.
    """
    for col in cat_features:
        if col in df.columns:
            s = df[col]
            if s.isna().any():
                # Cast to string first so fillna can insert the sentinel
                # (fillna on Categorical rejects unknown values); the
                # round-trip is cheap relative to Pool construction.
                s = s.astype("string").fillna("__MISSING__")
                df[col] = s.astype("category")
            elif s.dtype.name != "category":
                df[col] = s.astype("category")


def prepare_dfs_for_catboost_joint(
    *,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
    cat_features: List[str],
) -> None:
    """Cast cat_features to a Categorical dtype whose category set is the
    JOINT union of train + val. Held-out test must not contribute to the
    union (it has to look "truly unseen" to the model); test values absent
    from the train+val union land as null codes via ``strict=False`` semantics
    of ``pd.Categorical``.

    Pre-fix path used ``prepare_df_for_catboost`` separately per frame so the
    same string value could receive different codes in train vs val vs test -
    e.g. train ``{A,B} -> [0,1]`` and val ``{A,C} -> [0,1]`` mapped "B" and
    "C" to the same code 1 - silently corrupting CatBoost's split decisions.

    All three frames are mutated in place.
    """
    if train_df is None:
        return
    nullable_sentinel = "__MISSING__"
    for col in cat_features:
        if col not in train_df.columns:
            continue
        # Skip embedding-like columns: object-dtype Series whose first
        # non-null cell is an ndarray / list. Calling .astype("string") on
        # such columns calls repr() on every ndarray (~30s on 1M rows) and
        # then crashes with ``TypeError: unhashable type: 'numpy.ndarray'``
        # inside the downstream set()/sorted(). The auto-detect path should
        # already route List(Float32) / pl.Array via embedding_features and
        # exclude them from cat_features, but pandas object-dtype list-of-
        # arrays slipped through (iter#42 fuzz finding: 191s wall + crash).
        _col_series = train_df[col]
        if _col_series.dtype == object:
            try:
                _first = next((v for v in _col_series.head(8) if v is not None), None)
            except Exception:
                _first = None
            if _first is not None and (hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))):
                logger.warning(
                    "prepare_dfs_for_catboost_joint: column '%s' looks like an "
                    "embedding/list column (first cell type=%s); skipping joint-"
                    "Categorical cast. If this is intentional, route it via "
                    "FeatureTypesConfig.embedding_features instead of cat_features.",
                    col, type(_first).__name__,
                )
                continue
        # Collect every value seen in train + val. Apply the same NaN->sentinel
        # rewrite the per-frame variant did, otherwise the union would
        # silently drop null-bearing rows from the category set.
        def _stringify(series):
            if series.isna().any():
                return series.astype("string").fillna(nullable_sentinel)
            if series.dtype.name == "category":
                return series.astype("string")
            return series.astype("string")

        train_s = _stringify(train_df[col])
        union_values = set(train_s.unique().tolist())
        if val_df is not None and col in val_df.columns:
            val_s = _stringify(val_df[col])
            union_values |= set(val_s.unique().tolist())
        else:
            val_s = None

        # Sorted for stable code assignment across reruns; CategoricalDtype uses the supplied order for code positions.
        # Sentinel "__MISSING__" must land at the LAST code (max+1), not code 0: tree libs that pre-pass CTR/one-hot
        # under "low integer codes ~ frequent" heuristics get distorted when the synthetic null bucket sits at 0, and
        # plain alphabetical sort places "__" before letters/digits in ASCII. Split it out and append at the tail so
        # code position is shuffle-stable against the real-category set.
        real_categories = sorted(v for v in union_values if v != nullable_sentinel)
        categories = real_categories + ([nullable_sentinel] if nullable_sentinel in union_values else [])
        joint_dtype = pd.api.types.CategoricalDtype(categories=categories, ordered=False)

        # The docstring contract is explicit in-place mutation of the caller's frames; option_context silences the
        # conservative SettingWithCopy heuristic (fires when a caller hands in a sliced view) without any frame copy.
        with pd.option_context("mode.chained_assignment", None):
            train_df[col] = train_s.astype(joint_dtype)
            if val_df is not None and col in val_df.columns:
                val_df[col] = val_s.astype(joint_dtype)
            if test_df is not None and col in test_df.columns:
                # Test must NOT enlarge the union (see docstring). Use the same
                # dtype; pd.Categorical's astype maps OOV strings to NaN, which
                # matches the polars ``cast(enum_dt, strict=False)`` semantics in
                # apply_polars_categorical_fixes.
                test_s = _stringify(test_df[col])
                test_df[col] = test_s.astype(joint_dtype)


def _select_scalable_numeric_columns(
    train_df: pl.DataFrame,
    method: str,
    q_low: float = 0.25,
    q_high: float = 0.75,
    verbose: int = 0,
) -> List[str]:
    """Return the subset of numeric columns that are safe to feed to a
    polars-ds scaler (``robust`` / ``standard`` / ``min_max``).

    Skips columns that would cause divide-by-zero / divide-by-NaN inside
    the scaler's C++ kernel:

      * All-null columns (the scaler's quantile / mean / min returns
        ``None`` and the subsequent division panics).
      * Constant-value columns (``q_high - q_low == 0`` for ``robust``,
        ``std == 0`` for ``standard``, ``max - min == 0`` for ``min_max``).
      * Columns whose finite range collapses to zero after dropping
        ``inf`` / ``-inf`` / ``nan`` (e.g. ``[+inf, -inf, nan]`` is
        finite-empty).

    The historical workaround forced ``remove_constant_columns=True``
    upstream from the fuzz harness, which masked the bug. The proper
    fix is to filter at the scaler boundary so production users with
    a single zero-spread column don't blow up the whole pipeline
    (fuzz seeds c0008 / c0116).
    """
    scalable: List[str] = []
    skipped_reasons: dict = {}

    # Wave 55 (2026-05-20): validate method at entry rather than silently producing
    # zero stats for an unknown method (previously each per-col elif chain skipped
    # the zero-spread check entirely, and the bogus method propagated to
    # polars_ds.scale where it crashed with a cryptic message).
    if method not in ("robust", "standard", "min_max", "abs_max"):
        raise ValueError(f"_select_scalable_numeric_columns: unknown method={method!r}; " "expected one of 'robust', 'standard', 'min_max', 'abs_max'.")

    numeric_cols = [name for name, dtype in train_df.schema.items() if dtype.is_numeric()]
    if not numeric_cols:
        return scalable

    # Batch all per-col stats into ONE collect via lazy select. The
    # naive per-col path would do 3 collects per col (n_non_null check +
    # 2 stat computations for the chosen ``method``); on ~15 numeric
    # cols with method=robust that's ~45 PyLazyFrame.collect calls (~0.4s
    # wasted). Batched -> 1 collect total.
    has_drop_nans = all(hasattr(train_df[c], "drop_nans") for c in numeric_cols)
    select_exprs = []
    for c in numeric_cols:
        if has_drop_nans:
            # ``is_finite().sum()`` is bit-identical to the prior
            # drop_nulls/drop_nans + finite-filter + len chain: is_finite
            # yields null on null (excluded from sum), False on NaN/inf, so
            # the sum counts exactly the finite, non-null values. Avoids
            # recomputing drop_nulls().drop_nans() twice per column
            # (~4.8x on the n-count select at 50col x 200k).
            n_expr = pl.col(c).is_finite().sum().alias(f"__n__{c}")
        else:
            n_expr = pl.col(c).drop_nulls().len().alias(f"__n__{c}")
        select_exprs.append(n_expr)

        if method == "robust":
            select_exprs.append(pl.col(c).quantile(q_low, interpolation="linear").alias(f"__qlo__{c}"))
            select_exprs.append(pl.col(c).quantile(q_high, interpolation="linear").alias(f"__qhi__{c}"))
        elif method == "standard":
            select_exprs.append(pl.col(c).std().alias(f"__std__{c}"))
        elif method == "min_max":
            select_exprs.append(pl.col(c).min().alias(f"__mn__{c}"))
            select_exprs.append(pl.col(c).max().alias(f"__mx__{c}"))
        elif method == "abs_max":
            select_exprs.append(pl.col(c).abs().max().alias(f"__amx__{c}"))

    try:
        stats_row = train_df.lazy().select(select_exprs).collect()
    except Exception:
        # Fall back to per-col loop on any batched-eval failure.
        stats_row = None

    def _scalar(col_name: str, suffix: str):
        return stats_row[f"__{suffix}__{col_name}"][0] if stats_row is not None else None

    for col_name in numeric_cols:
        try:
            n_non_null = _scalar(col_name, "n")
            if stats_row is None:
                col = train_df[col_name]
                if has_drop_nans:
                    n_non_null = col.is_finite().sum()
                else:
                    n_non_null = col.drop_nulls().len()
            if n_non_null is None or n_non_null == 0:
                skipped_reasons[col_name] = "all-null/non-finite"
                continue

            if method == "robust":
                _q_lo = _scalar(col_name, "qlo")
                _q_hi = _scalar(col_name, "qhi")
                if stats_row is None:
                    col = train_df[col_name]
                    _q_lo = col.quantile(q_low, interpolation="linear")
                    _q_hi = col.quantile(q_high, interpolation="linear")
                if _q_lo is None or _q_hi is None:
                    skipped_reasons[col_name] = "quantile=None"
                    continue
                if _q_hi - _q_lo == 0:
                    skipped_reasons[col_name] = "zero-IQR"
                    continue
            elif method == "standard":
                _std = _scalar(col_name, "std")
                if stats_row is None:
                    _std = train_df[col_name].std()
                if _std is None or _std == 0:
                    skipped_reasons[col_name] = "zero-std"
                    continue
            elif method == "min_max":
                _mn = _scalar(col_name, "mn")
                _mx = _scalar(col_name, "mx")
                if stats_row is None:
                    col = train_df[col_name]
                    _mn, _mx = col.min(), col.max()
                if _mn is None or _mx is None or _mx - _mn == 0:
                    skipped_reasons[col_name] = "zero-range"
                    continue
            elif method == "abs_max":
                _amx = _scalar(col_name, "amx")
                if stats_row is None:
                    _amx = train_df[col_name].abs().max()
                if _amx is None or _amx == 0:
                    skipped_reasons[col_name] = "zero-abs-max"
                    continue
        except Exception as exc:
            skipped_reasons[col_name] = f"check-failed:{type(exc).__name__}"
            continue
        scalable.append(col_name)
    if skipped_reasons and verbose:
        logger.info(
            "  Scaler '%s': skipping %d zero-spread/all-null column(s): %s",
            method, len(skipped_reasons),
            ", ".join(f"{k}({v})" for k, v in list(skipped_reasons.items())[:10]),
        )
    return scalable


def _apply_safe_scaler(
    bp,
    train_df: pl.DataFrame,
    scaler_name: str,
    q_low: float = 0.25,
    q_high: float = 0.75,
    requested_cols: Optional[List[str]] = None,
    verbose: int = 0,
):
    """Append a polars-ds scaling step to ``bp`` with the zero-IQR/zero-spread
    guard applied UNCONDITIONALLY.

    polars-ds ``robust_scale`` divides by ``q_high - q_low`` (``scale`` divides
    by std / range), which yields NaN/inf for any zero-spread (constant /
    all-null / finite-empty) column -- silently corrupting every downstream
    row. This helper recomputes the safe column subset here so the guard holds
    even when a caller passes an explicit ``requested_cols`` that includes a
    zero-IQR column; the safety does not depend on the caller pre-filtering.

    Returns the (possibly unchanged) blueprint. When no column survives the
    filter the scaler step is skipped entirely.
    """
    method = "robust" if scaler_name == "robust" else scaler_name
    safe = _select_scalable_numeric_columns(train_df, method=method, q_low=q_low, q_high=q_high, verbose=verbose)
    if requested_cols is not None:
        _req = set(requested_cols)
        safe = [c for c in safe if c in _req]
    if not safe:
        if verbose:
            logger.info("  No numeric columns survived the zero-spread / all-null filter -- skipping scaler.")
        return bp
    if scaler_name == "robust":
        return bp.robust_scale(safe, q_low=q_low, q_high=q_high)
    return bp.scale(safe, method=scaler_name)


def create_polarsds_pipeline(
    train_df: pl.DataFrame,
    config: PreprocessingBackendConfig,
    pipeline_name: str = "feature_pipeline",
    verbose: int = 1,
    exclude_from_encoding: Optional[set] = None,
):
    """
    Create a Polars-ds pipeline for scaling and encoding.

    Args:
        train_df: Training DataFrame (Polars)
        config: Pipeline configuration
        pipeline_name: Name for the pipeline
        verbose: Verbosity level
        exclude_from_encoding: Column names (e.g. text / embedding features) that
            must NOT be ordinal/onehot-encoded. polars-ds's ``ordinal_encode(cols=None)``
            encodes ALL string-like columns it finds, which includes user-declared
            text_features like ``skills_text`` or synthetic fuzz ``text_0``
            (discovered on fuzz seeds c0085/c0049 -> CB Pool build failed with
            ``Invalid type for text_feature ... =187.0 : text_features must have
            string type`` because the text column arrived as float32 ordinal
            codes). When this set is non-empty, pass an explicit ``cols=`` list
            to the encoder that excludes those columns.

    Returns:
        Materialized PdsPipeline or None if polars-ds not available
    """
    try:
        from polars_ds.pipeline import Pipeline as PdsPipeline, Blueprint as PdsBlueprint
    except ImportError:
        # Wave 41 (2026-05-20): narrowed broad Exception -> ImportError; preserve traceback.
        logger.warning("Could not import polars-ds", exc_info=True)
        return None

    if verbose:
        logger.info("Creating Polars-ds pipeline...")

    excluded = set(exclude_from_encoding or ())

    t0_bp = timer()
    # Build blueprint
    bp = PdsBlueprint(train_df, name=pipeline_name)

    # Imputation -- runs BEFORE scaling so the scaler never sees NaN
    # (NaN * x = NaN propagates through scaling and would leave NaN in
    # the output). Phase M wiring of ``imputer_strategy``: the field was
    # declared since 2026-04 but never connected, so NaN in numeric
    # columns survived the pipeline and crashed downstream models.
    # Sensor tests in ``tests/training/test_imputer_wiring.py``.
    if config.imputer_strategy is not None:
        # Numeric-only target: text/string/categorical columns are
        # handled by the categorical encoder, not here. Reuse the same
        # column filter as the scaler so the two stay aligned.
        _imputable_cols = [name for name, dtype in train_df.schema.items() if dtype.is_numeric() and not dtype == pl.Boolean]
        if _imputable_cols:
            # polars-ds ``Blueprint.impute`` (and our mode path's ``fill_null``)
            # only fill polars NULL -- they leave float ``NaN`` untouched. Real
            # frames carry NaN from numpy/pandas origin and FE ratios (0/0, log
            # of non-positive), so NaN would survive the imputer and reach the
            # scaler / downstream model despite this step. Convert NaN -> NULL on
            # the float imputable columns first so every missing marker is filled.
            _float_imputable = [c for c in _imputable_cols if train_df.schema[c].is_float()]
            if _float_imputable:
                bp = bp.with_columns(*[pl.when(pl.col(_c).is_nan()).then(None).otherwise(pl.col(_c)).alias(_c) for _c in _float_imputable])

            # ``config.imputer_strategy`` has been canonicalised by the
            # validator to one of {mean, median, mode}. mean / median map
            # directly to polars-ds's ``Blueprint.impute``; ``mode`` does NOT
            # -- polars-ds's mode-impute does ``pl.col(...).mode().list.first()``
            # which raises ``expected List data type ... got Float32`` on polars
            # versions where ``Series.mode()`` returns a flat (non-List) result.
            # Compute the per-column mode natively and fill_null with the scalar
            # so the broken polars-ds code path is bypassed (version-safe).
            # A column with NO finite value (all-NULL / all-NaN degenerate column, e.g. the fuzz axis'
            # ``num_null``) has no statistic to impute from -- polars-ds computes ``median``/``mean`` as
            # None and ``fill_null(None)`` raises "must specify either a fill value or strategy". Exclude
            # such columns from the strategy-based impute step below.
            def _has_finite_value(_name: str) -> bool:
                _s = train_df.get_column(_name)
                if train_df.schema[_name].is_float():
                    return _s.drop_nulls().drop_nans().len() > 0
                return _s.drop_nulls().len() > 0
            _impute_targets = [c for c in _imputable_cols if _has_finite_value(c)]
            # An excluded (all-NULL/all-NaN) column must NOT be left null on the assumption that a
            # downstream "constant-column dropper" removes it -- that dropper is the user-controlled
            # ``remove_constant_columns_cfg`` flag (default True, but a real user CAN set it False to
            # keep a fixed column layout across train/val/test), so a null column previously survived
            # all the way to strict NaN-intolerant models (PytorchLightningEstimator's own guard
            # correctly refuses NaN input rather than silently producing all-NaN predictions -- fuzz
            # surfaced this on models=[linear,mlp] + recurrent_model=lstm + remove_constant_columns=False,
            # 2026-07-06). Fill with 0.0, mirroring the ``SimpleImputer(keep_empty_features=True)``
            # convention already used for the sklearn imputer path (``_setup_helpers.py`` /
            # ``_predict_guards.py``): the column stays FINITE (uninformative, not missing) regardless
            # of whether the constant-column dropper is enabled.
            _all_null_targets = [c for c in _imputable_cols if c not in _impute_targets]
            if _all_null_targets:
                bp = bp.with_columns(*[pl.col(c).fill_null(0.0) for c in _all_null_targets])
            if config.imputer_strategy == "mode":
                _mode_exprs = []
                for _c in _impute_targets:
                    _base = pl.col(_c).drop_nulls()
                    if train_df.schema[_c].is_float():
                        _base = _base.drop_nans()
                    _mv = train_df.select(_base.mode().sort().first().alias(_c)).item()
                    if _mv is not None:
                        _mode_exprs.append(pl.col(_c).fill_null(_mv).alias(_c))
                if _mode_exprs:
                    bp = bp.with_columns(*_mode_exprs)
            elif _impute_targets:
                bp = bp.impute(_impute_targets, method=config.imputer_strategy)
            if verbose:
                logger.info(
                    "  Imputer wired: strategy=%s on %d numeric columns",
                    config.imputer_strategy, len(_imputable_cols),
                )
        elif verbose:
            logger.info("  No numeric columns to impute; skipping imputer step")

    # Add scaling. polars-ds's ``robust_scale`` divides by ``q_high - q_low``
    # which collapses to zero (or NaN) for all-constant or all-null
    # columns, producing ``ComputeError: division by zero`` /
    # ``quantile(None)`` deep inside the polars-ds C++ kernel
    # (observed in fuzz seeds). The historical workaround
    # forced ``remove_constant_columns=True`` from the fuzz harness,
    # which masked the bug; the proper fix is to compute the
    # scalable-column subset in Python and pass it explicitly so
    # polars-ds never sees a zero-IQR column. The same risk applies
    # to ``standard`` / ``min_max`` scalers (zero variance / zero
    # range), so the filter is universal.
    if config.scaler_name:
        bp = _apply_safe_scaler(
            bp,
            train_df,
            scaler_name=config.scaler_name,
            q_low=config.robust_q_low,
            q_high=config.robust_q_high,
            verbose=verbose,
        )

    # Pre-compute the list of cat-like columns that SHOULD be encoded
    # (text/embedding features excluded). We pass this list explicitly
    # when ``excluded`` is non-empty so polars-ds never touches the
    # reserved columns. When ``excluded`` is empty, keep the historical
    # ``cols=None`` (auto-detect) behaviour for byte-for-byte
    # compatibility with the legacy fastpath behaviour.
    def _encodable_cols() -> List[str]:
        out: List[str] = []
        for name, dtype in train_df.schema.items():
            if name in excluded:
                continue
            # Mirror polars-ds's auto-detection for string-like dtypes.
            if dtype == pl.Utf8 or dtype == pl.String or dtype == pl.Categorical or dtype == pl.Boolean or (hasattr(pl, "Enum") and isinstance(dtype, pl.Enum)):
                out.append(name)
        return out

    # Add categorical encoding (skip when downstream models handle categoricals natively)
    if config.skip_categorical_encoding:
        if verbose:
            logger.info("  Skipping categorical encoding (downstream models handle categoricals natively)")
    elif config.categorical_encoding in ("ordinal", "onehot"):
        # Pre-check: polars-ds raises "Provided columns either do not exist or are not
        # string/categorical/enum types" when no cat-like columns exist. Skip the
        # encoding step in that case rather than letting polars-ds crash.
        candidate_cols = _encodable_cols()
        if not candidate_cols:
            if verbose:
                logger.info("  No string/categorical/enum columns to encode; skipping categorical encoding step")
        else:
            # polars-ds's ordinal_encode / one_hot_encode reject ``pl.Enum`` columns
            # ("not string/categorical types") -- it recognises only String / Categorical.
            # Cast any Enum candidate to Categorical first so Enum-typed inputs
            # (input_type='polars_enum') encode instead of crashing the pipeline build.
            _enum_cls = getattr(pl, "Enum", None)
            _enum_to_cast = [name for name in candidate_cols if _enum_cls is not None and isinstance(train_df.schema.get(name), _enum_cls)]
            if _enum_to_cast:
                bp = bp.with_columns(*[pl.col(_c).cast(pl.Categorical) for _c in _enum_to_cast])
            cols_arg = candidate_cols if excluded else None
            if config.categorical_encoding == "ordinal":
                bp = bp.ordinal_encode(cols=cols_arg, null_value=-1, unknown_value=-2)
            else:
                bp = bp.one_hot_encode(cols=cols_arg, drop_first=False, drop_cols=True)
    # Add more encoding methods as needed

    # Convert int to float32 for better compatibility.
    # Skip already-narrow Int8/Int16 columns (typically datetime decomposition
    # outputs: day/weekday/month/hour all fit Int8); widening them to float32
    # quadruples memory for zero downstream benefit since tree models accept
    # int8 directly. We cast only Int32/Int64/UInt32/UInt64 to f32. fix audit
    # row FE-L-3.
    try:
        _narrow_int_dtypes = {pl.Int8, pl.Int16, pl.UInt8, pl.UInt16}
        _wide_int_cols = [name for name, dtype in train_df.schema.items() if dtype.is_integer() and dtype not in _narrow_int_dtypes]
        if _wide_int_cols:
            _cast_exprs = [pl.col(c).cast(pl.Float32) for c in _wide_int_cols]
            # polars-ds Blueprint.with_columns is ``*exprs`` style; unpack.
            bp = bp.with_columns(*_cast_exprs)
    except Exception as _exc:  # pragma: no cover - polars-ds API drift fallback
        # If the per-column path errors (older polars-ds without with_columns,
        # schema dtype detection failure, ...) fall back to the legacy
        # whole-frame cast so we never silently emit raw int to consumers that
        # historically expected float.
        if verbose:
            logger.warning(
                "Narrow-int-aware int_to_float gating failed (%s); falling back to legacy int_to_float(f32=True).",
                _exc,
            )
        bp = bp.int_to_float(f32=True)

    # Materialize the pipeline
    pipeline = bp.materialize()
    maybe_clean_ram_adaptive()

    if verbose:
        bp_elapsed = timer() - t0_bp
        logger.info("  Polars-ds pipeline created -- scaler=%s, encoding=%s, %.1fs", config.scaler_name or 'none', config.categorical_encoding or 'none', bp_elapsed)
        log_ram_usage()

    return pipeline


def _warn_on_schema_drift(
    train_schema: "Dict[str, object]",
    other_df: "pl.DataFrame",
    split_name: str,
) -> None:
    """Warn when a non-train split (val / test) schema differs from train.

    Before this check landed: ``pipeline.transform()`` was called on
    val/test with no schema validation. Three failure
    modes silently propagated:
      - Missing column: polars-ds pipeline errored deep inside with an
        opaque traceback (column lookup failure).
      - Extra column: silently kept or dropped depending on pipeline
        internals; downstream shape mismatch at model.fit/predict.
      - Dtype change (e.g. train had pl.Int32, val has pl.Int64):
        silent coercion that may introduce NaN on bounds overflow
        or downcast truncation.

    This helper emits one WARN per failing category with the column
    names and diff. Does NOT raise -- some callers intentionally drop
    derived columns that the pipeline reconstructs. The WARN lets
    operators trace opaque downstream errors back here.
    """
    try:
        other_schema = dict(other_df.schema)
    except Exception:
        return  # not a polars frame or schema unavailable -- skip silently

    train_cols = set(train_schema.keys())
    other_cols = set(other_schema.keys())

    missing_in_other = train_cols - other_cols
    extra_in_other = other_cols - train_cols

    if missing_in_other:
        logger.warning(
            "Schema drift: %s split is missing %d column(s) that were "
            "present at fit time: %s. Polars-ds pipeline.transform() will "
            "likely raise deep inside with an opaque error; the column "
            "list above is the upstream cause.",
            split_name, len(missing_in_other), sorted(missing_in_other),
        )

    if extra_in_other:
        logger.warning(
            "Schema drift: %s split has %d extra column(s) not seen at "
            "fit time: %s. The pipeline may silently drop or keep them "
            "depending on step internals; downstream model.fit/predict "
            "shape mismatches usually trace back here.",
            split_name, len(extra_in_other), sorted(extra_in_other),
        )

    # Compare dtypes via str() instead of native ``!=``. The native
    # ``!=`` was triggering Series.equals via pandas Index.__eq__
    # machinery (some dtype values had pandas-like __ne__) costing
    # ~270ms per call x 6 calls = 1.6s wasted on the observed prod
    # frame. str() forces a plain Python string compare (microseconds)
    # and matches
    # the existing ``str(train_schema[col]), str(other_schema[col])``
    # used for the WARN message anyway. Semantic difference: two
    # otherwise-equal Enum dtypes with the SAME underlying category
    # set (but different memory layout) now compare equal via str
    # representation, which is what the user-visible warn was already
    # asserting.
    dtype_mismatches = []
    for col in train_cols & other_cols:
        train_dt_str = str(train_schema[col])
        other_dt_str = str(other_schema[col])
        if train_dt_str != other_dt_str:
            dtype_mismatches.append((col, train_dt_str, other_dt_str))
    if dtype_mismatches:
        logger.warning(
            "Schema drift: %s split has %d column(s) with dtype different "
            "from fit-time: %s. Polars will silently coerce at transform "
            "time, potentially introducing NaN on bounds overflow or "
            "truncating precision. Align upstream extraction to match "
            "train dtypes.",
            split_name, len(dtype_mismatches), dtype_mismatches,
        )


__all__ = [
    "prepare_df_for_catboost",
    "prepare_dfs_for_catboost_joint",
    "create_polarsds_pipeline",
    "fit_and_transform_pipeline",
    "apply_preprocessing_extensions",
]


# ----------------------------------------------------------------------
# Sibling-module re-exports. Big functions live in
# ``_pipeline_extensions.py`` and ``_pipeline_fit_transform.py`` so this
# file stays below the 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._pipeline_extensions import (  # noqa: E402,F401
    _apply_pysr_fe, apply_preprocessing_extensions, sparse_df_from_spmatrix,
)
from ._pipeline_fit_transform import (  # noqa: E402,F401
    fit_and_transform_pipeline,
)

# Public package surface for the pre-pipeline application + cache helpers. These were historically imported as ``from mlframe.training._pipeline_helpers import ...`` / ``_pipeline_cache``; the package re-exports them so importers resolve from the documented package path.
from ._pipeline_helpers import (  # noqa: E402,F401
    _test_df_is_raw_pipeline_input,
    _prepare_test_split,
    _extract_feature_selector,
    _is_fitted,
    _is_stale_fit_state_value_error,
    _multilabel_target_to_1d_for_supervised_encoders,
    _passthrough_cols_fit_transform,
    _apply_pre_pipeline_transforms,
)
from ._pipeline_cache import (  # noqa: E402,F401
    _PRE_PIPELINE_CACHE,
    _PRE_PIPELINE_CACHE_LOCK,
    _PRE_PIPELINE_CACHE_MAX,
    _UncachableSentinel,
    _fresh_uncachable,
    _content_fingerprint_for_cache,
    _full_x_content_hash,
    _full_target_content_hash,
    _pipeline_signature_for_cache,
    _pre_pipeline_cache_key,
    _pre_pipeline_cache_get,
    _pre_pipeline_cache_set,
    _pre_pipeline_cache_clear,
)
