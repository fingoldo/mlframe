"""
Pipeline functions for mlframe training.

Handles Polars-ds and sklearn pipeline creation, fitting, and transformation.
"""

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging
from timeit import default_timer as timer
import subprocess

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import polars as pl
import polars.selectors as cs
from typing import Union, Optional, List, Tuple
from collections import Counter
from pyutilz.system import clean_ram
from .utils import maybe_clean_ram_adaptive
from pyutilz.pandaslib import ensure_dataframe_float32_convertability

from .utils import log_ram_usage
from .configs import PreprocessingBackendConfig, PreprocessingExtensionsConfig
from .strategies import PANDAS_CATEGORICAL_DTYPES, get_polars_cat_columns


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
    "Normalizer_l2": lambda: __import__("sklearn.preprocessing", fromlist=["Normalizer"]).Normalizer(norm="l2"),
}


def _build_extension_steps(config: PreprocessingExtensionsConfig, n_features: int, random_state: int = 42) -> list:
    """Assemble the ordered list of (name, transformer) pairs for the extensions config.

    Raises ImportError for missing optional deps (UMAP) with an install hint.
    Raises ValueError when PolynomialFeatures would exceed memory_safety_max_features.
    """
    from sklearn.preprocessing import Binarizer, KBinsDiscretizer, PolynomialFeatures
    steps = []
    # NaN-imputation guard 2026-04-27 (batch 3): KBinsDiscretizer,
    # PolynomialFeatures, RBFSampler, Nystroem, and most sklearn
    # decompositions (PCA, TruncatedSVD, FastICA, ...) reject NaN at
    # fit time with ``ValueError: Input X contains NaN``. The
    # mlframe upstream preprocessing handles NaN for the GBDT
    # backends (CB / HGB / XGB) which tolerate NaN natively, so
    # numeric NaN can survive into ``apply_preprocessing_extensions``
    # untouched (fuzz seed=2024 c0040 -- n=1000 polars_utf8 with
    # inject_inf_nan + prep_ext_kbins=5). Prepend a SimpleImputer so
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
        steps.append(("imputer", SimpleImputer(strategy="median")))
    if config.scaler is not None:
        steps.append(("scaler", _SCALER_FACTORIES[config.scaler]()))
    if config.binarization_threshold is not None:
        steps.append(("binarizer", Binarizer(threshold=config.binarization_threshold)))
    if config.kbins is not None:
        steps.append(("kbins", KBinsDiscretizer(n_bins=config.kbins, encode=config.kbins_encode, strategy="quantile", quantile_method="averaged_inverted_cdf")))
    if config.polynomial_degree is not None:
        projected = n_features ** config.polynomial_degree
        if projected > config.memory_safety_max_features:
            raise ValueError(
                f"PolynomialFeatures(degree={config.polynomial_degree}) on {n_features} features "
                f"would produce up to {projected} columns, above memory_safety_max_features="
                f"{config.memory_safety_max_features}. Add dim_reducer='PCA' first or raise the guard."
            )
        steps.append(("poly", PolynomialFeatures(
            degree=config.polynomial_degree,
            interaction_only=config.polynomial_interaction_only,
            include_bias=False,
        )))
    if config.nonlinear_features is not None:
        from sklearn.kernel_approximation import RBFSampler, Nystroem, AdditiveChi2Sampler, SkewedChi2Sampler
        _nl = {"RBFSampler": RBFSampler, "Nystroem": Nystroem,
               "AdditiveChi2Sampler": AdditiveChi2Sampler, "SkewedChi2Sampler": SkewedChi2Sampler}
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


def _apply_pysr_fe(
    *,
    train_df: "pd.DataFrame",
    val_df,
    test_df,
    y_train,
    config: "PreprocessingExtensionsConfig",
    verbose: int = 1,
) -> list:
    """Run PySR symbolic regression on train, apply top equations to all
    splits, and add predictions as new numeric columns in-place. Returns
    the list of added column names.

    Gracefully skips on ImportError (Julia/PySR not installed) or
    when ``y_train`` is None (target not available at this stage).
    """
    if y_train is None:
        return []
    try:
        from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering
    except (ImportError, OSError, subprocess.CalledProcessError):
        if verbose:
            import logging
            logging.getLogger(__name__).warning(
                "PySR feature engineering is enabled but the pysr / Julia "
                "runtime is not importable. Skipping."
            )
        return []
    import numpy as np

    pysr_params = getattr(config, "pysr_params", None) or {}
    # Merge sensible defaults so the user doesn't need to specify
    # every knob. Low budget defaults keep the extension fast.
    defaults = dict(
        niterations=5,
        populations=5,
        population_size=20,
        tournament_selection_n=8,
        maxdepth=4,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "exp", "log_abs"],
        procs=1,
        verbosity=0,
    )
    defaults.update(pysr_params)
    # Use a shallow copy so underlying YAML/dict config isn't mutated.
    merged_params = dict(defaults)

    top_k = min(5, merged_params.get("population_size", 20) // 2)
    sample_n = min(len(train_df), 20000)
    temp_target_col = "_pysr_y_"

    # Inject y_train as a temporary column (bruteforce expects target
    # as a column in the DataFrame). Remove after PySR fit.
    existing_y = train_df.columns.tolist()
    while temp_target_col in existing_y:
        temp_target_col = "_" + temp_target_col
    train_df[temp_target_col] = np.asarray(y_train).ravel()

    try:
        model = run_pysr_feature_engineering(
            df=train_df,
            target_col=temp_target_col,
            sample_size=sample_n,
            encode_categoricals=False,
            verbose=0,
            pysr_params_override=merged_params,
        )
    except Exception:
        if verbose:
            logging.getLogger(__name__).warning(
                "PySR fit failed; skipping symbolic feature engineering.",
                exc_info=True,
            )
        train_df.drop(columns=[temp_target_col], inplace=True, errors="ignore")
        return []
    finally:
        train_df.drop(columns=[temp_target_col], inplace=True, errors="ignore")

    # Apply top-K equations (by score)
    eq_df = model.equations_
    if eq_df is None or len(eq_df) == 0:
        return []
    eq_df = eq_df.sort_values(["score"], ascending=[False]).head(top_k)

    new_cols = []
    for idx in eq_df.index:
        try:
            col_name = f"pysr_eq{idx}"
            if col_name in train_df.columns:
                continue
            train_df[col_name] = np.asarray(
                model.predict(train_df, index=idx), dtype=np.float32)
            if val_df is not None:
                val_df[col_name] = np.asarray(
                    model.predict(val_df, index=idx), dtype=np.float32)
            if test_df is not None:
                test_df[col_name] = np.asarray(
                    model.predict(test_df, index=idx), dtype=np.float32)
            new_cols.append(col_name)
        except Exception:
            continue
    return new_cols


def apply_preprocessing_extensions(
    train_df,
    val_df,
    test_df,
    config: Optional[PreprocessingExtensionsConfig],
    verbose: int = 1,
    y_train=None,
):
    """Apply shared sklearn-based extensions to train/val/test after the Polars-ds pipeline.

    Returns (train, val, test, fitted_pipeline_or_None). Fastpath: when ``config``
    is None OR has zero active stages, returns inputs untouched with None pipeline.
    """
    if config is None:
        return train_df, val_df, test_df, None
    # Polars input -> convert to pandas (extensions use sklearn; mixing with
    # the polars-native fastpath would defeat the point if user opted in).
    def _to_pandas(df):
        if df is None:
            return None
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        return df

    train = _to_pandas(train_df)
    val = _to_pandas(val_df)
    test = _to_pandas(test_df)
    if train is None:
        return train_df, val_df, test_df, None

    # PySR symbolic regression (step 0). Runs BEFORE TF-IDF and the sklearn
    # pipeline so that discovered equation features benefit from downstream
    # scaling, polynomial expansion, etc.
    if getattr(config, "pysr_enabled", False):
        _pysr_cols = _apply_pysr_fe(
            train_df=train, val_df=val, test_df=test,
            y_train=y_train,
            config=config,
            verbose=verbose,
        )
        # _pysr_cols are the names of the new columns; already added to
        # train/val/test in-place inside _apply_pysr_fe.

    # TF-IDF preflight: vectorize declared text columns and replace them with
    # numeric features before downstream sklearn steps (which expect numeric).
    #
    # Column-parity invariant (2026-04-19 round-9 probe): train, val, and
    # test MUST emerge from TF-IDF with the same column set. Pre-fix the
    # code only TF-IDF-expanded train and left val/test untouched when
    # the text column happened to be missing from val/test (sparse splits,
    # user typo in ``tfidf_columns`` matching only train's schema). Then
    # the downstream sklearn Pipeline, fit on train with e.g. 5050
    # columns, tried ``pipe.transform(val_with_50_cols)`` and raised a
    # shape-mismatch error that traced back to the scaler -- not TF-IDF.
    # Now: if a tfidf_column is missing from val/test, we skip it on
    # train too (WARN with the consequence) so all three splits stay
    # aligned. If it's a user typo, the typo WARN fires instead.
    tfidf_pipes = {}
    if config.tfidf_columns:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Precompute where each tfidf column lives.
        train_has = set(train.columns)
        val_has = set(val.columns) if val is not None else None
        test_has = set(test.columns) if test is not None else None
        usable_cols, skipped_typo, skipped_split_mismatch = [], [], []
        for col in config.tfidf_columns:
            if col not in train_has:
                skipped_typo.append(col)
                continue
            # If val/test exist and one of them lacks the column, we can't
            # produce aligned TF-IDF features across splits. Skip the col
            # entirely rather than silently diverge.
            if val is not None and val_has is not None and col not in val_has:
                skipped_split_mismatch.append((col, "val"))
                continue
            if test is not None and test_has is not None and col not in test_has:
                skipped_split_mismatch.append((col, "test"))
                continue
            usable_cols.append(col)

        if skipped_typo:
            logger.warning(
                "TF-IDF: %d column(s) listed in config.tfidf_columns not found "
                "in train DataFrame: %s. Possibly a typo in config vs the "
                "upstream feature-extraction schema.",
                len(skipped_typo), skipped_typo,
            )
        if skipped_split_mismatch:
            logger.warning(
                "TF-IDF: %d column(s) present in train but missing from a "
                "non-train split (val/test) -- skipping entirely to keep "
                "splits column-aligned for downstream sklearn transforms: "
                "%s. If these columns should be universally present, fix "
                "the upstream split so all three frames share the schema.",
                len(skipped_split_mismatch), skipped_split_mismatch,
            )

        for col in usable_cols:
            vec = TfidfVectorizer(
                max_features=config.tfidf_max_features,
                ngram_range=tuple(config.tfidf_ngram_range),
            )
            train_text = train[col].fillna("").astype(str).values
            tfidf_train = vec.fit_transform(train_text)
            tfidf_pipes[col] = vec
            new_cols = [f"{col}__tfidf_{i}" for i in range(tfidf_train.shape[1])]
            tfidf_train_df = pd.DataFrame(tfidf_train.toarray(), columns=new_cols, index=train.index)
            train = train.drop(columns=[col]).join(tfidf_train_df)
            for split_name, split_df in (("val", val), ("test", test)):
                if split_df is not None:
                    # Column presence was verified above in `usable_cols`
                    # filtering; this branch is now guaranteed safe.
                    text_arr = split_df[col].fillna("").astype(str).values
                    tfidf_arr = vec.transform(text_arr).toarray()
                    new_split_df = pd.DataFrame(tfidf_arr, columns=new_cols, index=split_df.index)
                    if split_name == "val":
                        val = split_df.drop(columns=[col]).join(new_split_df)
                    else:
                        test = split_df.drop(columns=[col]).join(new_split_df)

    n_features = train.shape[1]
    steps = _build_extension_steps(config, n_features=n_features)
    if not steps:
        if tfidf_pipes:
            # TF-IDF was applied but no other steps -- return TF-IDF-augmented frames.
            return train, val, test, tfidf_pipes
        return train_df, val_df, test_df, None

    from sklearn.pipeline import Pipeline as SkPipeline
    pipe = SkPipeline(steps=steps)
    t0 = timer()
    # LDA requires `y` during fit; forward y_train when provided.
    if y_train is not None:
        train_arr = pipe.fit_transform(train, y_train)
    else:
        train_arr = pipe.fit_transform(train)
    val_arr = pipe.transform(val) if val is not None and len(val) > 0 else None
    test_arr = pipe.transform(test) if test is not None and len(test) > 0 else None

    def _to_df(arr, template):
        if arr is None:
            return None
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        cols = [f"ext_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols, index=getattr(template, "index", None))

    train_out = _to_df(train_arr, train)
    val_out = _to_df(val_arr, val)
    test_out = _to_df(test_arr, test)
    # Two-level verbosity: caller-side ``verbose`` (function-level kill switch)
    # AND ``config.verbose_logging`` (per-config opt-out for this stage when
    # batching many folds whose output would drown the log). WARN paths above
    # (skipped_typo / split_mismatch) are intentionally NOT gated -- those are
    # configuration errors the user must see.
    if verbose and config.verbose_logging:
        elapsed = timer() - t0
        logger.info(
            "Applied preprocessing extensions (%d stages) -- train %s, %.2fs",
            len(steps), train_out.shape, elapsed,
        )
    return train_out, val_out, test_out, pipe


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
    (fuzz c0008 / c0116 / 2026-04-26).
    """
    scalable: List[str] = []
    skipped_reasons: dict = {}

    numeric_cols = [name for name, dtype in train_df.schema.items() if dtype.is_numeric()]
    if not numeric_cols:
        return scalable

    # 2026-05-08 perf: batch all per-col stats into ONE collect via
    # lazy select. Previous loop did 3 collects per col (n_non_null
    # check + 2 stat computations for the chosen ``method``); on c0031
    # (~15 numeric cols, method=robust) that was ~45 PyLazyFrame.collect
    # calls = ~0.4s wasted. Batched -> 1 collect total.
    has_drop_nans = all(hasattr(train_df[c], "drop_nans") for c in numeric_cols)
    select_exprs = []
    for c in numeric_cols:
        if has_drop_nans:
            n_expr = (
                pl.col(c).drop_nulls().drop_nans()
                .filter(pl.col(c).drop_nulls().drop_nans().is_finite())
                .len().alias(f"__n__{c}")
            )
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
                    nfin = col.drop_nulls().drop_nans()
                    n_non_null = nfin.filter(nfin.is_finite()).len()
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
            (discovered 2026-04-23 on fuzz c0085/c0049 -> CB Pool build failed with
            ``Invalid type for text_feature ... =187.0 : text_features must have
            string type`` because the text column arrived as float32 ordinal
            codes). When this set is non-empty, pass an explicit ``cols=`` list
            to the encoder that excludes those columns.

    Returns:
        Materialized PdsPipeline or None if polars-ds not available
    """
    try:
        from polars_ds.pipeline import Pipeline as PdsPipeline, Blueprint as PdsBlueprint
    except Exception as e:
        logger.warning(f"Could not import polars-ds: {e}")
        return None

    if verbose:
        logger.info(f"Creating Polars-ds pipeline...")

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
        _imputable_cols = [
            name for name, dtype in train_df.schema.items()
            if dtype.is_numeric() and not dtype == pl.Boolean
        ]
        if _imputable_cols:
            # ``config.imputer_strategy`` has been canonicalised by the
            # validator to one of {mean, median, mode} so it maps
            # directly to polars-ds's ``Blueprint.impute`` API.
            bp = bp.impute(_imputable_cols, method=config.imputer_strategy)
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
    # (fuzz c0008 / c0116 / 2026-04-26). The historical workaround
    # forced ``remove_constant_columns=True`` from the fuzz harness,
    # which masked the bug; the proper fix is to compute the
    # scalable-column subset in Python and pass it explicitly so
    # polars-ds never sees a zero-IQR column. The same risk applies
    # to ``standard`` / ``min_max`` scalers (zero variance / zero
    # range), so the filter is universal.
    if config.scaler_name:
        _scalable_numeric_cols = _select_scalable_numeric_columns(
            train_df,
            method="robust" if config.scaler_name == "robust" else config.scaler_name,
            q_low=config.robust_q_low,
            q_high=config.robust_q_high,
            verbose=verbose,
        )
        if _scalable_numeric_cols:
            if config.scaler_name == "robust":
                bp = bp.robust_scale(_scalable_numeric_cols, q_low=config.robust_q_low, q_high=config.robust_q_high)
            else:
                bp = bp.scale(_scalable_numeric_cols, method=config.scaler_name)
        elif verbose:
            logger.info(
                "  No numeric columns survived the zero-spread / all-null "
                "filter -- skipping scaler entirely."
            )

    # Pre-compute the list of cat-like columns that SHOULD be encoded
    # (text/embedding features excluded). We pass this list explicitly
    # when ``excluded`` is non-empty so polars-ds never touches the
    # reserved columns. When ``excluded`` is empty, keep the historical
    # ``cols=None`` (auto-detect) behaviour for byte-for-byte
    # compatibility with the pre-2026-04-23 fastpath.
    def _encodable_cols() -> List[str]:
        out: List[str] = []
        for name, dtype in train_df.schema.items():
            if name in excluded:
                continue
            # Mirror polars-ds's auto-detection for string-like dtypes.
            if (
                dtype == pl.Utf8
                or dtype == pl.String
                or dtype == pl.Categorical
                or dtype == pl.Boolean
                or (hasattr(pl, "Enum") and isinstance(dtype, pl.Enum))
            ):
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
            cols_arg = candidate_cols if excluded else None
            if config.categorical_encoding == "ordinal":
                bp = bp.ordinal_encode(cols=cols_arg, null_value=-1, unknown_value=-2)
            else:
                bp = bp.one_hot_encode(cols=cols_arg, drop_first=False, drop_cols=True)
    # Add more encoding methods as needed

    # Convert int to float32 for better compatibility
    bp = bp.int_to_float(f32=True)

    # Materialize the pipeline
    pipeline = bp.materialize()
    maybe_clean_ram_adaptive()

    if verbose:
        bp_elapsed = timer() - t0_bp
        logger.info(f"  Polars-ds pipeline created -- scaler={config.scaler_name or 'none'}, encoding={config.categorical_encoding or 'none'}, {bp_elapsed:.1f}s")
        log_ram_usage()

    return pipeline


def _warn_on_schema_drift(
    train_schema: "Dict[str, object]",
    other_df: "pl.DataFrame",
    split_name: str,
) -> None:
    """Warn when a non-train split (val / test) schema differs from train.

    Before this check (2026-04-19 probe finding): ``pipeline.transform()``
    was called on val/test with no schema validation. Three failure
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

    # 2026-05-08 perf: compare dtypes via str() instead of native ``!=``.
    # On c0034 the native ``!=`` was triggering Series.equals via
    # pandas Index.__eq__ machinery (some dtype values had pandas-like
    # __ne__) costing ~270ms per call x 6 calls = 1.6s wasted. str()
    # forces a plain Python string compare (microseconds) and matches
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


def fit_and_transform_pipeline(
    train_df: Union[pd.DataFrame, pl.DataFrame],
    val_df: Optional[Union[pd.DataFrame, pl.DataFrame]],
    test_df: Optional[Union[pd.DataFrame, pl.DataFrame]],
    config: PreprocessingBackendConfig,
    ensure_float32: bool = True,
    verbose: int = 1,
    text_features: Optional[List[str]] = None,
    embedding_features: Optional[List[str]] = None,
) -> Tuple[Union[pd.DataFrame, pl.DataFrame], Optional[Union[pd.DataFrame, pl.DataFrame]], Optional[Union[pd.DataFrame, pl.DataFrame]], object, List[str]]:
    """
    Fit and apply a data pipeline to train/val/test splits.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame (optional)
        test_df: Test DataFrame (optional)
        config: Pipeline configuration
        ensure_float32: Whether to ensure float32 dtypes
        verbose: Verbosity level
        text_features: Columns to exclude from encoding/scaling (free-text for CatBoost)
        embedding_features: Columns to exclude from encoding/scaling (list-of-float vectors)

    Returns:
        Tuple of (train_df, val_df, test_df, pipeline, cat_features)
    """
    # Columns that must be excluded from encoding (they're not categoricals)
    _exclude_from_encoding = set(text_features or []) | set(embedding_features or [])
    pipeline = None
    cat_features = []

    # 2026-04-24: datetime column decomposition moved to
    # ``train_mlframe_models_suite`` (core.py) BEFORE the pre-pipeline
    # polars-clone point so the clone inherits the numeric decomposition.
    # Calling it here too would be a no-op -- the caller has already
    # decomposed any datetime columns by the time we run.

    # Handle Polars DataFrames with polars-ds
    if isinstance(train_df, pl.DataFrame) and config.prefer_polarsds:
        # Detect cat_features from the ORIGINAL schema before the pipeline possibly
        # ordinal/one-hot-encodes them to numeric (which would erase their categorical dtype).
        _orig_cat_features = [
            c for c in get_polars_cat_columns(train_df) if c not in _exclude_from_encoding
        ]
        pipeline = create_polarsds_pipeline(
            train_df, config, verbose=verbose,
            exclude_from_encoding=_exclude_from_encoding,
        )

        if pipeline is not None:
            if verbose:
                logger.info(f"Applying Polars-ds pipeline...")

            # Capture train schema BEFORE the fit-time transform so we
            # can compare val/test schemas against it below (2026-04-19
            # schema-drift probe finding: pipeline.transform(val_df) was
            # called without any schema validation; missing/extra cols
            # or dtype mismatches silently propagated either to a
            # downstream sklearn shape-error or garbage output).
            _train_schema_snapshot = dict(train_df.schema)

            t0_transform = timer()
            # Transform all splits and ensure float32 dtypes
            train_df = pipeline.transform(train_df)
            if ensure_float32:
                train_df = ensure_dataframe_float32_convertability(train_df)

            if val_df is not None and len(val_df) > 0:
                _warn_on_schema_drift(_train_schema_snapshot, val_df, "val")
                val_df = pipeline.transform(val_df)
                if ensure_float32:
                    val_df = ensure_dataframe_float32_convertability(val_df)

            if test_df is not None and len(test_df) > 0:
                _warn_on_schema_drift(_train_schema_snapshot, test_df, "test")
                test_df = pipeline.transform(test_df)
                if ensure_float32:
                    test_df = ensure_dataframe_float32_convertability(test_df)

            if verbose:
                transform_elapsed = timer() - t0_transform
                logger.info(f"  Polars-ds transform done -- train: {train_df.shape[0]:_}x{train_df.shape[1]}, {transform_elapsed:.1f}s")
                logger.info(f"  train_df dtypes after pipeline: {Counter(train_df.dtypes)}")

        # Detect categorical features from schema (works whether pipeline succeeded or not)
        # This ensures cat_features is populated even if polars-ds is not available.
        # Prefer the ORIGINAL cat columns (captured before transform) -- after ordinal/onehot
        # encoding they're no longer Categorical/Utf8 in the transformed frame.
        post_cat = [c for c in get_polars_cat_columns(train_df) if c not in _exclude_from_encoding]
        cat_features = _orig_cat_features if _orig_cat_features else post_cat

    # Handle Polars DataFrames without polars-ds pipeline - just detect cat_features
    elif isinstance(train_df, pl.DataFrame) and not config.prefer_polarsds:
        # Detect categorical features from schema (no transformation, just detection)
        cat_features = [c for c in get_polars_cat_columns(train_df) if c not in _exclude_from_encoding]
        if verbose and cat_features:
            logger.info(f"Detected {len(cat_features)} categorical features from Polars schema: {cat_features}")

    # Handle pandas DataFrames with sklearn-style pipeline
    elif isinstance(train_df, pd.DataFrame):
        # Identify categorical features (exclude text/embedding columns)
        cat_features = [
            col for col in train_df.columns
            if train_df[col].dtype.name in PANDAS_CATEGORICAL_DTYPES
            and col not in _exclude_from_encoding
        ]

        # Apply categorical encoding if specified (for models that don't support categorical natively)
        if cat_features and config.categorical_encoding in ["ordinal", "onehot"] and not config.skip_categorical_encoding:
            if verbose:
                logger.info(f"Applying {config.categorical_encoding} encoding to {len(cat_features)} categorical features: {cat_features}")

            t0_encode = timer()
            from category_encoders import OrdinalEncoder, OneHotEncoder

            # Create appropriate encoder
            if config.categorical_encoding == "ordinal":
                encoder = OrdinalEncoder(cols=cat_features, handle_unknown="value", handle_missing="value")
            else:  # onehot
                encoder = OneHotEncoder(cols=cat_features, use_cat_names=True, drop_invariant=False)

            # Fit on train and transform all splits
            train_df = encoder.fit_transform(train_df)
            if val_df is not None and len(val_df) > 0:
                val_df = encoder.transform(val_df)
            if test_df is not None and len(test_df) > 0:
                test_df = encoder.transform(test_df)

            pipeline = encoder  # Store encoder as pipeline

            if verbose:
                encode_elapsed = timer() - t0_encode
                logger.info(f"  Encoding done -- train: {train_df.shape[0]:_}x{train_df.shape[1]}, {encode_elapsed:.1f}s")

            # After encoding, cat_features are no longer categorical (they're numeric)
            cat_features = []

        # Prepare categorical features for CatBoost (if not already encoded)
        elif cat_features:
            if verbose:
                logger.info(f"Preparing {len(cat_features)} categorical features for CatBoost...")

            for df in [train_df, val_df, test_df]:
                if df is not None and len(df) > 0:
                    prepare_df_for_catboost(df, cat_features)

    # Clean up empty validation/test sets
    if val_df is not None and len(val_df) == 0:
        val_df = None

    if test_df is not None and len(test_df) == 0:
        test_df = None

    # gc.collect on a 20-30GB heap with Arrow buffers can take a full minute
    # after a just-freed raw DataFrame. Previously this was the mystery "PHASE 3
    # black box" -- a minute passed between "Detected N categorical features"
    # and "Done. RAM usage:" with no log. Now wrapped so we can see it and
    # reason about disabling for polars-fastpath runs.
    t0_gc = timer()
    maybe_clean_ram_adaptive()
    gc_elapsed = timer() - t0_gc
    if verbose:
        if gc_elapsed > 1.0:
            logger.info("  maybe_clean_ram_adaptive took %.1fs (gc + arena trim)", gc_elapsed)
        log_ram_usage()

    return train_df, val_df, test_df, pipeline, cat_features


__all__ = [
    "prepare_df_for_catboost",
    "create_polarsds_pipeline",
    "fit_and_transform_pipeline",
    "apply_preprocessing_extensions",
]
