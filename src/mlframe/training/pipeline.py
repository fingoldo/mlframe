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
        projected_upper = n_features ** config.polynomial_degree
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


def _apply_pysr_fe(
    *,
    train_df: "pd.DataFrame",
    val_df,
    test_df,
    y_train,
    config: "PreprocessingExtensionsConfig",
    verbose: int = 1,
    out_equations: Optional[Dict[str, str]] = None,
    out_transformer: Optional[list] = None,
) -> list:
    """Run PySR symbolic regression on train, apply top equations to all
    splits, and add predictions as new numeric columns in-place. Returns
    the list of added column names.

    Column naming uses ``pysr__{blake2b(equation_str)[:8]}__{seed}`` so a given symbolic equation always lands on the same column across seeds / runs / processes; two seeds that discover different equations get distinct column names instead of silently overlaying onto a shared ``pysr_eq{idx}`` slot (the prior naming collided across runs).

    When ``out_equations`` is provided, the equation-string -> column-name map is populated for predict-time replay persistence.

    Gracefully skips on ImportError (Julia/PySR not installed). Raises a ``logger.warning`` when ``y_train`` is None (target not threaded through from the calling phase) - silent skip used to mask wiring bugs where ``pysr_enabled=True`` was set but the suite never invoked PySR.
    """
    if y_train is None:
        logger.warning(
            "_apply_pysr_fe: pysr_enabled=True but y_train was not passed in "
            "(caller did not thread the target through). PySR feature "
            "engineering SKIPPED. Pass a 1-D y_train array to enable it."
        )
        return []
    # Set Julia thread-count env BEFORE importing run_pysr_feature_engineering (which boots juliacall).
    # Deferred from module-import time so callers who never trigger PySR don't get their env mutated.
    _maybe_set_pysr_thread_env()
    try:
        from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering
    except (ImportError, OSError, subprocess.CalledProcessError):
        if verbose:
            logger.warning(
                "PySR feature engineering is enabled but the pysr / Julia "
                "runtime is not importable. Skipping."
            )
        return []
    import numpy as np

    pysr_params = getattr(config, "pysr_params", None) or {}
    # Operator preset (minimal / standard / physics) -- standard is the in-suite default. The preset
    # supplies binary_operators, unary_operators, complexity_of_operators, nested_constraints, and
    # extra_sympy_mappings; the raw pysr_params dict can still override any individual key.
    from mlframe.feature_engineering.pysr_operators import get_preset_kwargs
    _preset_name = getattr(config, "pysr_operator_preset", None) or "standard"
    _preset = get_preset_kwargs(_preset_name)

    # Defaults tuned for the in-suite path. Key knobs (rationales in docs/pysr_fe_upgrade_research.md):
    # - Multithreading auto-on via PYTHON_JULIACALL_THREADS + JULIA_NUM_THREADS env set at module import.
    # - batching=True, batch_size=10000 -- each GA iter samples 10K rows from the pool, bounded per-iter
    #   cost regardless of pool size.
    # - precision=32 -- f32 SIMD eval ~2x faster than f64; f16 is broken on Julia 1.10 under turbo=True.
    # - turbo=True, bumper=True -- SIMD + bumper-allocator.
    # - update=False, progress=False -- skip Julia registry probe + Jupyter progress in embedded use.
    # - parsimony=1e-4 + weight_optimize=0.001 -- tuning-guide recommended for ncycles_per_iteration=380.
    # - maxsize=20 + maxdepth=5 -- tabular FE doesn't need 30-node 30-deep trees; smaller = faster eval.
    # - populations capped at min(15, ncpu//3) -- tuning-guide says 3*ncpu but PySR + juliacall on
    #   Windows OOMs at 24 populations on machines with already-committed RAM (e.g. notebook with 10GB
    #   df loaded). Cap conservatively; users with idle workstations can override via pysr_params.
    # - tournament_selection_n=15 -- matches PySR master; weaker tournament loses good equations.
    # - heap_size_hint_in_bytes=256MB -- LOWER means MORE-frequent GC = lower peak memory. Setting hint
    #   too high (RAM/10 ~= 1.6GB on 16GB box) defers GC and triggers Julia "malloc: Not enough space"
    #   SIGABRT under populations>=10 on Windows. 256MB is the smallest value that doesn't cripple GA
    #   throughput per gh discussion #441.
    _ncpu_local = os.cpu_count() or 4
    defaults = dict(
        niterations=400,
        populations=max(4, min(15, _ncpu_local // 3)),
        population_size=33,
        tournament_selection_n=15,
        maxsize=20,
        maxdepth=5,
        parsimony=1e-4,
        weight_optimize=0.001,
        heap_size_hint_in_bytes=256 * 1024 * 1024,
        binary_operators=_preset["binary_operators"],
        unary_operators=_preset["unary_operators"],
        complexity_of_operators=_preset["complexity_of_operators"],
        nested_constraints=_preset["nested_constraints"],
        extra_sympy_mappings=_preset["extra_sympy_mappings"],
        batching=True,
        batch_size=10000,
        precision=32,
        turbo=True,
        bumper=True,
        update=False,
        progress=False,
        verbosity=0,
    )
    # Typed knobs from PreprocessingExtensionsConfig (override defaults when not None).
    for _typed_name, _pysr_name in (
        ("pysr_niterations", "niterations"),
        ("pysr_batching", "batching"),
        ("pysr_batch_size", "batch_size"),
        ("pysr_precision", "precision"),
        ("pysr_warm_start", "warm_start"),
    ):
        _typed_val = getattr(config, _typed_name, None)
        if _typed_val is not None:
            defaults[_pysr_name] = _typed_val
    # pysr_params dict is the final override -- power-user escape hatch beats typed fields.
    defaults.update(pysr_params)
    # Use a shallow copy so underlying YAML/dict config isn't mutated.
    merged_params = dict(defaults)

    _top_k_override = getattr(config, "pysr_top_k", None)
    top_k = int(_top_k_override) if _top_k_override is not None else min(5, merged_params.get("population_size", 20) // 2)
    # No hard cap on pool size by default: with batching=True PySR samples batch_size rows per iter,
    # so pool-size only controls diversity (the pool acts as the universe sampled-from across iters).
    # Caller can pin via PreprocessingExtensionsConfig.pysr_sample_size when memory is tight (each row
    # is ~26 floats * 4 bytes = ~100B in pandas; 4M rows = ~400 MB after the polars->pandas copy at
    # bruteforce.py:_run_pysr_feature_engineering).
    _sample_override = getattr(config, "pysr_sample_size", None)
    sample_n = min(len(train_df), int(_sample_override)) if _sample_override is not None else len(train_df)
    # Log when pool is large enough to noticeably affect memory; users can opt to cap via the config.
    if sample_n > 1_000_000:
        logger.info(
            "PySR pool size %d rows (no cap; set PreprocessingExtensionsConfig.pysr_sample_size "
            "to cap). batching=%s, batch_size=%s -- per-iter cost bounded by batch_size, not pool.",
            sample_n, merged_params.get("batching"), merged_params.get("batch_size"),
        )
    temp_target_col = "_pysr_y_"

    # Inject y_train as a temporary column (bruteforce expects target as a column in the DataFrame).
    # Caller already feeds the local ``train`` frame from ``apply_preprocessing_extensions._to_pandas``
    # so this isn't visible to caller code; the ``finally`` block below removes the temp column on any
    # exit path.
    existing_y = train_df.columns.tolist()
    while temp_target_col in existing_y:
        temp_target_col = "_" + temp_target_col
    train_df[temp_target_col] = np.asarray(y_train).ravel()

    # Thread the suite-level seed through to PySR's internal sampler. Without
    # this, run_pysr_feature_engineering's df.sample(...) draws a fresh row
    # subset each call and equations drift run-to-run.
    pysr_random_state = int(getattr(config, "random_seed", 42))
    try:
        model = run_pysr_feature_engineering(
            df=train_df,
            target_col=temp_target_col,
            sample_size=sample_n,
            encode_categoricals=False,
            verbose=0,
            pysr_params_override=merged_params,
            random_state=pysr_random_state,
        )
    except Exception:
        if verbose:
            logger.warning(
                "PySR fit failed; skipping symbolic feature engineering.",
                exc_info=True,
            )
        # The ``finally`` clause below ALWAYS runs and drops the temp column, including on this
        # except-return path. The previous duplicate drop here was redundant.
        return []
    finally:
        # Wave 52 (2026-05-20): wrap drop in try/except so a pandas KeyError chain
        # on a corrupted MultiIndex column or a read-only frame doesn't mask the
        # in-flight exception (errors="ignore" covers the missing-column case but
        # not deeper pandas-internal failures).
        try:
            train_df.drop(columns=[temp_target_col], inplace=True, errors="ignore")
        except Exception as _drop_err:
            logger.debug("pipeline: temp_target_col drop failed in finally: %s", _drop_err)

    # Apply top-K equations (by score)
    eq_df = model.equations_
    if eq_df is None or len(eq_df) == 0:
        return []
    eq_df = eq_df.sort_values(["score"], ascending=[False]).head(top_k)

    import hashlib

    new_cols = []
    _col_to_index: Dict[str, int] = {}
    # Equation-string column lives under several possible names depending on the PySR version (``equation``, ``sympy_format``, ``lambda_format``); fall back to the row repr if none are present so the hash still has a deterministic basis.
    _eq_col = next((c for c in ("equation", "sympy_format", "lambda_format") if c in eq_df.columns), None)
    for idx in eq_df.index:
        # Compute equation_str / col_name outside the predict try so any failure during
        # name construction itself surfaces (it's pure computation; if it raises it's a
        # real bug not a per-equation skip).
        if _eq_col is not None:
            equation_str = str(eq_df.loc[idx, _eq_col])
        else:
            equation_str = repr(eq_df.loc[idx].to_dict())
        hash8 = hashlib.blake2b(equation_str.encode("utf-8"), digest_size=4).hexdigest()
        col_name = f"pysr__{hash8}__{pysr_random_state}"
        if col_name in train_df.columns:
            # Same equation rediscovered in this seed -- the column already carries the
            # same values, skip recompute.
            if out_equations is not None:
                out_equations[col_name] = equation_str
            _col_to_index[col_name] = int(idx)
            continue
        # Per-equation try wraps all three predict-and-assign calls. Pre-fix bare
        # ``except: continue`` left schema drift when predict succeeded on train but
        # raised on val (e.g. odd dtype quirk, single edge value): train_df kept the
        # column, val_df / test_df didn't, and downstream fit raised a cryptic
        # feature-count mismatch with no log line. Now: on any failure, roll back
        # all three frames so the column is either uniformly present or uniformly
        # absent across splits, and log the skip so the operator sees how many
        # equations were dropped.
        try:
            train_df[col_name] = np.asarray(
                model.predict(train_df, index=idx), dtype=np.float32)
            if val_df is not None:
                val_df[col_name] = np.asarray(
                    model.predict(val_df, index=idx), dtype=np.float32)
            if test_df is not None:
                test_df[col_name] = np.asarray(
                    model.predict(test_df, index=idx), dtype=np.float32)
        except Exception as _eq_err:
            # Roll back any partial writes so train / val / test stay schema-consistent.
            for _frame in (train_df, val_df, test_df):
                if _frame is not None and col_name in getattr(_frame, "columns", []):
                    try:
                        _frame.drop(columns=[col_name], inplace=True)
                    except (TypeError, ValueError):
                        # polars (no inplace=) or unusual frame -- best-effort drop.
                        pass
            logger.warning(
                "PySR equation idx=%s skipped (col=%s): %s: %s. Train/val/test "
                "rolled back to keep splits schema-consistent.",
                idx, col_name, type(_eq_err).__name__, _eq_err,
            )
            continue
        new_cols.append(col_name)
        _col_to_index[col_name] = int(idx)
        if out_equations is not None:
            out_equations[col_name] = equation_str
    if out_transformer is not None and _col_to_index:
        out_transformer.append(PySRTransformer(model=model, col_to_index=_col_to_index, equations=out_equations or {}))
    return new_cols


def apply_preprocessing_extensions(
    train_df,
    val_df,
    test_df,
    config: Optional[PreprocessingExtensionsConfig],
    verbose: int = 1,
    y_train=None,
    out_pysr_equations: Optional[Dict[str, str]] = None,
):
    """Apply shared sklearn-based extensions to train/val/test after the Polars-ds pipeline.

    Returns (train, val, test, fitted_pipeline_or_None). Fastpath: when ``config``
    is None OR has zero active stages, returns inputs untouched with None pipeline.

    When ``out_pysr_equations`` is provided AND the PySR stage runs, the dict is populated with ``{column_name: equation_str}`` so the caller can persist the mapping under ``metadata["pysr_equations"]`` for predict-time replay (column names are content-hashed so different seeds discover distinct columns; loaders need the equation -> column mapping to rebind predict-time features).
    """
    if config is None:
        return train_df, val_df, test_df, None
    # Polars input -> convert to pandas (extensions use sklearn; mixing with the polars-native fastpath
    # would defeat the point if user opted in). Bare ``df.to_pandas()`` collapses pl.Enum / pl.Categorical
    # columns to object-dtype and copies through pyarrow's slow path. Use Arrow split-blocks bridge
    # (~32x throughput vs naive copy) and preserve Arrow-backed dtypes (pyarrow CategoricalDtype etc.) so
    # downstream sklearn estimators don't see "object" where pandas Categorical was expected.
    # Audit D P2-6 (2026-05-18): the bare ``df.to_pandas()`` fallback is the slow consolidation
    # copy path (~30x slower on wide frames; degrades pl.Enum → object). It fires only when the
    # installed polars version predates ``split_blocks=True`` (polars < 0.20.4). Log once at WARN
    # so operators on stale polars know they are on the slow bridge; ``logger`` is the module
    # logger so the message goes through the project's standard logging pipeline.
    _fallback_warned = [False]
    def _to_pandas(df):
        if df is None:
            return None
        if isinstance(df, pl.DataFrame):
            # 2026-05-21 fix: ``use_pyarrow_extension_array=True`` and
            # ``split_blocks=True`` are MUTUALLY EXCLUSIVE in polars 1.x
            # because the extension-array path internally passes
            # ``split_blocks=False`` to pyarrow's ``to_pandas``, then
            # user's explicit ``split_blocks=True`` produces a
            # ``TypeError: got multiple values for keyword argument
            # 'split_blocks'`` — the prior code mis-classified that as
            # "polars version too old" and dropped BOTH optimisations,
            # losing the 30x speedup AND the Arrow-backed dtypes.
            # Modern polars (>=0.20.4) supports ``split_blocks`` via
            # **kwargs; the user-visible warning was a false positive.
            # Strategy: prefer ``split_blocks=True`` (the 30x win),
            # accept that pl.Enum / pl.Categorical materialise as
            # pandas ``object`` (downstream cat encoders already
            # tolerate object-dtype string columns -- see the
            # pandas-2.1+ ``is_string_dtype`` audit committed 2026-05-20).
            try:
                return df.to_pandas(split_blocks=True, self_destruct=True)
            except TypeError as _err:
                if not _fallback_warned[0]:
                    logger.warning(
                        "polars.to_pandas(split_blocks=True, self_destruct=True) failed "
                        "with %s; falling back to bare .to_pandas() -- wide-frame "
                        "conversion will be ~30x slower. polars version=%s",
                        _err, pl.__version__,
                    )
                    _fallback_warned[0] = True
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
    _pysr_transformer_holder: list = []
    if getattr(config, "pysr_enabled", False):
        # _apply_pysr_fe mutates train/val/test in place; its return value
        # (the new column names) is intentionally discarded here. The fitted
        # PySRTransformer is captured via the out_transformer holder so the
        # caller can persist it in extensions_pipeline for predict-time replay.
        _apply_pysr_fe(
            train_df=train, val_df=val, test_df=test,
            y_train=y_train,
            config=config,
            verbose=verbose,
            out_equations=out_pysr_equations,
            out_transformer=_pysr_transformer_holder,
        )
    _pysr_transformer = _pysr_transformer_holder[0] if _pysr_transformer_holder else None

    # TF-IDF preflight: vectorize declared text columns and replace them with
    # numeric features before downstream sklearn steps (which expect numeric).
    #
    # Column-parity invariant: train, val, and
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

        # When ``tfidf_keep_sparse=True`` (default), the per-column TF-IDF
        # csr_matrix is wrapped as a pandas Sparse DataFrame; downstream
        # sparse-aware backends extract csr without densifying. At
        # ``max_features=5000`` on 1M rows this is ~40 GB dense vs ~hundreds
        # of MB sparse. When False, retain the legacy ``.toarray()`` path.
        _keep_sparse = bool(getattr(config, "tfidf_keep_sparse", True))

        def _spmatrix_to_df(spmat, columns, index):
            """Return a DataFrame whose ``columns`` are sparse-dtype when
            ``_keep_sparse`` is on, dense otherwise. Sparse-dtype DataFrames
            keep ``.columns`` and ``.index`` semantics; consumers densify
            implicitly via ``.to_numpy()``."""
            if _keep_sparse:
                return pd.DataFrame.sparse.from_spmatrix(spmat, columns=columns, index=index)
            return pd.DataFrame(spmat.toarray(), columns=columns, index=index)

        for col in usable_cols:
            vec = TfidfVectorizer(
                max_features=config.tfidf_max_features,
                ngram_range=tuple(config.tfidf_ngram_range),
            )
            train_text = train[col].fillna("").astype(str).values
            tfidf_train = vec.fit_transform(train_text)
            tfidf_pipes[col] = vec
            new_cols = [f"{col}__tfidf_{i}" for i in range(tfidf_train.shape[1])]
            tfidf_train_df = _spmatrix_to_df(tfidf_train, columns=new_cols, index=train.index)
            train = train.drop(columns=[col]).join(tfidf_train_df)
            for split_name, split_df in (("val", val), ("test", test)):
                if split_df is not None:
                    # Column presence was verified above in `usable_cols`
                    # filtering; this branch is now guaranteed safe.
                    text_arr = split_df[col].fillna("").astype(str).values
                    tfidf_sparse = vec.transform(text_arr)
                    new_split_df = _spmatrix_to_df(tfidf_sparse, columns=new_cols, index=split_df.index)
                    if split_name == "val":
                        val = split_df.drop(columns=[col]).join(new_split_df)
                    else:
                        test = split_df.drop(columns=[col]).join(new_split_df)

    # Numeric-only gate for the sklearn-bridge pipeline. The downstream
    # extensions (scaler / kbins / polynomial / nonlinear / dim_reducer
    # and the median-imputer in front) all reject object/string dtypes
    # with errors that range from clear (``Cannot use median strategy
    # with non-numeric data``) to opaque (``ValueError: The truth value
    # of an array with more than one element is ambiguous`` from inside
    # PolynomialFeatures or RobustScaler). The contract is "if you turn
    # on the sklearn-bridge, your frame should be numeric". When
    # non-numeric columns survived (unencoded cat_mid, embedding object
    # dtypes that the upstream cat-encoder skipped, etc.), drop them
    # here with a single-line WARN. Surfaced by 1M-harness seed=11.
    #
    # Note: the cat-encoder pre-pipeline normally runs BEFORE this
    # function, so under standard configs this drop is a no-op. The
    # gate exists to keep production callers + the 1M profiler harness
    # robust against axis combinations where cat_encoding canonicalised
    # to a path that bypassed the encoder.
    def _filter_to_numeric(_df):
        if _df is None:
            return _df, []
        # Wave 29 P2 fix (2026-05-20): pre-fix silently passed through
        # polars DataFrames; downstream ``_df.select_dtypes(...)`` then
        # raised AttributeError with no diagnostic naming the type. Coerce
        # polars -> pandas explicitly (matches the sibling helpers that
        # call ``get_pandas_view_of_polars_df``); raise on truly
        # unsupported types so the upstream caller sees the boundary.
        if not isinstance(_df, pd.DataFrame):
            try:
                import polars as _pl_local
                if isinstance(_df, _pl_local.DataFrame):
                    _df = _df.to_pandas()
                else:
                    return _df, []
            except ImportError:
                return _df, []
        # Bool columns are numerically valid for sklearn KBins /
        # StandardScaler / PolynomialFeatures (False=0, True=1) but
        # ``select_dtypes(include="number")`` EXCLUDES bool dtype - the
        # default code path silently drops useful binary features
        # (e.g. ``is_after_ps`` event-membership flags). Cast bool -> int8
        # in place so they pass the "number" gate. int8 is the smallest
        # dtype that round-trips True/False without precision loss; mass
        # cast is cheap (~1 byte per row).
        import numpy as _np_local
        _bool_cols = _df.select_dtypes(include="bool").columns.tolist()
        if _bool_cols:
            _df = _df.copy()  # avoid mutating caller's frame
            _df[_bool_cols] = _df[_bool_cols].astype(_np_local.int8)
        _num_cols = _df.select_dtypes(include="number").columns.tolist()
        _dropped = [c for c in _df.columns if c not in _num_cols]
        return _df[_num_cols], _dropped

    train, _dropped_train = _filter_to_numeric(train)
    val, _ = _filter_to_numeric(val)
    test, _ = _filter_to_numeric(test)
    if _dropped_train:
        logger.warning(
            "apply_preprocessing_extensions: dropped %d non-numeric column(s) "
            "before the sklearn-bridge pipeline (kbins / polynomial / scaler / "
            "dim_reducer all reject object dtype): %s. Encode these upstream "
            "(e.g. via OrdinalEncoder / OneHotEncoder in the suite's cat-encoder "
            "pre-pipeline) if you want them to participate in the extension "
            "transforms.",
            len(_dropped_train), _dropped_train[:8],
        )

    # All-null column filter. SimpleImputer(strategy="median") silently
    # drops columns with no observed values
    # (UserWarning: ``Skipping features without any observed values: ...``)
    # which silently shrinks n_features BELOW the dim_reducer's clamped
    # n_components. Surfaced by 1M-harness seed=99: PCA n_components
    # clamped from 10 -> 7 based on pre-imputer n_features=8, but
    # imputer then dropped x4 + x5 (both all-null in the synthetic
    # frame's missingness pattern), leaving 6 features, and PCA's
    # internal check raised n_components=7 vs n_features=6. Hoist the
    # all-null drop into our filter so the dim_n_components clamp can
    # see the post-imputation count up-front.
    if isinstance(train, pd.DataFrame) and train.shape[1] > 0:
        _all_null_cols = [c for c in train.columns if train[c].isna().all()]
        if _all_null_cols:
            train = train.drop(columns=_all_null_cols)
            if isinstance(val, pd.DataFrame):
                val = val.drop(columns=[c for c in _all_null_cols if c in val.columns])
            if isinstance(test, pd.DataFrame):
                test = test.drop(columns=[c for c in _all_null_cols if c in test.columns])
            logger.warning(
                "apply_preprocessing_extensions: dropped %d all-null "
                "column(s) before the sklearn-bridge pipeline "
                "(SimpleImputer median strategy would silently drop them "
                "and shrink the post-imputer n_features below the "
                "dim_reducer's n_components clamp, causing a downstream "
                "PCA / TruncatedSVD / FastICA ValueError): %s.",
                len(_all_null_cols), _all_null_cols[:8],
            )

    n_features = train.shape[1]
    # Dim-reducer n_components clamp. PCA / TruncatedSVD / KernelPCA / NMF
    # / FastICA / LDA etc. raise
    # ``ValueError: n_components=K must be between 0 and min(n_samples,
    # n_features)`` when the requested K exceeds the available feature
    # count. The numeric-only filter above can reduce n_features below
    # the user's configured dim_n_components (surfaced by 1M-harness
    # seed=99: PCA n_components=10 on a 9-feature frame after cat_low /
    # cat_mid were filtered). Clamp to min(n_features, n_samples,
    # dim_n_components) and emit a WARN; the user explicitly chose
    # dimensionality reduction, so silently dropping the step would be
    # worse than running it at a lower K.
    if config.dim_reducer is not None and config.dim_n_components is not None:
        n_samples = train.shape[0]
        _clamp_max = max(1, min(n_features - 1, n_samples - 1))
        if config.dim_n_components > _clamp_max:
            try:
                config = config.model_copy(update={"dim_n_components": _clamp_max})
            except AttributeError:
                # Older pydantic / plain-attribute fallback: mutate a shallow copy.
                import copy as _copy
                config = _copy.copy(config)
                config.dim_n_components = _clamp_max
            logger.warning(
                "apply_preprocessing_extensions: clamped dim_n_components "
                "from user-requested value to %d (= min(n_features-1=%d, "
                "n_samples-1=%d)) so the %s reducer's n_components stays "
                "within the sklearn-required (0, min(n_samples, n_features)) "
                "range. Increase the upstream feature count or drop the "
                "dim_reducer if you need K=%d.",
                _clamp_max, n_features - 1, n_samples - 1,
                config.dim_reducer, _clamp_max,
            )
    # iter-69 byte-aware polynomial auto-tune. ``memory_safety_max_features``
    # gates by column count alone; on wide post-onehot frames at degree=2
    # the column count stays under the cap but the dense
    # (n_samples, projected) float64 array exceeds available RAM
    # (iter-69 surfaced n=81000, projected=1711 -> 1.03 GiB allocation
    # MemoryError). When ``memory_safety_max_bytes`` is set, compute the
    # exact byte-cost via the same shape formula sklearn would use and
    # auto-tune the polynomial step downward (flip interaction_only ->
    # decrement degree -> skip) until the projected array fits.
    _byte_cap = getattr(config, "memory_safety_max_bytes", None)
    if (
        config.polynomial_degree is not None
        and config.polynomial_degree > 0
        and _byte_cap not in (None, 0)
        and isinstance(train, pd.DataFrame)
        and train.shape[0] > 0
        and n_features > 0
    ):
        from mlframe.training.feature_handling.polynomial import _projected_output_cols
        _n_samples = train.shape[0]
        _eff_degree = int(config.polynomial_degree)
        _eff_interaction = bool(config.polynomial_interaction_only)
        _eff_skip = False

        def _proj_bytes(_d: int, _io: bool) -> int:
            _p = _projected_output_cols(n_features, _d, _io)
            return int(_n_samples) * int(_p) * 8

        _bytes = _proj_bytes(_eff_degree, _eff_interaction)
        if _bytes > _byte_cap and not _eff_interaction:
            logger.warning(
                "apply_preprocessing_extensions: polynomial output would "
                "allocate %.2f MiB at n_samples=%d * projected=%d * 8 bytes; "
                "cap=%.2f MiB. Flipping polynomial_interaction_only=True "
                "to drop pure-power terms.",
                _bytes / (1024 * 1024), _n_samples,
                _projected_output_cols(n_features, _eff_degree, _eff_interaction),
                _byte_cap / (1024 * 1024),
            )
            _eff_interaction = True
            _bytes = _proj_bytes(_eff_degree, _eff_interaction)
        while _bytes > _byte_cap and _eff_degree > 1:
            logger.warning(
                "apply_preprocessing_extensions: polynomial output still "
                "%.2f MiB > cap %.2f MiB at degree=%d, interaction_only=%s; "
                "decrementing degree -> %d.",
                _bytes / (1024 * 1024), _byte_cap / (1024 * 1024),
                _eff_degree, _eff_interaction, _eff_degree - 1,
            )
            _eff_degree -= 1
            _bytes = _proj_bytes(_eff_degree, _eff_interaction)
        if _bytes > _byte_cap:
            logger.warning(
                "apply_preprocessing_extensions: polynomial output still "
                "%.2f MiB > cap %.2f MiB even at degree=1; skipping the "
                "polynomial step entirely.",
                _bytes / (1024 * 1024), _byte_cap / (1024 * 1024),
            )
            _eff_skip = True
        if _eff_skip or _eff_degree != int(config.polynomial_degree) or _eff_interaction != bool(config.polynomial_interaction_only):
            try:
                _new_deg = None if _eff_skip else _eff_degree
                config = config.model_copy(update={
                    "polynomial_degree": _new_deg,
                    "polynomial_interaction_only": _eff_interaction,
                })
            except AttributeError:
                import copy as _copy
                config = _copy.copy(config)
                config.polynomial_degree = None if _eff_skip else _eff_degree
                config.polynomial_interaction_only = _eff_interaction

    steps = _build_extension_steps(config, n_features=n_features)
    if not steps:
        if _pysr_transformer is not None or tfidf_pipes:
            # PySR / TF-IDF applied but no sklearn pipeline. Bundle so predict-time
            # replay sees the PySR transformer; legacy raw-dict shape is used only
            # when PySR is absent so untouched persisted artefacts keep loading.
            if _pysr_transformer is not None:
                return train, val, test, PreprocessingExtensionsBundle(
                    pysr=_pysr_transformer, tfidf=tfidf_pipes or None, sklearn_pipe=None,
                )
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

    # Preserve named-transformer output column names so downstream
    # diagnostics and feature-importance reports stay interpretable. Pre-fix
    # we relabelled every column to ``ext_<i>`` which collapsed all
    # provenance ("which scaler? which equation? which TF-IDF token?").
    # sklearn>=1.3 exposes ``get_feature_names_out``; fall back to
    # ``ext_<step>_<i>`` derived from the last step's name otherwise.
    def _build_output_column_names(n_cols: int) -> list:
        try:
            names = pipe.get_feature_names_out()
            if names is not None and len(names) == n_cols:
                return [str(n) for n in names]
        except (AttributeError, ValueError, NotImplementedError):
            pass
        # Fallback: tag with the final transformer's step name so at least
        # the stage is recoverable (e.g. ``ext_dim_reducer_3``).
        last_step_name = steps[-1][0] if steps else "ext"
        return [f"ext_{last_step_name}_{i}" for i in range(n_cols)]

    def _to_df(arr, template):
        if arr is None:
            return None
        # sklearn pipeline output may still be sparse if no dim_reducer was
        # configured (PCA/TruncatedSVD/etc densify; pass-through stages preserve
        # sparsity). Wrap into a Sparse-dtype DataFrame when keep-sparse is on.
        try:
            import scipy.sparse as _sp
            _is_sparse = _sp.issparse(arr)
        except ImportError:
            _is_sparse = False
        col_names = _build_output_column_names(arr.shape[1])
        if _is_sparse:
            if bool(getattr(config, "tfidf_keep_sparse", True)):
                return pd.DataFrame.sparse.from_spmatrix(
                    arr, columns=col_names,
                    index=getattr(template, "index", None),
                )
            arr = arr.toarray()
        return pd.DataFrame(arr, columns=col_names, index=getattr(template, "index", None))

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
    # Bundle PySR transformer / TFIDF dict alongside the sklearn pipe so
    # predict-time replay can re-emit symbolic columns BEFORE the TF-IDF /
    # sklearn replay. Without the bundle, the persisted artefact dropped the
    # PySR step and predict frames silently lacked the pysr_* columns the
    # model was trained on.
    if _pysr_transformer is not None or tfidf_pipes:
        return train_out, val_out, test_out, PreprocessingExtensionsBundle(
            pysr=_pysr_transformer, tfidf=tfidf_pipes or None, sklearn_pipe=pipe,
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
            if _first is not None and (hasattr(_first, "shape") or (
                hasattr(_first, "__len__") and not isinstance(_first, (str, bytes))
            )):
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
    if method not in ("robust", "standard", "min_max"):
        raise ValueError(
            f"_select_scalable_numeric_columns: unknown method={method!r}; "
            "expected one of 'robust', 'standard', 'min_max'."
        )

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
    # (observed in fuzz seeds). The historical workaround
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
    # compatibility with the legacy fastpath behaviour.
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

    # Convert int to float32 for better compatibility.
    # Skip already-narrow Int8/Int16 columns (typically datetime decomposition
    # outputs: day/weekday/month/hour all fit Int8); widening them to float32
    # quadruples memory for zero downstream benefit since tree models accept
    # int8 directly. We cast only Int32/Int64/UInt32/UInt64 to f32. fix audit
    # row FE-L-3.
    try:
        _narrow_int_dtypes = {pl.Int8, pl.Int16, pl.UInt8, pl.UInt16}
        _wide_int_cols = [
            name for name, dtype in train_df.schema.items()
            if dtype.is_integer() and dtype not in _narrow_int_dtypes
        ]
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
        logger.info(f"  Polars-ds pipeline created -- scaler={config.scaler_name or 'none'}, encoding={config.categorical_encoding or 'none'}, {bp_elapsed:.1f}s")
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

    # Datetime column decomposition is performed in
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
            # can compare val/test schemas against it below. Without
            # this snapshot, pipeline.transform(val_df) was called
            # without any schema validation; missing/extra cols or dtype
            # mismatches silently propagated either to a downstream
            # sklearn shape-error or garbage output.
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
        # Identify categorical features (exclude text/embedding columns).
        # Embedding columns can sneak past the dtype filter when stored as
        # pandas object-of-ndarray (an embedding vector per row). They look
        # categorical via dtype.name=='object' but their cells are ndarrays
        # that hash() raises on, crashing category_encoders.OrdinalEncoder's
        # internal .unique() call. Detect and exclude via first-cell shape.
        def _looks_embedding(_series):
            if _series.dtype != object:
                return False
            try:
                _first = next((v for v in _series.head(8) if v is not None), None)
            except Exception:
                return False
            if _first is None:
                return False
            return hasattr(_first, "shape") or (
                hasattr(_first, "__len__") and not isinstance(_first, (str, bytes))
            )

        # High-cardinality object/string columns are free-text, not categoricals.
        # The downstream auto-detect (_phase_auto_detect_feature_types) will
        # promote them to text_features, but it runs AFTER this pipeline-fit
        # phase. Without this guard, the elif branch below converts text_col
        # to pandas Categorical in-place; CB Pool then rejects it ("dtype
        # 'category' but not in cat_features list") because by the time the
        # CB Pool is built, text_col is correctly listed in text_features.
        # Threshold mirrors FeatureTypesConfig.cat_text_cardinality_threshold
        # default (300) and uses a SAMPLE-based unique count for cheap detection
        # on million-row frames. Surfaced by fuzz iter#49 (object+text_col+cb).
        _CAT_CARDINALITY_LIMIT = 300
        _SAMPLE_SIZE = 5000

        def _looks_text(_series):
            dtype_name = _series.dtype.name
            if dtype_name not in ("object", "string", "string[pyarrow]", "large_string[pyarrow]"):
                return False
            n_rows = len(_series)
            if n_rows == 0:
                return False
            sample = _series.iloc[: min(_SAMPLE_SIZE, n_rows)]
            try:
                n_unique_sample = sample.nunique(dropna=True)
            except TypeError:
                return False
            return n_unique_sample > _CAT_CARDINALITY_LIMIT

        cat_features = [
            col for col in train_df.columns
            if train_df[col].dtype.name in PANDAS_CATEGORICAL_DTYPES
            and col not in _exclude_from_encoding
            and not _looks_embedding(train_df[col])
            and not _looks_text(train_df[col])
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

            # Joint train+val union for stable codes across splits.
            _safe_val = val_df if (val_df is not None and len(val_df) > 0) else None
            _safe_test = test_df if (test_df is not None and len(test_df) > 0) else None
            if train_df is not None and len(train_df) > 0:
                prepare_dfs_for_catboost_joint(
                    train_df=train_df, val_df=_safe_val, test_df=_safe_test,
                    cat_features=cat_features,
                )

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
    "prepare_dfs_for_catboost_joint",
    "create_polarsds_pipeline",
    "fit_and_transform_pipeline",
    "apply_preprocessing_extensions",
]
