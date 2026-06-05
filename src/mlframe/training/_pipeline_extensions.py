"""Carved out of ``mlframe.training.pipeline``.

Re-imported at the parent's module bottom so historical
``from mlframe.training.pipeline import apply_preprocessing_extensions``
resolves transparently.
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

import pandas as pd
import polars as pl
import polars.selectors as cs
from typing import Dict, Union, Optional, List, Tuple
from collections import Counter
from pyutilz.system import clean_ram
from .utils import maybe_clean_ram_adaptive, log_ram_usage
from pyutilz.pandaslib import ensure_dataframe_float32_convertability

from .configs import PreprocessingBackendConfig, PreprocessingExtensionsConfig
from .strategies import PANDAS_CATEGORICAL_DTYPES, get_polars_cat_columns

logger = logging.getLogger("mlframe.training.pipeline")


def sparse_df_from_spmatrix(spmat, columns, index):
    """``pd.DataFrame.sparse.from_spmatrix`` with a guaranteed ``0`` fill value.

    pandas 3.0 changed ``DataFrame.sparse.from_spmatrix`` to default the
    per-column ``SparseDtype`` fill_value to ``nan`` (it was ``0`` through
    pandas 2.x). The structural-zero entries of a TF-IDF / one-hot CSR then
    densify to ``NaN`` downstream (``.to_numpy()`` / sklearn's input
    validation), wiping the signal -- a linear model trained on the result
    collapses to chance. Reproduced 2026-05-27: the tfidf-lift test scores
    AUROC 1.0 on pandas 2.3.3 but 0.55 on pandas 3.0.3 with everything else
    held equal.

    We relabel each column's fill_value to ``0`` by reusing the existing
    ``sp_values`` / ``sp_index`` backing arrays -- O(nnz), NO densification,
    so the large-sparse memory guarantee (the reason the sparse path exists:
    ~40 GB dense vs hundreds of MB sparse at max_features=5000 on 1M rows) is
    preserved. On pandas 2.x where the fill is already ``0`` this is a no-op.
    """
    df = pd.DataFrame.sparse.from_spmatrix(spmat, columns=columns, index=index)
    _needs_zero_fill = any(
        isinstance(_dt, pd.SparseDtype) and (pd.isna(_dt.fill_value) or _dt.fill_value != 0)
        for _dt in df.dtypes
    )
    if not _needs_zero_fill:
        return df
    from pandas.arrays import SparseArray
    _zero_sparse = pd.SparseDtype("float64", 0.0)
    _fixed = {}
    for _name in df.columns:
        _arr = df[_name].array
        _fixed[_name] = SparseArray(
            _arr.sp_values, sparse_index=_arr.sp_index,
            fill_value=0.0, dtype=_zero_sparse,
        )
    return pd.DataFrame(_fixed, index=df.index)


# Thread-count env vars must be set BEFORE Julia/PySR boots; we defer the set until the first
# ``_apply_pysr_fe`` call so importers who never touch PySR don't get their env mutated.


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
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .pipeline import PySRTransformer, _maybe_set_pysr_thread_env
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
    # Explicit None-check rather than ``or "standard"``: the latter would silently rewrite ``pysr_operator_preset=""`` (a config-fixture mistake) to "standard" and mask the typo. Same class of bug as the prior ``or 42`` rewrite for ``random_state=0`` in _phase_composite_discovery.
    _preset_raw = getattr(config, "pysr_operator_preset", None)
    _preset_name = "standard" if _preset_raw is None else _preset_raw
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

    # Inject y_train as a temporary column (bruteforce expects target as a column in the DataFrame). Caller already feeds the local ``train`` frame from ``apply_preprocessing_extensions._to_pandas`` so this isn't visible to caller code; the ``finally`` block below removes the temp column on any exit path.
    #
    # The injection MUST live INSIDE the try block. The pre-fix shape did the assignment one line before ``try:``, leaving a narrow leak window: an exception fired between injection and try entry (e.g. ``int(getattr(config, "random_seed", 42))`` on a malformed config value) bypassed the ``finally`` and the temp target column leaked back to the caller's frame as a fake numeric feature.
    existing_y = train_df.columns.tolist()
    while temp_target_col in existing_y:
        temp_target_col = "_" + temp_target_col

    # Thread the suite-level seed through to PySR's internal sampler. Without this, run_pysr_feature_engineering's df.sample(...) draws a fresh row subset each call and equations drift run-to-run.
    _column_was_injected = False
    try:
        pysr_random_state = int(getattr(config, "random_seed", 42))
        train_df[temp_target_col] = np.asarray(y_train).ravel()
        _column_was_injected = True
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
        return []
    finally:
        # Wrap drop in try/except so a pandas KeyError chain on a corrupted MultiIndex column or a read-only frame doesn't mask the in-flight exception (errors="ignore" covers the missing-column case but not deeper pandas-internal failures). Skip the drop when injection itself failed -- nothing to remove.
        if _column_was_injected:
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


def _has_active_extension_stage(config) -> bool:
    """True iff ``config`` activates at least one extension stage (PySR / TF-IDF / sklearn-bridge).

    Used to short-circuit ``apply_preprocessing_extensions`` BEFORE the polars->pandas down-convert: with zero active stages the conversion is pure cost (and OOM risk on 100+GB polars frames) for a function that would return the inputs unchanged anyway.
    """
    if config is None:
        return False
    if getattr(config, "pysr_enabled", False):
        return True
    if getattr(config, "tfidf_columns", None):
        return True
    return any(
        getattr(config, _attr, None) is not None
        for _attr in (
            "scaler",
            "binarization_threshold",
            "kbins",
            "polynomial_degree",
            "nonlinear_features",
            "dim_reducer",
        )
    )


def _filter_to_numeric(_df, keep_cols=None):
    """Drop non-numeric columns and bool-to-int8 promote in place, returning ``(filtered_view, dropped_names)``.

    Module-level (not nested inside ``apply_preprocessing_extensions``) so the regression sensor for the 100GB no-copy rule can import it directly. Caller-frame mutation contract: bool columns ARE promoted to int8 in the caller's frame -- this is the documented price of obeying the no-full-frame-copy rule on 100+GB workloads. The float / numeric columns are exposed via a column-subset view; ``_df[_num_cols]`` returns a view that shares the underlying block buffers with the caller's frame for the unchanged columns.

    Cross-split parity: ``keep_cols`` pins the column set decided on TRAIN so val/test keep exactly the same columns (mirrors the ``_all_null_cols`` cross-split alignment). Without it, a per-split ``select_dtypes`` recompute can diverge (a column numeric on train but object on val, or vice versa) and break the downstream sklearn ``transform`` with an opaque feature-count mismatch. The caller passes ``list(train_filtered.columns)`` as ``keep_cols`` for the val/test calls.
    """
    if _df is None:
        return _df, []
    # Wave 29 P2 (2026-05-20): pre-fix silently passed through polars DataFrames; downstream ``_df.select_dtypes(...)`` then raised AttributeError with no diagnostic naming the type. Coerce polars -> pandas explicitly; raise on truly unsupported types so the upstream caller sees the boundary.
    if not isinstance(_df, pd.DataFrame):
        try:
            import polars as _pl_local
            if isinstance(_df, _pl_local.DataFrame):
                # Mirror the ``apply_preprocessing_extensions._to_pandas`` Arrow split-blocks path (this file lines ~360-371): bare .to_pandas() consolidates Arrow buffers (~30x slower on wide frames + degrades pl.Enum to object dtype). self_destruct=True is safe here because the caller's _df reference is being rebound (we don't read _df after this line).
                try:
                    _df = _df.to_pandas(split_blocks=True, self_destruct=True)
                except TypeError:
                    _df = _df.to_pandas()
            else:
                return _df, []
        except ImportError:
            return _df, []
    import numpy as _np_local
    # When ``keep_cols`` is pinned (val/test follow train), promote only the pinned bool columns then select exactly the pinned set, preserving cross-split column parity even if a column's per-split dtype differs.
    if keep_cols is not None:
        _keep = [c for c in keep_cols if c in _df.columns]
        for _c in _keep:
            if _df[_c].dtype == bool:
                _df[_c] = _df[_c].astype(_np_local.int8)
        _dropped = [c for c in _df.columns if c not in set(_keep)]
        return _df[_keep], _dropped
    # Bool columns are numerically valid for sklearn KBins / StandardScaler / PolynomialFeatures (False=0, True=1) but ``select_dtypes(include="number")`` EXCLUDES bool dtype - the default code path silently drops useful binary features (e.g. ``is_after_ps`` event-membership flags). Cast bool -> int8 so they pass the "number" gate; int8 is the smallest dtype that round-trips True/False without precision loss.
    # The CRITICAL no-df.copy() rule (CLAUDE.md "Memory / RAM constraints") forbids a full-frame clone here -- on a 100+ GB frame that would OOM the host. Pre-2026-05-24 fix did ``_df = _df.copy()``; replaced with a per-column in-place dtype mutation. Single-column assignment is a block-level dtype swap, so the unchanged numeric columns keep their original buffers (the regression sensor asserts ``np.shares_memory`` on the float columns).
    _bool_cols = _df.select_dtypes(include="bool").columns.tolist()
    for _c in _bool_cols:
        _df[_c] = _df[_c].astype(_np_local.int8)
    _num_cols = _df.select_dtypes(include="number").columns.tolist()
    _dropped = [c for c in _df.columns if c not in set(_num_cols)]
    return _df[_num_cols], _dropped


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
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .pipeline import PreprocessingExtensionsBundle, _build_extension_steps
    if config is None:
        return train_df, val_df, test_df, None
    # Fastpath: zero active stages -> no work to do. Return inputs UNTOUCHED (no polars->pandas down-convert). Without this gate the function paid the full Arrow->pandas conversion on every frame even when nothing was configured, defeating the polars fastpath and risking OOM on 100+GB polars frames for a no-op call.
    if not _has_active_extension_stage(config):
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
            # ``use_pyarrow_extension_array=True`` and
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
                return sparse_df_from_spmatrix(spmat, columns, index)
            return pd.DataFrame(spmat.toarray(), columns=columns, index=index)

        for col in usable_cols:
            vec = TfidfVectorizer(
                max_features=config.tfidf_max_features,
                ngram_range=tuple(config.tfidf_ngram_range),
            )
            # bench-attempt-rejected (_benchmarks/bench_tfidf_input_path.py): feeding the Series directly or .to_numpy(object, na_value="") instead of .values is within ~1% (noise) at n=5k/50k; .values stays.
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

    # Numeric-only gate for the sklearn-bridge pipeline. The downstream extensions (scaler / kbins / polynomial / nonlinear / dim_reducer and the median-imputer in front) all reject object/string dtypes with errors that range from clear (``Cannot use median strategy with non-numeric data``) to opaque (``ValueError: The truth value of an array with more than one element is ambiguous`` from inside PolynomialFeatures or RobustScaler). The contract is "if you turn on the sklearn-bridge, your frame should be numeric". When non-numeric columns survived (unencoded cat_mid, embedding object dtypes that the upstream cat-encoder skipped, etc.), drop them here with a single-line WARN. Surfaced by 1M-harness seed=11.
    #
    # Note: the cat-encoder pre-pipeline normally runs BEFORE this function, so under standard configs this drop is a no-op. The gate exists to keep production callers + the 1M profiler harness robust against axis combinations where cat_encoding canonicalised to a path that bypassed the encoder.

    # Decide the kept-numeric column set ONCE on train, then pin val/test to the SAME list so a column that is numeric on train but object on val (or vice versa) can't silently diverge the per-split schema and break the downstream sklearn transform.
    train, _dropped_train = _filter_to_numeric(train)
    _kept_train = list(train.columns) if isinstance(train, pd.DataFrame) else None
    val, _ = _filter_to_numeric(val, keep_cols=_kept_train)
    test, _ = _filter_to_numeric(test, keep_cols=_kept_train)
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
                # Older pydantic / plain-attribute fallback: deepcopy so nested mutable config fields aren't aliased back to the caller's object.
                import copy as _copy
                config = _copy.deepcopy(config)
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
                config = _copy.deepcopy(config)
                config.polynomial_degree = None if _eff_skip else _eff_degree
                config.polynomial_interaction_only = _eff_interaction

    # Thread the suite seed into RBFSampler / Nystroem / dim-reducer so their random projections are reproducible across reruns (was hardcoded 42).
    _ext_random_state = int(getattr(config, "random_seed", 42) or 42)
    steps = _build_extension_steps(config, n_features=n_features, random_state=_ext_random_state)
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
                return sparse_df_from_spmatrix(
                    arr, col_names, getattr(template, "index", None),
                )
            arr = arr.toarray()
        # bench-attempt-rejected (2026-05-24): dtype/contiguity gate around this constructor (skip if C-contig + dtype-matched) measured 0.06 ms vs 0.06 ms
        # at N=100k D=50 across all three input shapes (float64 C-contig, float32 C-contig, float64 F-contig). Modern pandas block management makes the
        # constructor essentially free; gate adds branches without speedup.
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
