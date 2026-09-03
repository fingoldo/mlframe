"""PySR symbolic-regression stage of ``apply_preprocessing_extensions``.

Carved out of ``_pipeline_extensions`` (which had grown past the 1000-LOC house limit) and re-imported
there, so ``from ._pipeline_extensions import _apply_pysr_fe`` and every existing call site keep resolving.
This stage is fully optional -- it only runs when ``PreprocessingExtensionsConfig.pysr_enabled`` is set --
which makes it the natural seam: nothing else in the parent module depends on it.
"""
from __future__ import annotations

import logging
import os
import subprocess  # nosec B404 - subprocess used below with list args only, no shell=True
from typing import Dict, Optional

import pandas as pd

from ..configs import PreprocessingExtensionsConfig
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.training.pipeline")


def _apply_pysr_fe(
    *,
    train_df: pd.DataFrame,
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
    from . import PySRTransformer, _maybe_set_pysr_thread_env
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
            logger.warning("PySR feature engineering is enabled but the pysr / Julia " "runtime is not importable. Skipping.")
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
    except Exception:  # best-effort: symbolic feature engineering is an optional enhancement
        if verbose:
            logger.warning(
                "PySR fit failed; skipping symbolic feature engineering.",
                exc_info=True,
            )
        else:
            logger.debug("PySR fit failed; skipping symbolic feature engineering.", exc_info=True)
        return []
    finally:
        # Wrap drop in try/except so a pandas KeyError chain on a corrupted MultiIndex column or a read-only frame doesn't mask the in-flight exception (errors="ignore" covers the missing-column case but not deeper pandas-internal failures). Skip the drop when injection itself failed -- nothing to remove.
        if _column_was_injected:
            try:
                train_df.drop(columns=[temp_target_col], inplace=True, errors="ignore")  # noqa: PD002 -- must mutate the caller's train_df object in place; a rebind here would not propagate outside this function
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
            train_df[col_name] = np.asarray(model.predict(train_df, index=idx), dtype=np.float32)
            if val_df is not None:
                val_df[col_name] = np.asarray(model.predict(val_df, index=idx), dtype=np.float32)
            if test_df is not None:
                test_df[col_name] = np.asarray(model.predict(test_df, index=idx), dtype=np.float32)
        except Exception as _eq_err:
            # Roll back any partial writes so train / val / test stay schema-consistent.
            for _frame in (train_df, val_df, test_df):
                if _frame is not None and col_name in getattr(_frame, "columns", []):
                    try:
                        _frame.drop(columns=[col_name], inplace=True)  # noqa: PD002 -- _frame is a loop var aliasing train_df/val_df/test_df; a rebind would not propagate the rollback to the caller's actual frames
                    except (TypeError, ValueError):
                        # polars (no inplace=) or unusual frame -- best-effort drop.
                        pass
            log_throttle(
                logger,
                "pysr_equation_skipped",
                logging.WARNING,
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
