"""Perf-mode downgrade of a combo for fast suite-wiring smoke runs."""

from __future__ import annotations

from .combo import FuzzCombo  # noqa: F401  (annotation strings under PEP 563)


# ---------------------------------------------------------------------------
# Frame builder — turns a combo into (df, target_col, cat_feature_names)
# ---------------------------------------------------------------------------


def apply_perf_mode(combo: FuzzCombo) -> FuzzCombo:
    """FUZZ-1 (2026-05-23): return a config-coverage downgrade of ``combo``.

    Goal: verify suite wiring on every combo in seconds instead of minutes.
    Pins ``n_rows=1000`` (10-100x smaller), ``iterations=1`` (1-10x smaller),
    and disables the heavy optional phases (MRMR, BorutaShap, ensembles,
    baseline_diagnostics, dummy_baselines, composite_discovery, target-
    distribution-analyzer). Useful when iterating on suite-level wiring
    correctness without paying real fit cost.

    Quality / metric assertions are NOT meaningful after this downgrade --
    use only as a smoke-test pre-filter (e.g., ``MLFRAME_FUZZ_PERF_MODE=1
    pytest tests/training/test_fuzz_suite.py``). Real perf / accuracy runs
    use the untouched combo straight from ``enumerate_combos``.

    Returns a fresh ``FuzzCombo`` instance (dataclass.replace) so the
    original is untouched -- enables caller to bench the same combo in
    BOTH modes back-to-back.
    """
    import dataclasses

    return dataclasses.replace(
        combo,
        n_rows=1000,
        iterations=1,
        # FS / heavy passes
        use_mrmr_fs=False,
        use_boruta_shap_cfg=False,
        # Ensembles
        use_ensembles=False,
        # Diagnostics / baselines
        baseline_diagnostics_enabled_cfg=False,
        dummy_baselines_enabled_cfg=False,
        # Composite discovery
        composite_discovery_enabled_cfg=False,
        # Target distribution analyzer (heavy polars + LGB quick model)
        enable_target_distribution_analyzer_cfg=False,
        # Feature-handling config (PCA / dim reducers / TF-IDF wrap)
        custom_prep=None,
        # Reduce RFECV (only fires if rfecv_estimator_cfg != 'none', but
        # downgrade to bare-minimum splits if a combo enabled it).
        rfecv_cv_n_splits_cfg=2,
        # Tighter eval
        early_stopping_rounds_cfg=2,
    )
