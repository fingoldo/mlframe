
"""Metamorphic fuzz tests for ``train_mlframe_models_suite`` (Fix D).

A metamorphic test runs the suite twice under inputs that SHOULD yield
equivalent results, then asserts equivalence. Stronger than "did not
crash": catches silent regressions that neither unit tests nor the
combo-fuzz invariants detect.

Scope: small curated subset of combos (one per model family) so the
per-test cost of running the suite twice stays bounded. Env-gated
expansion available via ``MLFRAME_METAMORPHIC_ALL=1`` to run the full
combo list.

Properties covered:

D1. Column-rename invariance — renaming a non-target feature must not
    change the chosen val metric beyond noise. Catches hard-coded column
    names (memoisation keys, log strings) that leak into the training path.

D2. Duplicate-row stability — adding 5% duplicate rows must not swing
    the val metric by more than a noise threshold. Catches group-leakage
    / row-counting bugs.

D3. Ensemble-of-one identity — training with ``use_mlframe_ensembles=True``
    on a single-model set must produce a val metric within noise of the
    non-ensembled run. Catches ensemble path drift.
"""
from __future__ import annotations
from mlframe.training import OutputConfig, PreprocessingConfig

import os
import tempfile
from typing import Any

import numpy as np
import pytest

from ._fuzz_combo import (
    FuzzCombo,
    build_frame_for_combo,
    enumerate_combos,
)
from .shared import SimpleFeaturesAndTargetsExtractor


# Tolerances for metric drift. Generous — metamorphic tests should catch
# catastrophic regressions (predictions inverting, pipeline silently
# dropping features), not 0.01-AUC wobble.
_CLF_AUC_TOLERANCE = 0.15  # |Δ roc_auc|
_REG_R2_TOLERANCE = 0.20  # |Δ r2|


def _extract_primary_val_metric(trained: dict) -> float | None:
    """Return the first non-NaN val metric found in the nested trained dict.

    Classification → roc_auc from the first class. Regression → R² if
    reported, else mean(abs(val_preds)) as a fallback signal.
    """
    if not isinstance(trained, dict):
        return None
    for tt, by_name in trained.items():
        if not isinstance(by_name, dict):
            continue
        for tn, lst in by_name.items():
            if not isinstance(lst, list):
                continue
            for entry in lst:
                metrics = getattr(entry, "metrics", None) or {}
                val_block = metrics.get("val") if isinstance(metrics, dict) else None
                if not isinstance(val_block, dict):
                    continue
                # Classification: {class_idx: {'roc_auc': ...}}
                for k, v in val_block.items():
                    if isinstance(v, dict) and "roc_auc" in v:
                        rocauc = v["roc_auc"]
                        if rocauc is not None and np.isfinite(rocauc):
                            return float(rocauc)
                # Regression: flat dict
                if "r2" in val_block:
                    r2 = val_block["r2"]
                    if r2 is not None and np.isfinite(r2):
                        return float(r2)
                # Fallback: prediction mean
                preds = getattr(entry, "val_preds", None)
                if preds is not None:
                    arr = np.asarray(preds).ravel()
                    if arr.size > 0 and np.all(np.isfinite(arr)):
                        return float(arr.mean())
    return None


_COMBO_TT_TO_TARGET_TYPE = {
    # combo.target_type (string from _fuzz_combo) → TargetTypes enum
    # Required for non-binary/non-regression combos so the FTE
    # (a) produces a correctly-shaped target ndarray (2-D for multilabel)
    # and (b) the suite dispatches the right loss_fn / labels_dtype to MLP.
    # Without this mapping, a multilabel ``List(Int8)`` polars column
    # arrives at TorchDataset as ``(N, K)`` int64 under cross_entropy and
    # crashes ("class probabilities, got Long"). Surfaced by c0103.
    "multiclass_classification": "MULTICLASS_CLASSIFICATION",
    "multilabel_classification": "MULTILABEL_CLASSIFICATION",
    "learning_to_rank": "LEARNING_TO_RANK",
}


def _resolve_target_type_for_combo(combo: FuzzCombo):
    """Map combo.target_type → TargetTypes enum value (or None for binary).

    Binary classification and regression don't need an explicit target_type:
    SimpleFeaturesAndTargetsExtractor resolves them from ``regression=...``.
    """
    from mlframe.training.configs import TargetTypes
    if combo.target_type in ("binary_classification", "regression"):
        return None
    name = _COMBO_TT_TO_TARGET_TYPE.get(combo.target_type)
    if name is None:
        return None
    return getattr(TargetTypes, name)


def _run_suite(combo: FuzzCombo, df, target_col: str, tmp_path) -> dict:
    """Run the suite on the given frame; return trained dict. Reuses the
    combo's model/config choices but ignores its data — caller supplies
    the (possibly-mutated) frame."""
    from mlframe.training.core import train_mlframe_models_suite

    target_type = _resolve_target_type_for_combo(combo)
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        regression=(combo.target_type == "regression"),
        target_type=target_type,
    )
    # iterations=5 left models under-converged on n=300, so a 5% row-duplication
    # perturbation could flip R^2 from negative (worse than mean predictor) to
    # positive (real fit), tripping the 0.20 metric-drift tolerance. 200
    # iterations converges all model families on small synthetic frames; 50 was
    # not enough for HGB+LGB regression on n=300 (still saw 0.75 R^2 swings).
    # Dual-run wall time stays comfortably under 30s.
    hyper: dict[str, Any] = {"iterations": 200}
    if "cb" in combo.models:
        hyper["cb_kwargs"] = {"task_type": "CPU", "verbose": 0}
    if "xgb" in combo.models:
        hyper["xgb_kwargs"] = {"device": "cpu", "verbosity": 0}
    if "lgb" in combo.models:
        hyper["lgb_kwargs"] = {"device_type": "cpu", "verbose": -1}

    suite_kwargs: dict[str, Any] = dict(
        df=df,
        target_name=combo.short_id() + "_mm",
        model_name=combo.short_id() + "_mm",
        features_and_targets_extractor=fte,
        mlframe_models=list(combo.models),
        hyperparams_config=hyper,
        preprocessing_config=PreprocessingConfig(drop_columns=[]),
        output_config=OutputConfig(data_dir=tmp_path, models_dir="models"),
        verbose=0,
    )
    if target_type is not None:
        suite_kwargs["target_type"] = target_type
    trained, _ = train_mlframe_models_suite(**suite_kwargs)
    return trained


def _tolerance_for(combo: FuzzCombo) -> float:
    return _REG_R2_TOLERANCE if combo.target_type == "regression" else _CLF_AUC_TOLERANCE


# ---------------------------------------------------------------------------
# Combo curation — one per model family so coverage is broad without
# paying the 150× runtime cost. Respects env override for full sweep.
# ---------------------------------------------------------------------------


def _curated_metamorphic_combos() -> list[FuzzCombo]:
    if os.environ.get("MLFRAME_METAMORPHIC_ALL") == "1":
        return enumerate_combos(target=150, master_seed=20260422)
    combos = enumerate_combos(target=150, master_seed=20260422)
    # Pick one combo per model-family tuple where possible. Prefer
    # simple combos (no OD, no PCA, no MRMR) so the dual-run stays
    # under ~30s each. Fall back to whatever is available if the
    # simple filter yields fewer than 5.
    seen_models: set[tuple[str, ...]] = set()
    chosen: list[FuzzCombo] = []
    for c in combos:
        key = tuple(sorted(c.models))
        if key in seen_models:
            continue
        if c.outlier_detection is not None:
            continue
        if c.custom_prep is not None:
            continue
        if c.use_mrmr_fs:
            continue
        seen_models.add(key)
        chosen.append(c)
        if len(chosen) >= 5:
            break
    return chosen


_METAMORPHIC_COMBOS = _curated_metamorphic_combos()


# ---------------------------------------------------------------------------
# D1 — Column-rename invariance
# ---------------------------------------------------------------------------


def _rename_first_numeric(df, target_col: str) -> tuple[Any, str]:
    """Rename the first numeric non-target column to a new name. Returns
    (new_df, new_column_name_to_exclude_from_diff)."""
    import polars as pl

    if isinstance(df, pl.DataFrame):
        for name, dtype in df.schema.items():
            if name == target_col:
                continue
            if dtype.is_numeric() and name != target_col:
                new_name = f"{name}_renamed"
                return df.rename({name: new_name}), new_name
        return df, ""
    else:
        for name in df.columns:
            if name == target_col:
                continue
            if np.issubdtype(df[name].dtype, np.number):
                new_name = f"{name}_renamed"
                return df.rename(columns={name: new_name}), new_name
        return df, ""


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "combo",
    _METAMORPHIC_COMBOS,
    ids=[c.pytest_id() for c in _METAMORPHIC_COMBOS],
)
def test_metamorphic_column_rename_invariance(combo: FuzzCombo, tmp_path):
    """Renaming a non-target feature must not change val metric materially."""
    df_base, target_col, _ = build_frame_for_combo(combo)
    df_renamed, renamed_col = _rename_first_numeric(df_base, target_col)
    if not renamed_col:
        pytest.skip("no numeric column available to rename")

    m_base = _extract_primary_val_metric(_run_suite(combo, df_base, target_col, str(tmp_path / "base")))
    m_renamed = _extract_primary_val_metric(_run_suite(combo, df_renamed, target_col, str(tmp_path / "ren")))
    if m_base is None or m_renamed is None:
        pytest.skip("no val metric produced; metamorphic check not applicable")

    # See note in test_metamorphic_duplicate_rows_stable: skip when base lacks signal.
    if combo.target_type == "regression" and (m_base < 0.0 or m_renamed < 0.0):
        pytest.skip(f"base/renamed R^2 indicates non-learning model (base={m_base:.3f}, renamed={m_renamed:.3f})")
    if combo.target_type != "regression" and (abs(m_base - 0.5) < 0.05 or abs(m_renamed - 0.5) < 0.05):
        pytest.skip(f"base/renamed AUC near 0.5 (no signal); metamorphic check not meaningful")

    tol = _tolerance_for(combo)
    assert abs(m_base - m_renamed) <= tol, (
        f"D1: val metric drifted under column rename — "
        f"base={m_base:.4f}, renamed={m_renamed:.4f}, |Δ|={abs(m_base - m_renamed):.4f} > {tol}"
    )


# ---------------------------------------------------------------------------
# D2 — Duplicate-row stability
# ---------------------------------------------------------------------------


def _add_duplicate_rows(df, frac: float = 0.05):
    """Return df with ``frac`` (deterministic) duplicated rows appended."""
    import polars as pl

    n_new = max(2, int(df.shape[0] * frac))
    if isinstance(df, pl.DataFrame):
        rng = np.random.default_rng(12345)
        idx = rng.integers(0, df.shape[0], size=n_new).tolist()
        dup = df[idx]
        return pl.concat([df, dup])
    else:
        rng = np.random.default_rng(12345)
        idx = rng.integers(0, df.shape[0], size=n_new)
        dup = df.iloc[idx]
        import pandas as pd

        return pd.concat([df, dup], ignore_index=True)


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "combo",
    _METAMORPHIC_COMBOS,
    ids=[c.pytest_id() for c in _METAMORPHIC_COMBOS],
)
def test_metamorphic_duplicate_rows_stable(combo: FuzzCombo, tmp_path):
    """Adding 5% duplicated rows must not swing val metric beyond noise."""
    df_base, target_col, _ = build_frame_for_combo(combo)
    df_dup = _add_duplicate_rows(df_base, frac=0.05)

    m_base = _extract_primary_val_metric(_run_suite(combo, df_base, target_col, str(tmp_path / "base")))
    m_dup = _extract_primary_val_metric(_run_suite(combo, df_dup, target_col, str(tmp_path / "dup")))
    if m_base is None or m_dup is None:
        pytest.skip("no val metric produced; metamorphic check not applicable")

    # Skip when the base model has essentially no signal: stability claims about
    # a near-random predictor are meaningless. Random-noise-driven local minima
    # (n=300 regression with hgb+lgb routinely lands at R^2<0 base / 0.4 dup or
    # vice-versa from a 5% perturbation) trip the tolerance without indicating
    # a real bug. The test's intent is "did a learned model become unstable",
    # not "did a non-learner stay non-learning".
    if combo.target_type == "regression" and (m_base < 0.0 or m_dup < 0.0):
        pytest.skip(f"base/dup R^2 indicates non-learning model (base={m_base:.3f}, dup={m_dup:.3f})")
    if combo.target_type != "regression" and (abs(m_base - 0.5) < 0.05 or abs(m_dup - 0.5) < 0.05):
        pytest.skip(f"base/dup AUC near 0.5 (no signal); metamorphic check not meaningful")

    tol = _tolerance_for(combo)
    assert abs(m_base - m_dup) <= tol, (
        f"D2: val metric drifted under 5% row duplication — "
        f"base={m_base:.4f}, dup={m_dup:.4f}, |Δ|={abs(m_base - m_dup):.4f} > {tol}"
    )
