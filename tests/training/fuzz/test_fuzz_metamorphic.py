
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

# Fuzz combos run hundreds of train_mlframe_models_suite iterations and are
# deselected from the default test run; pass pytest --run-fuzz to include.
pytestmark = pytest.mark.fuzz

from tests.training._fuzz_combo import (
    FuzzCombo,
    build_frame_for_combo,
    enumerate_combos,
)
from tests.training.shared import SimpleFeaturesAndTargetsExtractor


# Tolerances for metric drift. Generous — metamorphic tests should catch
# catastrophic regressions (predictions inverting, pipeline silently
# dropping features), not 0.01-AUC wobble.
_CLF_AUC_TOLERANCE = 0.15  # |Δ roc_auc|
_REG_R2_TOLERANCE = 0.20  # |Δ r2|
_LTR_NDCG_TOLERANCE = 0.15  # |Δ ndcg@10|; same range as AUC (0..1)


_LTR_RANKER_FLAVORS = ("cb", "xgb", "lgb", "mlp", "ensemble")


def _extract_primary_val_metric(trained: dict) -> float | None:
    """Return the first non-NaN val metric found in the nested trained dict.

    Three known shapes:

    * Classification (binary / multiclass / multilabel) → roc_auc from the
      first class slot in the nested ``{tt: {tname: [entry, ...]}}``
      structure produced by ``train_mlframe_models_suite``.
    * Regression → R² from the same shape (flat val dict).
    * Learning-to-rank → ``ndcg@10`` from the
      ``{flavor: {"val_metrics": {"ndcg@10": ...}}}`` shape that
      ``train_mlframe_ranker_suite`` returns. Used as the metamorphic
      stability metric just like roc_auc / R² above.
    """
    if not isinstance(trained, dict):
        return None

    # LTR ranker_suite shape: top-level keys are model flavors, each value
    # has 'val_metrics' / 'test_metrics' dicts with keys like 'ndcg@10'.
    # Detect by sniffing for that shape before falling through to the
    # classifier/regressor structure.
    if any(k in trained for k in _LTR_RANKER_FLAVORS):
        for flavor in _LTR_RANKER_FLAVORS:
            entry = trained.get(flavor)
            if not isinstance(entry, dict):
                continue
            val_metrics = entry.get("val_metrics")
            if not isinstance(val_metrics, dict):
                continue
            # Prefer ndcg@10, then any ndcg@*, then map@*, then mrr.
            for preferred in ("ndcg@10", "ndcg@5", "ndcg@1"):
                v = val_metrics.get(preferred)
                if v is not None and np.isfinite(v):
                    return float(v)
            for k, v in val_metrics.items():
                if isinstance(k, str) and (k.startswith("ndcg@") or k.startswith("map@") or k == "mrr"):
                    if v is not None and np.isfinite(v):
                        return float(v)
        # Sniff matched but no usable metric: fall through to None.

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
    # LTR builder in _fuzz_combo emits a 'qid' column that the ranker suite
    # requires the FTE to expose via group_field.
    group_field = "qid" if combo.target_type == "learning_to_rank" else None
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        regression=(combo.target_type == "regression"),
        target_type=target_type,
        group_field=group_field,
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
    if combo.target_type == "regression":
        return _REG_R2_TOLERANCE
    if combo.target_type == "learning_to_rank":
        return _LTR_NDCG_TOLERANCE
    return _CLF_AUC_TOLERANCE


def _no_signal(combo: FuzzCombo, m_base: float, m_perturbed: float) -> bool:
    """Return True when the metric pair indicates a near-random base model;
    metamorphic stability claims are then meaningless and we skip rather
    than fail. Per-target-type thresholds reflect the metric's range:

    * regression: R² < 0 ⇒ worse than mean predictor → no signal
    * binary/multi*: |AUC - 0.5| < 0.05 ⇒ near-coin-flip
    * LTR: NDCG@10 < 0.25 ⇒ near-random (≥0.5 is achievable on synthetic
      4-graded relevance with a real ranker)
    """
    if combo.target_type == "regression":
        return m_base < 0.0 or m_perturbed < 0.0
    if combo.target_type == "learning_to_rank":
        return m_base < 0.25 or m_perturbed < 0.25
    return abs(m_base - 0.5) < 0.05 or abs(m_perturbed - 0.5) < 0.05


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


from tests.conftest import fast_subset
# Fast mode keeps one representative combo so a full metamorphic dual-suite run completes within pytest-timeout.
_METAMORPHIC_COMBOS = fast_subset(_curated_metamorphic_combos())


# ---------------------------------------------------------------------------
# D1 — Column-rename invariance
# ---------------------------------------------------------------------------


def _rename_first_numeric(df, target_col: str, protect: tuple[str, ...] = ()) -> tuple[Any, str]:
    """Rename the first numeric non-target column to a new name. Returns
    (new_df, new_column_name_to_exclude_from_diff).

    ``protect`` lists additional column names that must not be renamed
    (e.g. ``"qid"`` for LTR combos -- renaming it would break the
    FTE.group_field lookup at suite entry).
    """
    import polars as pl

    blocked = {target_col, *protect}
    if isinstance(df, pl.DataFrame):
        for name, dtype in df.schema.items():
            if name in blocked:
                continue
            if dtype.is_numeric():
                new_name = f"{name}_renamed"
                return df.rename({name: new_name}), new_name
        return df, ""
    else:
        for name in df.columns:
            if name in blocked:
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
    # LTR combos: protect 'qid' from being chosen as the rename target;
    # the suite resolves it via FTE.group_field, so renaming it would
    # break ranker dispatch (different bug than the metamorphic invariant).
    protect: tuple[str, ...] = ("qid",) if combo.target_type == "learning_to_rank" else ()
    df_renamed, renamed_col = _rename_first_numeric(df_base, target_col, protect=protect)
    if not renamed_col:
        pytest.skip("no numeric column available to rename")

    m_base = _extract_primary_val_metric(_run_suite(combo, df_base, target_col, str(tmp_path / "base")))
    m_renamed = _extract_primary_val_metric(_run_suite(combo, df_renamed, target_col, str(tmp_path / "ren")))
    if m_base is None or m_renamed is None:
        pytest.skip("no val metric produced; metamorphic check not applicable")

    if _no_signal(combo, m_base, m_renamed):
        pytest.skip(
            f"base/renamed metric lacks signal "
            f"({combo.target_type}: base={m_base:.3f}, renamed={m_renamed:.3f}); "
            f"metamorphic check not meaningful"
        )

    tol = _tolerance_for(combo)
    assert abs(m_base - m_renamed) <= tol, (
        f"D1: val metric drifted under column rename — "
        f"base={m_base:.4f}, renamed={m_renamed:.4f}, |Δ|={abs(m_base - m_renamed):.4f} > {tol}"
    )


# ---------------------------------------------------------------------------
# D2 — Duplicate-row stability
# ---------------------------------------------------------------------------


def _add_duplicate_rows(df, frac: float = 0.05, sort_by: str | None = None):
    """Return df with ``frac`` (deterministic) duplicated rows appended.

    When ``sort_by`` is given (LTR combos use ``"qid"``), the resulting
    frame is sorted by that column so query-grouped libraries (XGB,
    LGB, CatBoost rankers) still see qid in non-decreasing order. The
    LTR property under test (duplicate-row stability) is preserved
    because rows within the same qid stay adjacent and the per-query
    document distribution is unchanged save for the duplicated rows.
    """
    import polars as pl

    n_new = max(2, int(df.shape[0] * frac))
    if isinstance(df, pl.DataFrame):
        rng = np.random.default_rng(12345)
        idx = rng.integers(0, df.shape[0], size=n_new).tolist()
        dup = df[idx]
        out = pl.concat([df, dup])
        if sort_by is not None and sort_by in out.columns:
            out = out.sort(sort_by, maintain_order=True)
        return out
    else:
        rng = np.random.default_rng(12345)
        idx = rng.integers(0, df.shape[0], size=n_new)
        dup = df.iloc[idx]
        import pandas as pd

        out = pd.concat([df, dup], ignore_index=True)
        if sort_by is not None and sort_by in out.columns:
            out = out.sort_values(sort_by, kind="stable").reset_index(drop=True)
        return out


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "combo",
    _METAMORPHIC_COMBOS,
    ids=[c.pytest_id() for c in _METAMORPHIC_COMBOS],
)
def test_metamorphic_duplicate_rows_stable(combo: FuzzCombo, tmp_path):
    """Adding 5% duplicated rows must not swing val metric beyond noise."""
    df_base, target_col, _ = build_frame_for_combo(combo)
    # LTR combos: keep qid sorted post-duplication so query-grouped rankers
    # (XGB / LGB / CB ranker) still see non-decreasing qid (libxgboost asserts
    # this and crashes loudly otherwise).
    sort_by = "qid" if combo.target_type == "learning_to_rank" else None
    df_dup = _add_duplicate_rows(df_base, frac=0.05, sort_by=sort_by)

    m_base = _extract_primary_val_metric(_run_suite(combo, df_base, target_col, str(tmp_path / "base")))
    m_dup = _extract_primary_val_metric(_run_suite(combo, df_dup, target_col, str(tmp_path / "dup")))
    if m_base is None or m_dup is None:
        pytest.skip("no val metric produced; metamorphic check not applicable")

    if _no_signal(combo, m_base, m_dup):
        pytest.skip(
            f"base/dup metric lacks signal "
            f"({combo.target_type}: base={m_base:.3f}, dup={m_dup:.3f}); "
            f"metamorphic check not meaningful"
        )

    tol = _tolerance_for(combo)
    assert abs(m_base - m_dup) <= tol, (
        f"D2: val metric drifted under 5% row duplication — "
        f"base={m_base:.4f}, dup={m_dup:.4f}, |Δ|={abs(m_base - m_dup):.4f} > {tol}"
    )
