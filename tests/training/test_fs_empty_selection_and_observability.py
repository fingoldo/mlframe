"""FS empty-selection contract + FS-report observability against REAL fitted selectors.

Two integration concerns the existing suite tests under-cover:

training_integration-07 (empty selection): ``tests/training/test_core.py::test_mrmr_no_impact_classification``
pins the empty-selection path with a single vacuous ``assert isinstance(models, dict)``. Worse, the forcing
it uses (``min_relevance_gain=10.0`` under the DEFAULT ``min_relevance_gain_mode='relative_to_entropy'``) does
NOT actually empty selection -- the relative mode rescales the floor to ``min_relevance_gain_frac * H(y)`` and
ignores the absolute 10.0. Even when screening returns 0 features, MRMR's ``min_features_fallback`` (default 1)
back-fills one raw feature, so a model still trains. This module forces a genuinely empty selection
(``min_relevance_gain_mode='absolute'`` + a high floor + ``min_features_fallback=0``) and asserts the documented
degrade behaviour: the MRMR-branch ``selected_features`` surface is absent/empty, and with
``use_ordinary_models=False`` no model entry is trained AND the prod WARNING fires.

training_integration-09 (FS-report observability): ``tests/training/test_metadata_feature_selection_report_observability.py``
tests ``_build_feature_selection_report`` only against hand-rolled duck-typed STUB selectors whose attribute shapes
the test itself authored. Those stubs cannot catch the day a REAL MRMR / RFECV's fitted attribute surface drifts
from what the report builder reads. This module fits REAL ``MRMR`` (simple-mode) and REAL ``RFECV`` selectors and
asserts the report's ``kept_features`` / ``dropped_features`` / ``reason_per_feature`` / ``scores`` against the
genuine fitted state -- no skip-on-exception escape hatch, no vacuous ``None``-in-kinds assert.

All heavy paths are CPU-forced (the module sets ``CUDA_VISIBLE_DEVICES=''`` at import) and sized to finish well
under the suite's ~60s/test budget (n<=200, 6 features, ``lgb`` with cv defaults, single seed).
"""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd
import pytest

# CPU-only: forces CatBoost / cupy off the GPU so the suite call cannot trip the native GPU crash.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training import FeatureSelectionConfig
from mlframe.training.core._phase_train_one_target import _build_feature_selection_report
from tests.training.shared import SimpleFeaturesAndTargetsExtractor

try:
    from tests.conftest import is_fast_mode
except ImportError:  # pragma: no cover

    def is_fast_mode() -> bool:
        """Fallback used when tests.conftest isn't importable: always run the full (non-fast) suite."""
        return False


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def noise_binary_6feat():
    """200 rows x 6 features, target INDEPENDENT of every feature (pure noise).

    No feature clears any reasonable relevance floor, so MRMR screening returns 0 and -- with the
    fallback disabled -- selects nothing. This is the genuine empty-selection driver.
    """
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 6)).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = rng.integers(0, 2, n).astype(np.int8)  # independent of X
    return df


@pytest.fixture
def informative_binary_6feat():
    """200 rows x 6 features; only f0 / f1 carry signal, f2..f5 are pure noise.

    A real selector should KEEP a small informative subset and DROP the noise -- giving non-empty
    kept_features AND non-empty dropped_features for the observability assertions.
    """
    rng = np.random.default_rng(7)
    n = 200
    X = rng.standard_normal((n, 6)).astype(np.float64)
    logits = 2.5 * X[:, 0] - 2.0 * X[:, 1] + 0.05 * rng.standard_normal(n)
    y = (logits > 0).astype(np.int64)
    cols = [f"f{i}" for i in range(6)]
    df = pd.DataFrame(X, columns=cols)
    return df, cols, y


def _extract_metadata(result):
    """Normalise the suite return (tuple / ctx / dict) to the metadata dict carrying model_schemas."""
    if isinstance(result, tuple):
        for slot in result:
            if isinstance(slot, dict) and "model_schemas" in slot:
                return slot
        for slot in result:
            if isinstance(slot, dict) and ("selected_features" in slot or "selected_features_per_model" in slot):
                return slot
    if hasattr(result, "metadata"):
        return result.metadata
    if isinstance(result, dict):
        return result
    return None


def _extract_models(result):
    """Pulls the models dict out of a suite result, whichever tuple/object shape it was returned as."""
    if isinstance(result, tuple):
        for slot in result:
            if isinstance(slot, dict) and any(hasattr(k, "value") or isinstance(k, str) for k in slot) and "model_schemas" not in slot:
                return slot
        return result[0] if result else None
    return getattr(result, "models", None)


# ----------------------------------------------------------------------------
# (a) Empty-selection contract (training_integration-07)
# ----------------------------------------------------------------------------


def _empty_selection_fs_config():
    """FeatureSelectionConfig that genuinely empties MRMR selection.

    ``min_relevance_gain_mode='absolute'`` is REQUIRED: the default ``'relative_to_entropy'`` rescales the
    floor to a fraction of H(y) and ignores the verbatim ``min_relevance_gain``, so a 10.0 floor would be
    silently ineffective. ``min_features_fallback=0`` disables the back-fill that otherwise keeps one raw
    feature even when screening returns nothing.
    """
    return FeatureSelectionConfig(
        use_mrmr_fs=True,
        mrmr_kwargs={
            "verbose": 0,
            "use_simple_mode": True,
            "max_runtime_mins": 0.3,
            "min_relevance_gain_mode": "absolute",
            "min_relevance_gain": 10.0,
            "min_features_fallback": 0,
        },
    )


@pytest.mark.slow
def test_empty_selection_clears_mrmr_branch_selected_features(noise_binary_6feat, temp_data_dir):
    """Forcing MRMR to select 0 features leaves the MRMR-branch selected-features surface absent or empty.

    The suite stamps ``metadata['selected_features']`` / ``['selected_features_per_model']`` ONLY when at
    least one model trained with a non-empty column set. A genuinely empty MRMR selection (no fallback) must
    therefore leave that surface absent OR empty -- the assertion that the forcing actually worked.
    """
    df = noise_binary_6feat
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    from mlframe.training import OutputConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="empty_sel_mrmr",
            features_and_targets_extractor=fte,
            mlframe_models=["lgb"],
            feature_selection_config=_empty_selection_fs_config(),
            use_ordinary_models=False,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

    metadata = _extract_metadata(result)
    assert isinstance(metadata, dict), f"expected a metadata dict from the suite, got {type(result)}"

    sel = metadata.get("selected_features")
    sel_per_model = metadata.get("selected_features_per_model")
    # The MRMR-branch selected-features surface must be absent OR empty. If a non-empty list survives,
    # the forcing no longer empties selection (a fallback / floor-mode regression) and the test below
    # would be silently meaningless -- fail loudly with the diagnostic.
    forced_empty = ("selected_features" not in metadata) or (not sel) or (sel_per_model is not None and not sel_per_model)
    assert forced_empty, (
        "forcing no longer empties selection: "
        f"selected_features={sel!r}, selected_features_per_model={sel_per_model!r}. "
        "min_relevance_gain_mode='absolute' + min_features_fallback=0 must yield an empty MRMR selection."
    )


@pytest.mark.slow
def test_empty_selection_degrades_to_no_trained_entry_and_warns(noise_binary_6feat, temp_data_dir, caplog):
    """With ``use_ordinary_models=False`` an empty MRMR selection trains NO model AND emits the prod WARNING.

    Documented degrade behaviour: when FS removes every feature and there is no ordinary (no-FS) branch to
    fall back to, the suite must (1) NOT return a trained entry for the target and (2) log the prod warning
    so the operator sees WHY nothing trained. A missing log line is a small prod observability gap -- flagged
    in the assertion message rather than silently tolerated.
    """
    df = noise_binary_6feat
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    from mlframe.training import OutputConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with caplog.at_level(logging.WARNING):
            result = train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name="empty_sel_degrade",
                features_and_targets_extractor=fte,
                mlframe_models=["lgb"],
                feature_selection_config=_empty_selection_fs_config(),
                use_ordinary_models=False,
                use_mlframe_ensembles=False,
                output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
                verbose=0,
            )

    models = _extract_models(result)
    assert isinstance(models, dict), f"expected a models dict, got {type(models)}"

    # No trained entry for the target: every target bucket must carry zero entries.
    total_entries = 0
    for by_name in models.values():
        if isinstance(by_name, dict):
            for entries in by_name.values():
                if isinstance(entries, list):
                    total_entries += len(entries)
    assert (
        total_entries == 0
    ), f"expected NO trained entries when MRMR empties selection and use_ordinary_models=False; got {total_entries} entry/entries: {models!r}"

    # The prod WARNING at _trainer_train_and_evaluate.py:421 explains the empty-selection skip.
    _msgs = "\n".join(r.getMessage() for r in caplog.records)
    _has_skip_log = ("removed all features" in _msgs) or ("skipping training" in _msgs.lower())
    assert _has_skip_log, (
        "PROD OBSERVABILITY GAP: empty FS selection trained no model but no "
        "'removed all features ... skipping training' WARNING was logged. Captured WARN+ records:\n" + (_msgs or "<none>")
    )


# ----------------------------------------------------------------------------
# (b) FS-report observability vs REAL fitted selectors (training_integration-09)
# ----------------------------------------------------------------------------


def test_report_against_real_fitted_mrmr_surfaces_kept_dropped_reasons(informative_binary_6feat):
    """A REAL fitted MRMR drives ``_build_feature_selection_report`` to non-empty kept/dropped + full reason cover.

    Unlike the stub-selector tests, this fits the actual ``MRMR`` estimator and reads ITS post-fit ``support_``
    / ``feature_names_in_``. If MRMR's fitted attribute surface drifts from what the report builder reads, these
    assertions fail -- the stub tests cannot catch that.
    """
    from mlframe.feature_selection.filters import MRMR

    df, cols, y = informative_binary_6feat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(use_simple_mode=True, verbose=0, max_runtime_mins=0.5)
        sel.fit(df, pd.Series(y, name="target"))

    support = list(getattr(sel, "support_"))
    assert support, "real MRMR selected zero features on a clearly-informative fixture (f0/f1 carry signal)"
    kept = [cols[i] for i in support]

    report = _build_feature_selection_report(
        pre_pipeline=sel,
        pre_pipeline_name="MRMR ",
        fitted_columns_in=cols,
        kept_columns=kept,
    )

    assert report["selector_name"] == "MRMR"
    assert report["selector_params_hash"] is not None
    assert report["kept_features"], "kept_features must be non-empty for a real MRMR that selected features"
    assert set(report["kept_features"]) == set(kept)
    assert report["dropped_features"], "dropped_features must be non-empty (noise columns f2..f5 are dropped)"
    # kept and dropped partition the fitted input columns.
    assert set(report["kept_features"]) | set(report["dropped_features"]) == set(cols)
    assert set(report["kept_features"]).isdisjoint(report["dropped_features"])

    reasons = report["reason_per_feature"]
    assert isinstance(reasons, dict) and reasons, "MRMR report must carry a per-feature reason map"
    assert set(reasons) == set(cols), f"reason_per_feature must cover every input column; missing {set(cols) - set(reasons)}, extra {set(reasons) - set(cols)}"
    for c in cols:
        assert reasons[c] in ("kept", "dropped")
        assert reasons[c] == ("kept" if c in set(kept) else "dropped")


def test_report_against_real_fitted_rfecv_surfaces_kept_dropped(informative_binary_6feat):
    """A REAL fitted RFECV drives ``_build_feature_selection_report`` to non-empty kept/dropped.

    RFECV is forced to a 2-feature target (``max_nfeatures=2``) so it genuinely eliminates the noise columns,
    giving a non-empty dropped set. ``cv=2``, ``max_refits=2`` keep it well inside the per-test budget.
    """
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection.wrappers import RFECV

    df, cols, y = informative_binary_6feat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = RFECV(
            estimator=LogisticRegression(max_iter=200),
            cv=2,
            max_refits=2,
            max_nfeatures=2,
            verbose=0,
        )
        sel.fit(df, y)

    support = list(getattr(sel, "support_"))
    kept = [c for c, s in zip(cols, support) if s]
    assert kept, "real RFECV kept zero features on an informative fixture"
    assert len(kept) < len(cols), "RFECV with max_nfeatures=2 must drop at least one column"

    report = _build_feature_selection_report(
        pre_pipeline=sel,
        pre_pipeline_name="lgb ",
        fitted_columns_in=cols,
        kept_columns=kept,
    )

    assert report["selector_name"] == "RFECV"
    assert report["selector_params_hash"] is not None
    assert report["kept_features"], "kept_features must be non-empty for a real RFECV"
    assert set(report["kept_features"]) == set(kept)
    assert report["dropped_features"], "dropped_features must be non-empty (RFECV eliminated noise columns)"
    assert set(report["kept_features"]) | set(report["dropped_features"]) == set(cols)
    assert set(report["kept_features"]).isdisjoint(report["dropped_features"])


def test_report_against_real_fitted_rfecv_surfaces_scores(informative_binary_6feat):
    """A REAL fitted RFECV surfaces a non-empty per-feature ``scores`` dict in the report.

    RFECV exposes per-fold ``{feature: importance}`` maps under ``feature_importances_``, so the report's
    ``scores`` must be a non-empty ``{feature: float}`` map aggregated across folds.
    """
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection.wrappers import RFECV

    df, cols, y = informative_binary_6feat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = RFECV(
            estimator=LogisticRegression(max_iter=200),
            cv=2,
            max_refits=2,
            max_nfeatures=2,
            verbose=0,
        )
        sel.fit(df, y)

    support = list(getattr(sel, "support_"))
    kept = [c for c, s in zip(cols, support) if s]

    report = _build_feature_selection_report(
        pre_pipeline=sel,
        pre_pipeline_name="lgb ",
        fitted_columns_in=cols,
        kept_columns=kept,
    )

    scores = report["scores"]
    assert (
        isinstance(scores, dict) and scores
    ), "RFECV report['scores'] must be a non-empty {feature: float} dict built from the selector's per-fold feature importances"
    assert all(isinstance(v, float) for v in scores.values())


def test_report_rfecv_scores_read_dict_valued_feature_importances_not_ndarray(informative_binary_6feat):
    """Regression: RFECV report scores must read the dict-of-dicts ``feature_importances_`` surface.

    The real fitted RFECV stores ``feature_importances_`` as ``{"<n>_<fold>": {feature: score}}``. The
    pre-fix builder coerced those per-fold DICTS through ``np.asarray(..., float64)`` (expecting ndarrays
    aligned to ``feature_names_in_``), which silently excepted and left ``scores=None`` on every real fit.
    Pins: the fitted attribute IS dict-of-dicts, and the report aggregates a float score for the kept
    informative features (f0/f1) from it.
    """
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection.wrappers import RFECV

    df, cols, y = informative_binary_6feat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = RFECV(
            estimator=LogisticRegression(max_iter=200),
            cv=2,
            max_refits=2,
            max_nfeatures=2,
            verbose=0,
        )
        sel.fit(df, y)

    fi = getattr(sel, "feature_importances_")
    assert isinstance(fi, dict) and fi
    assert all(isinstance(row, dict) for row in fi.values()), "fixture invariant drifted: RFECV.feature_importances_ values are no longer per-fold dicts"

    support = list(getattr(sel, "support_"))
    kept = [c for c, s in zip(cols, support) if s]
    report = _build_feature_selection_report(
        pre_pipeline=sel,
        pre_pipeline_name="lgb ",
        fitted_columns_in=cols,
        kept_columns=kept,
    )

    scores = report["scores"]
    assert isinstance(scores, dict) and scores, "report['scores'] None means the dict-valued FI surface was not read"
    for c in kept:
        assert c in scores and isinstance(scores[c], float)
