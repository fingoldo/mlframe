"""Suite-level business-value tests for `preprocessing_extensions`.

NOTE: These are regression sensors, not scientific benchmarks. Synthetic data parameters
are intentionally tuned so that the effect is stably visible across all seeds. If a
wiring/logic change breaks preprocessing extensions tomorrow, these tests will catch it.
They do NOT prove the features work on real-world data.

Companion to `test_bizvalue_preproc_extensions.py` — that file establishes the
baseline contract on a single seed/dataset shape; this file adds independent
suite-level evidence under different seeds and stricter model-count checks so
a regression that happens to pass on one fixture is still caught here.

Coverage gap motivating this file:
  Existing `test_preprocessing_extensions.py` is unit-level only (calls
  `apply_preprocessing_extensions` directly with arrays). Suite-level wiring
  (preprocessing_extensions plumbed through `train_mlframe_models_suite` →
  `fit_and_transform_pipeline` → per-model strategy) is exercised here.

Tests:
  1. PCA dim_reducer end-to-end at suite level: enabled run AUROC stays within
     5% of the baseline AND suite returns >=1 fitted model entry.
  2. PolynomialFeatures(degree=2, interaction_only=True) materially lifts a
     linear classifier on XOR-like interaction-only signal (>=0.05 AUROC lift).
  3. TF-IDF column path — currently skipped: the FTE used by the suite does
     not declare/preserve raw text columns through the Polars-native preflight,
     so the baseline-without-tfidf run would crash on the string column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip whole module if config feature isn't present (audit #02 plan partially
# pending). Do NOT invent imports.
try:
    from mlframe.training.configs import PreprocessingExtensionsConfig, TargetTypes
    from mlframe.training.core import train_mlframe_models_suite
except Exception as exc:  # pragma: no cover
    pytest.skip(
        "PreprocessingExtensionsConfig / suite not importable — audit #02 "
        f"feature may be pending. TODO: re-enable when available. ({exc!r})",
        allow_module_level=True,
    )

from .shared import SimpleFeaturesAndTargetsExtractor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_auc_in(bag) -> float | None:
    """Recursively locate any roc_auc / auc-style float value in a metrics dict."""
    if isinstance(bag, dict):
        # Prefer roc_auc-named keys at this level.
        for k, v in bag.items():
            if isinstance(v, (int, float)) and "auc" in str(k).lower() and "pr" not in str(k).lower():
                if not np.isnan(v):
                    return float(v)
        # Otherwise recurse one level deeper (per-class dicts: {1: {...}}).
        for v in bag.values():
            found = _find_auc_in(v)
            if found is not None:
                return found
    return None


def _extract_auroc(model_entry) -> float | None:
    """Best-effort AUROC pull from a fitted model entry (test → val → train)."""
    metrics = getattr(model_entry, "metrics", None)
    if not metrics or not isinstance(metrics, dict):
        return None
    for split in ("test", "val", "train"):
        bag = metrics.get(split)
        if not bag:
            continue
        found = _find_auc_in(bag)
        if found is not None:
            return found
    return None


def _run_suite(df, models_list, tmp_path, ext_cfg, iters=80):
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    models, _metadata = train_mlframe_models_suite(
        df=df,
        target_name="target",
        model_name="bizvalue_preproc_suite",
        features_and_targets_extractor=fte,
        mlframe_models=models_list,
        init_common_params={"show_perf_chart": False, "show_fi": False},
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        data_dir=str(tmp_path),
        models_dir="models",
        verbose=0,
        hyperparams_config={"iterations": iters},
        preprocessing_extensions=ext_cfg,
    )
    entries = models[TargetTypes.BINARY_CLASSIFICATION]["target"]
    assert len(entries) >= 1, "Suite produced no model entries"
    aurocs = [a for a in (_extract_auroc(e) for e in entries) if a is not None]
    assert aurocs, "No AUROC metric found on any model entry"
    return max(aurocs), entries


# ---------------------------------------------------------------------------
# Test 1 — PCA dim_reducer ≥95% baseline metric (independent seed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 7, 99])
def test_suite_pca_dim_reducer_within_5pct_of_baseline(tmp_path, seed):
    rng = np.random.default_rng(seed)  # seed-parametrized; companion file uses same set
    n, k_signal, k_redundant, k_noise = 2500, 6, 24, 10  # 40 features total
    X_signal = rng.standard_normal((n, k_signal))
    mix = rng.standard_normal((k_signal, k_redundant))
    X_redundant = X_signal @ mix + rng.standard_normal((n, k_redundant)) * 0.05
    X_noise = rng.standard_normal((n, k_noise))
    X = np.hstack([X_signal, X_redundant, X_noise])
    logits = (
        1.4 * X_signal[:, 0]
        - 1.1 * X_signal[:, 1]
        + 0.7 * X_signal[:, 2]
        - 0.5 * X_signal[:, 3]
    )
    y = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)

    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y

    auroc_a, entries_a = _run_suite(df, ["linear"], tmp_path / "A", ext_cfg=None, iters=80)

    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler",
        dim_reducer="PCA",
        dim_n_components=15,
    )
    auroc_b, entries_b = _run_suite(df, ["linear"], tmp_path / "B", ext_cfg=cfg, iters=80)

    print(
        f"\n[SuiteTest1 PCA] baseline AUROC={auroc_a:.4f}  "
        f"pca AUROC={auroc_b:.4f}  delta={auroc_b - auroc_a:+.4f}  "
        f"ratio={auroc_b / max(auroc_a, 1e-9):.3f}  "
        f"models_a={len(entries_a)} models_b={len(entries_b)}"
    )

    assert len(entries_b) >= 1, "Enabled-extension run produced no fitted models"
    assert auroc_b >= auroc_a * 0.95, (
        f"PCA suite run regressed >5% vs baseline: A={auroc_a:.4f} B={auroc_b:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Polynomial features lift linear model on XOR-like data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 7, 99])
def test_suite_polynomial_features_lift_on_xor(tmp_path, seed):
    rng = np.random.default_rng(seed)  # seed-parametrized
    n = 2000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n) * 0.2  # near-noise
    x4 = rng.standard_normal(n) * 0.2
    # Pure interaction signal — sign(x1*x2). Linear-on-raw cannot separate.
    interaction = x1 * x2
    y = (interaction > 0).astype(int)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "target": y})

    auroc_a, _ = _run_suite(df, ["linear"], tmp_path / "A", ext_cfg=None, iters=100)

    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler",
        polynomial_degree=2,
        polynomial_interaction_only=True,
    )
    auroc_b, _ = _run_suite(df, ["linear"], tmp_path / "B", ext_cfg=cfg, iters=100)

    print(
        f"\n[SuiteTest2 Poly] baseline AUROC={auroc_a:.4f}  "
        f"poly AUROC={auroc_b:.4f}  delta={auroc_b - auroc_a:+.4f}"
    )

    assert auroc_b > auroc_a + 0.05, (
        f"PolynomialFeatures(degree=2, interaction_only) failed to lift AUROC "
        f"by 0.05 on XOR-like data: A={auroc_a:.4f} B={auroc_b:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — TF-IDF column path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 7, 99])
def test_suite_tfidf_column_path_lifts_auroc(tmp_path, seed):
    pytest.importorskip("sklearn.feature_extraction.text")
    # Pragmatic fix: baseline drops the text column manually; TF-IDF run keeps
    # it and routes via `tfidf_columns` + `feature_types_config.text_features`.
    rng = np.random.default_rng(seed)
    n = 1500
    x1 = rng.standard_normal(n) * 0.2
    x2 = rng.standard_normal(n) * 0.2
    pos_words = ["excellent", "amazing", "great", "wonderful", "fantastic", "superb"]
    neg_words = ["terrible", "awful", "bad", "horrible", "poor", "worst"]
    y = rng.integers(0, 2, size=n)
    texts = []
    for label in y:
        vocab = pos_words if label == 1 else neg_words
        n_words = rng.integers(3, 8)
        words = list(rng.choice(vocab, size=n_words))
        if rng.random() < 0.15:
            words.append(str(rng.choice(neg_words if label == 1 else pos_words)))
        texts.append(" ".join(words))
    df = pd.DataFrame({"x1": x1, "x2": x2, "text": texts, "target": y})
    df_baseline = df.drop(columns=["text"])
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

    def _run(use_text):
        cur_df = df if use_text else df_baseline
        ext_cfg = PreprocessingExtensionsConfig(tfidf_columns=["text"], tfidf_max_features=50) if use_text else None
        feat_cfg = {"text_features": ["text"]} if use_text else None
        models, _ = train_mlframe_models_suite(
            df=cur_df,
            target_name="target",
            model_name="bizvalue_tfidf_suite",
            features_and_targets_extractor=fte,
            mlframe_models=["linear"],
            init_common_params={"show_perf_chart": False, "show_fi": False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=str(tmp_path / ("B" if use_text else "A")),
            models_dir="models",
            verbose=0,
            hyperparams_config={"iterations": 100},
            preprocessing_extensions=ext_cfg,
            feature_types_config=feat_cfg,
        )
        entries = models[TargetTypes.BINARY_CLASSIFICATION]["target"]
        aurocs = [a for a in (_extract_auroc(e) for e in entries) if a is not None]
        return max(aurocs)

    auroc_a = _run(use_text=False)
    auroc_b = _run(use_text=True)
    print(f"\n[SuiteTest3 TFIDF] baseline AUROC={auroc_a:.4f}  tfidf AUROC={auroc_b:.4f}  delta={auroc_b - auroc_a:+.4f}")
    assert auroc_b > auroc_a + 0.05, (
        f"TF-IDF on text column failed to lift AUROC by 0.05 over numeric-only baseline: "
        f"A={auroc_a:.4f} B={auroc_b:.4f}"
    )
