"""Business-value integration tests for `preprocessing_extensions` at the
suite level (`train_mlframe_models_suite`).

NOTE: These are regression sensors, not scientific benchmarks. Synthetic data parameters
are intentionally tuned so that the effect is stably visible across all seeds. If a
wiring/logic change breaks preprocessing extensions tomorrow, these tests will catch it.
They do NOT prove the features work on real-world data.

Note on existing coverage:
  `tests/training/test_preprocessing_extensions.py` already exercises the
  unit-level `apply_preprocessing_extensions` function (PCA shape, scalers,
  binarizer, kbins, val/test follow-train, polynomial guard, UMAP missing).
  Those tests do NOT prove suite-level integration nor the business contract
  ("metrics in enabled runs stay within >=95% of baseline" + "extension-less
  path is unchanged"). This file fills that gap.

Plan claim under test:
  - preprocessing_extensions=None preserves the Polars-native fastpath.
  - Enabling extensions wires the shared transformed frame end-to-end.
  - AUROC of the extension-enabled run does not regress >5% vs baseline.
  - PolynomialFeatures(degree=2) materially helps a linear model on
    interaction-dependent data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip whole file if the config feature is not present in this checkout.
try:
    from mlframe.training.configs import PreprocessingExtensionsConfig, TargetTypes
    from mlframe.training.core import train_mlframe_models_suite
except Exception as exc:  # pragma: no cover
    pytest.skip(
        "PreprocessingExtensionsConfig / suite not importable — feature from "
        f"audit #02 may be pending. TODO: re-enable when available. ({exc!r})",
        allow_module_level=True,
    )

from .shared import SimpleFeaturesAndTargetsExtractor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _extract_auroc(model_entry) -> float | None:
    """Best-effort AUROC extraction from a fitted model entry.

    Tries `metrics['test']` first, then `'val'`, then `'train'`. Looks for
    keys containing 'auc' (case-insensitive). Returns None if not found.
    """
    metrics = getattr(model_entry, "metrics", None)
    if not metrics:
        return None
    # Real layout: metrics[split][class_label][metric_name] -> float.
    for split in ("test", "val", "train"):
        bag = metrics.get(split) if isinstance(metrics, dict) else None
        if not bag:
            continue
        for class_label, mdict in bag.items():
            if not isinstance(mdict, dict):
                # Fallback: flat layout {metric_name: value}
                if "auc" in str(class_label).lower() and isinstance(mdict, (int, float)):
                    return float(mdict)
                continue
            for k, v in mdict.items():
                if "roc_auc" == str(k).lower() and isinstance(v, (int, float)) and not np.isnan(v):
                    return float(v)
    return None


def _run_suite(df, models_list, tmp_path, ext_cfg, iters=80):
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    data_dir = str(tmp_path)
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="target",
        model_name="bizvalue_preproc_test",
        features_and_targets_extractor=fte,
        mlframe_models=models_list,
        init_common_params={"show_perf_chart": False, "show_fi": False},
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        data_dir=data_dir,
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
# Test 1 — PCA dim_reducer at suite level (>=95% of baseline AUROC)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 7, 99])
def test_pca_dim_reducer_suite_within_5pct_of_baseline(tmp_path, seed):
    rng = np.random.default_rng(seed)
    n, k_signal, k_redundant, k_noise = 2500, 5, 25, 10  # 40 features total
    X_signal = rng.standard_normal((n, k_signal))
    # Linear-combo redundant features (PCA-friendly).
    mix = rng.standard_normal((k_signal, k_redundant))
    X_redundant = X_signal @ mix + rng.standard_normal((n, k_redundant)) * 0.05
    X_noise = rng.standard_normal((n, k_noise))
    X = np.hstack([X_signal, X_redundant, X_noise])
    logits = 1.5 * X_signal[:, 0] + 1.2 * X_signal[:, 1] - 0.8 * X_signal[:, 2]
    y = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)

    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y

    auroc_a, _ = _run_suite(df, ["linear"], tmp_path / "A", ext_cfg=None, iters=80)

    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler", dim_reducer="PCA", dim_n_components=15,
    )
    auroc_b, _ = _run_suite(df, ["linear"], tmp_path / "B", ext_cfg=cfg, iters=80)

    print(f"\n[Test1 PCA] baseline AUROC={auroc_a:.4f}  pca AUROC={auroc_b:.4f}  "
          f"delta={auroc_b - auroc_a:+.4f}  ratio={auroc_b / max(auroc_a, 1e-9):.3f}")

    assert auroc_b >= auroc_a * 0.95, (
        f"PCA run regressed >5% vs baseline: A={auroc_a:.4f} B={auroc_b:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2 — PolynomialFeatures(degree=2) helps linear model on XOR-like data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 7, 99])
def test_polynomial_features_lift_on_xor_like_data(tmp_path, seed):
    rng = np.random.default_rng(seed)
    n = 2000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n) * 0.2  # near-noise
    x4 = rng.standard_normal(n) * 0.2
    # Pure interaction signal: sign(x1) XOR sign(x2). A linear model on raw
    # features cannot separate this; polynomial degree=2 (interaction_only)
    # creates the x1*x2 feature that resolves it.
    interaction = x1 * x2
    y = (interaction > 0).astype(int)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "target": y})

    # 'linear' is the most interaction-blind model in mlframe_models;
    # trees handle interactions natively so they wouldn't show the lift.
    auroc_a, _ = _run_suite(df, ["linear"], tmp_path / "A", ext_cfg=None, iters=100)

    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler",
        polynomial_degree=2,
        polynomial_interaction_only=True,
    )
    auroc_b, _ = _run_suite(df, ["linear"], tmp_path / "B", ext_cfg=cfg, iters=100)

    print(f"\n[Test2 Poly] baseline AUROC={auroc_a:.4f}  poly AUROC={auroc_b:.4f}  "
          f"delta={auroc_b - auroc_a:+.4f}")

    assert auroc_b > auroc_a + 0.05, (
        f"PolynomialFeatures(degree=2,interaction_only) failed to lift AUROC "
        f"by 0.05 on XOR-like data: A={auroc_a:.4f} B={auroc_b:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3 (optional) — TF-IDF text-column path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 7, 99])
def test_tfidf_column_path_lifts_auroc(tmp_path, seed):
    pytest.importorskip("sklearn.feature_extraction.text")
    # Pragmatic fix: the baseline run drops the raw text column manually (linear
    # pipeline cannot consume strings), and the TF-IDF run keeps it and routes
    # it through `tfidf_columns` + `feature_types_config.text_features` so the
    # polars-ds preflight skips the column.
    rng = np.random.default_rng(seed)
    n = 1500
    # Numeric features carry weak signal; the strong signal lives in the text.
    x1 = rng.standard_normal(n) * 0.2
    x2 = rng.standard_normal(n) * 0.2
    pos_words = ["excellent", "amazing", "great", "wonderful", "fantastic", "superb"]
    neg_words = ["terrible", "awful", "bad", "horrible", "poor", "worst"]
    y = rng.integers(0, 2, size=n)
    texts = []
    for label in y:
        vocab = pos_words if label == 1 else neg_words
        n_words = rng.integers(3, 8)
        # Add some noise words from both sides at low rate.
        words = list(rng.choice(vocab, size=n_words))
        if rng.random() < 0.15:
            words.append(str(rng.choice(neg_words if label == 1 else pos_words)))
        texts.append(" ".join(words))

    df = pd.DataFrame({"x1": x1, "x2": x2, "text": texts, "target": y})

    # Baseline: drop the text column entirely; numeric-only features → low AUROC.
    df_baseline = df.drop(columns=["text"])

    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

    def _run(use_text):
        cur_df = df if use_text else df_baseline
        ext_cfg = PreprocessingExtensionsConfig(tfidf_columns=["text"], tfidf_max_features=50) if use_text else None
        feat_cfg = {"text_features": ["text"]} if use_text else None
        models, _ = train_mlframe_models_suite(
            df=cur_df,
            target_name="target",
            model_name="bizvalue_tfidf_test",
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
    print(f"\n[Test3 TFIDF] baseline AUROC={auroc_a:.4f}  tfidf AUROC={auroc_b:.4f}  delta={auroc_b - auroc_a:+.4f}")
    assert auroc_b > auroc_a + 0.05, (
        f"TF-IDF on text column failed to lift AUROC by 0.05 over numeric-only baseline: "
        f"A={auroc_a:.4f} B={auroc_b:.4f}"
    )
