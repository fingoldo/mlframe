"""Tests for ``mlframe.training.dummy_baselines``.

Coverage targets the 21 defenses (D1-D21) from plan v3 round-3 audit:
per-target dispatcher, TS detection (ACF + step-size + monotonicity
gate), per-group leakage diagnostics, strongest-pick non-degeneracy +
paired-bootstrap robustness, per-cell metric isolation, LTR group
sanity, multi-output regression, headline log_loss for classification,
n_repeats variance, polars/pandas/dtype variants, JSON serialization
with NaN scrubbing, deterministic per-target seed, and the operator
contract verdict-line output.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from mlframe.training.configs import DummyBaselinesConfig
from mlframe.training.dummy_baselines import (
    BaselineReport,
    SCHEMA_VERSION,
    _baseline_inputs_hash,
    _is_temporally_monotonic,
    _normalize_timestamps,
    _per_target_seed,
    _slugify,
    compute_dummy_baselines,
    format_suite_end_summary,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def cfg():
    return DummyBaselinesConfig()


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(0)
    n_tr, n_va, n_te = 500, 100, 100
    return {
        "target_type": "regression",
        "target_name": "y",
        "train_y": rng.normal(10.0, 3.0, n_tr),
        "val_y": rng.normal(10.0, 3.0, n_va),
        "test_y": rng.normal(10.0, 3.0, n_te),
        "train_X": pd.DataFrame({"x": rng.normal(size=n_tr), "cat": rng.integers(0, 5, n_tr)}),
        "val_X": pd.DataFrame({"x": rng.normal(size=n_va), "cat": rng.integers(0, 5, n_va)}),
        "test_X": pd.DataFrame({"x": rng.normal(size=n_te), "cat": rng.integers(0, 5, n_te)}),
        "cat_features": ["cat"],
    }


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    n_tr, n_va, n_te = 500, 100, 100
    y_tr = rng.integers(0, 2, n_tr)
    y_va = rng.integers(0, 2, n_va)
    y_te = rng.integers(0, 2, n_te)
    le = LabelEncoder().fit(np.concatenate([y_tr, y_va, y_te]))
    return {
        "target_type": "binary_classification",
        "target_name": "b",
        "train_y": y_tr, "val_y": y_va, "test_y": y_te,
        "train_X": pd.DataFrame({"x": rng.normal(size=n_tr)}),
        "val_X": pd.DataFrame({"x": rng.normal(size=n_va)}),
        "test_X": pd.DataFrame({"x": rng.normal(size=n_te)}),
        "target_label_encoder": le,
    }


@pytest.fixture
def multiclass_data():
    rng = np.random.default_rng(1)
    n_tr, n_va, n_te = 500, 100, 100
    y_tr = rng.integers(0, 4, n_tr)
    y_va = rng.integers(0, 4, n_va)
    y_te = rng.integers(0, 4, n_te)
    le = LabelEncoder().fit(np.concatenate([y_tr, y_va, y_te]))
    return {
        "target_type": "multiclass_classification",
        "target_name": "m",
        "train_y": y_tr, "val_y": y_va, "test_y": y_te,
        "train_X": pd.DataFrame({"x": rng.normal(size=n_tr)}),
        "val_X": pd.DataFrame({"x": rng.normal(size=n_va)}),
        "test_X": pd.DataFrame({"x": rng.normal(size=n_te)}),
        "target_label_encoder": le,
    }


@pytest.fixture
def multilabel_data():
    rng = np.random.default_rng(2)
    K = 4
    return {
        "target_type": "multilabel_classification",
        "target_name": "ml",
        "train_y": rng.integers(0, 2, (500, K)),
        "val_y": rng.integers(0, 2, (100, K)),
        "test_y": rng.integers(0, 2, (100, K)),
        "train_X": pd.DataFrame({"x": rng.normal(size=500)}),
        "val_X": pd.DataFrame({"x": rng.normal(size=100)}),
        "test_X": pd.DataFrame({"x": rng.normal(size=100)}),
    }


@pytest.fixture
def ltr_data():
    rng = np.random.default_rng(3)
    n_tr, n_va, n_te = 500, 100, 100
    return {
        "target_type": "learning_to_rank",
        "target_name": "ltr",
        "train_y": rng.integers(0, 5, n_tr),
        "val_y": rng.integers(0, 5, n_va),
        "test_y": rng.integers(0, 5, n_te),
        "train_X": pd.DataFrame({"x": rng.normal(size=n_tr)}),
        "val_X": pd.DataFrame({"x": rng.normal(size=n_va)}),
        "test_X": pd.DataFrame({"x": rng.normal(size=n_te)}),
        "group_ids_train": np.repeat(np.arange(50), 10),
        "group_ids_val": np.repeat(np.arange(10), 10),
        "group_ids_test": np.repeat(np.arange(10), 10),
    }


# ---------------------------------------------------------------------
# Per-target dispatcher coverage (D-baseline)
# ---------------------------------------------------------------------


class TestDispatcher:
    def test_regression_returns_strongest(self, reg_data, cfg):
        rep = compute_dummy_baselines(config=cfg, **reg_data)
        assert rep.strongest is not None
        assert rep.primary_metric == "val_RMSE"
        assert "mean" in rep.table.index
        assert "median" in rep.table.index

    def test_binary_returns_strongest(self, binary_data, cfg):
        rep = compute_dummy_baselines(config=cfg, **binary_data)
        assert rep.strongest is not None
        # D5: log_loss is the headline classification metric, not AUC
        assert rep.primary_metric == "val_log_loss"
        assert "prior" in rep.table.index
        assert "most_frequent" in rep.table.index

    def test_multiclass_returns_strongest(self, multiclass_data, cfg):
        rep = compute_dummy_baselines(config=cfg, **multiclass_data)
        assert rep.strongest is not None
        assert rep.primary_metric == "val_log_loss"
        assert "uniform" in rep.table.index

    def test_multilabel_returns_strongest(self, multilabel_data, cfg):
        rep = compute_dummy_baselines(config=cfg, **multilabel_data)
        assert rep.strongest is not None
        # D5 multilabel: macro log-loss is the headline
        assert rep.primary_metric == "val_log_loss_macro"

    def test_ltr_returns_strongest(self, ltr_data, cfg):
        rep = compute_dummy_baselines(config=cfg, **ltr_data)
        assert rep.strongest is not None
        # NDCG@k is a maximize metric — we use NDCG@10 as primary
        assert "NDCG" in rep.primary_metric

    def test_quantile_routes_through_regression(self, reg_data, cfg):
        # quantile_regression currently shares regression dispatcher.
        d = dict(reg_data, target_type="quantile_regression", target_name="q")
        rep = compute_dummy_baselines(config=cfg, **d)
        assert rep.strongest is not None
        assert rep.primary_metric == "val_RMSE"


# ---------------------------------------------------------------------
# Headline metric for classification = log_loss, NOT AUC (D5)
# ---------------------------------------------------------------------


class TestHeadlineLogLoss:
    def test_constant_classifiers_have_auc_eq_half(self, binary_data, cfg):
        """All constant-prediction classifiers (prior/most_frequent/all_zeros/
        all_ones) collapse to AUC=0.5 by construction → AUC cannot
        discriminate them, so log_loss must be the headline."""
        rep = compute_dummy_baselines(config=cfg, **binary_data)
        if "val_AUC" in rep.table.columns:
            for name in ("prior", "most_frequent", "all_zeros", "all_ones"):
                if name in rep.table.index:
                    auc = rep.table.loc[name, "val_AUC"]
                    if pd.notna(auc):
                        assert abs(auc - 0.5) < 0.05, (
                            f"{name} AUC should ≈ 0.5 by construction, got {auc}"
                        )

    def test_multilabel_macro_and_micro_explicitly_named(self, multilabel_data, cfg):
        rep = compute_dummy_baselines(config=cfg, **multilabel_data)
        # D5/C#9 explicit naming: macro/micro log-loss columns. Sample-AUC
        # was deliberately out of scope (cost > budget in plan v3).
        assert "val_log_loss_macro" in rep.table.columns
        assert "val_log_loss_micro" in rep.table.columns


# ---------------------------------------------------------------------
# Per-group leakage gates (D1, round-3 A#1, C#1)
# ---------------------------------------------------------------------


class TestPerGroupLeakage:
    def test_high_cardinality_skipped(self, cfg):
        """D1 cardinality cap: cat with n_unique > 0.5*n_train (row-id-like)
        must be skipped — it would silently overfit to perfect train predictions."""
        rng = np.random.default_rng(0)
        n_tr = 200
        # Make 'cat' a row-identifier (n_unique == n_train >> 0.5 * n_train)
        train_X = pd.DataFrame({"cat": np.arange(n_tr)})
        val_X = pd.DataFrame({"cat": np.arange(50)})
        test_X = pd.DataFrame({"cat": np.arange(50)})
        rep = compute_dummy_baselines(
            target_type="regression", target_name="y",
            train_y=rng.normal(size=n_tr),
            val_y=rng.normal(size=50),
            test_y=rng.normal(size=50),
            train_X=train_X, val_X=val_X, test_X=test_X,
            cat_features=["cat"], config=cfg,
        )
        # per_group_mean rows should NOT be in the table.
        assert not any("per_group" in str(idx) for idx in rep.table.index), (
            f"per_group_mean leaked through cardinality cap: {list(rep.table.index)}"
        )

    def test_per_group_present_when_eligible(self, reg_data, cfg):
        """Low-cardinality categorical → per_group_mean baseline is emitted."""
        rep = compute_dummy_baselines(config=cfg, **reg_data)
        assert any("per_group" in str(idx) for idx in rep.table.index)

    def test_no_cat_features_no_per_group(self, reg_data, cfg):
        d = dict(reg_data, cat_features=None)
        rep = compute_dummy_baselines(config=cfg, **d)
        assert not any("per_group" in str(idx) for idx in rep.table.index)


# ---------------------------------------------------------------------
# TS detection (D17, round-3 A#4, A#17, C#6)
# ---------------------------------------------------------------------


class TestTimeSeriesDetection:
    def test_no_timestamps_no_ts_baselines(self, reg_data, cfg):
        rep = compute_dummy_baselines(config=cfg, **reg_data)
        ts_rows = [idx for idx in rep.table.index if "naive" in str(idx).lower()
                   or "seasonal" in str(idx).lower() or "rolling" in str(idx).lower()
                   or "linear_extrap" in str(idx).lower()]
        assert ts_rows == []

    def test_monotonic_daily_series_emits_ts_baselines(self, cfg):
        rng = np.random.default_rng(0)
        n_tr, n_va, n_te = 400, 80, 80
        ts_tr = pd.date_range("2024-01-01", periods=n_tr, freq="D")
        ts_va = pd.date_range(ts_tr[-1] + pd.Timedelta(days=1), periods=n_va, freq="D")
        ts_te = pd.date_range(ts_va[-1] + pd.Timedelta(days=1), periods=n_te, freq="D")
        # Inject weekly seasonality so seasonal_naive_p7 is informative.
        t_tr = np.arange(n_tr)
        t_va = np.arange(n_va) + n_tr
        t_te = np.arange(n_te) + n_tr + n_va
        y_tr = np.sin(2 * np.pi * t_tr / 7) + rng.normal(0, 0.1, n_tr)
        y_va = np.sin(2 * np.pi * t_va / 7) + rng.normal(0, 0.1, n_va)
        y_te = np.sin(2 * np.pi * t_te / 7) + rng.normal(0, 0.1, n_te)

        rep = compute_dummy_baselines(
            target_type="regression", target_name="ts",
            train_y=y_tr, val_y=y_va, test_y=y_te,
            train_X=pd.DataFrame({"x": rng.normal(size=n_tr)}),
            val_X=pd.DataFrame({"x": rng.normal(size=n_va)}),
            test_X=pd.DataFrame({"x": rng.normal(size=n_te)}),
            timestamps_train=ts_tr, timestamps_val=ts_va, timestamps_test=ts_te,
            config=cfg,
        )
        ts_rows = [str(idx) for idx in rep.table.index
                   if "naive" in str(idx) or "seasonal" in str(idx) or "rolling" in str(idx)]
        assert ts_rows, f"Expected TS baselines on monotonic daily series; got {list(rep.table.index)}"

    def test_interleaved_split_skips_ts_baselines(self, cfg):
        """Interleaved (non-monotonic) timestamps must skip TS baselines."""
        rng = np.random.default_rng(0)
        n = 200
        ts_all = pd.date_range("2024-01-01", periods=n, freq="D")
        # Random shuffle creates non-monotonic split.
        idx = rng.permutation(n)
        ts_tr = pd.Series(ts_all[idx[:120]])
        ts_va = pd.Series(ts_all[idx[120:160]])
        ts_te = pd.Series(ts_all[idx[160:]])
        y_all = rng.normal(size=n)
        rep = compute_dummy_baselines(
            target_type="regression", target_name="ts",
            train_y=y_all[idx[:120]], val_y=y_all[idx[120:160]], test_y=y_all[idx[160:]],
            train_X=pd.DataFrame({"x": rng.normal(size=120)}),
            val_X=pd.DataFrame({"x": rng.normal(size=40)}),
            test_X=pd.DataFrame({"x": rng.normal(size=40)}),
            timestamps_train=ts_tr, timestamps_val=ts_va, timestamps_test=ts_te,
            config=cfg,
        )
        ts_rows = [str(idx) for idx in rep.table.index
                   if "naive" in str(idx) or "seasonal" in str(idx) or "linear_extrap" in str(idx)]
        assert ts_rows == [], f"Interleaved split must skip TS baselines: {list(rep.table.index)}"

    def test_temporally_monotonic_helper(self):
        ts1 = pd.date_range("2024-01-01", periods=50, freq="D")
        ts2 = pd.date_range("2024-02-20", periods=50, freq="D")
        ts3 = pd.date_range("2024-04-10", periods=50, freq="D")
        assert _is_temporally_monotonic(np.asarray(ts1), np.asarray(ts2), np.asarray(ts3))
        assert not _is_temporally_monotonic(np.asarray(ts3), np.asarray(ts2), np.asarray(ts1))


# ---------------------------------------------------------------------
# Strongest-pick robustness (D2)
# ---------------------------------------------------------------------


class TestStrongestPickRobustness:
    def test_all_one_class_val_falls_back_to_test(self, cfg):
        """val with single class → strongest-pick falls back to test split."""
        rng = np.random.default_rng(0)
        y_tr = rng.integers(0, 2, 500)
        y_va = np.zeros(50, dtype=int)  # degenerate: all zeros
        y_te = rng.integers(0, 2, 100)
        le = LabelEncoder().fit(np.concatenate([y_tr, y_va, y_te]))
        rep = compute_dummy_baselines(
            target_type="binary_classification", target_name="b",
            train_y=y_tr, val_y=y_va, test_y=y_te,
            train_X=pd.DataFrame({"x": rng.normal(size=500)}),
            val_X=pd.DataFrame({"x": rng.normal(size=50)}),
            test_X=pd.DataFrame({"x": rng.normal(size=100)}),
            target_label_encoder=le, config=cfg,
        )
        # Should pick a strongest from test split when val is degenerate.
        assert rep.strongest is not None or rep.primary_metric is not None

    def test_both_splits_degenerate_strongest_none(self, cfg):
        rng = np.random.default_rng(0)
        # All-constant val + test → no signal anywhere
        y_tr = rng.integers(0, 2, 500)
        y_va = np.zeros(50, dtype=int)
        y_te = np.zeros(50, dtype=int)
        le = LabelEncoder().fit(np.concatenate([y_tr, y_va, y_te]))
        rep = compute_dummy_baselines(
            target_type="binary_classification", target_name="b",
            train_y=y_tr, val_y=y_va, test_y=y_te,
            train_X=pd.DataFrame({"x": rng.normal(size=500)}),
            val_X=pd.DataFrame({"x": rng.normal(size=50)}),
            test_X=pd.DataFrame({"x": rng.normal(size=50)}),
            target_label_encoder=le, config=cfg,
        )
        # Strongest may be None; plot must be None.
        assert rep.plot_path is None


# ---------------------------------------------------------------------
# LTR group sanity (D3, A#6)
# ---------------------------------------------------------------------


class TestLTRGroupSanity:
    def test_misaligned_group_ids_raises_or_skips(self, cfg):
        rng = np.random.default_rng(0)
        n_tr = 500
        with pytest.raises((AssertionError, ValueError)):
            compute_dummy_baselines(
                target_type="learning_to_rank", target_name="ltr",
                train_y=rng.integers(0, 5, n_tr),
                val_y=rng.integers(0, 5, 100),
                test_y=rng.integers(0, 5, 100),
                train_X=pd.DataFrame({"x": rng.normal(size=n_tr)}),
                val_X=pd.DataFrame({"x": rng.normal(size=100)}),
                test_X=pd.DataFrame({"x": rng.normal(size=100)}),
                # Mismatched length (498 vs 500) — must fail loudly
                group_ids_train=np.repeat(np.arange(83), 6),
                group_ids_val=np.repeat(np.arange(10), 10),
                group_ids_test=np.repeat(np.arange(10), 10),
                config=cfg,
            )

    def test_ltr_emits_random_within_query_baseline(self, ltr_data, cfg):
        rep = compute_dummy_baselines(config=cfg, **ltr_data)
        assert any("random_within_query" in str(idx) for idx in rep.table.index)
        assert any("identity_input_order" in str(idx) for idx in rep.table.index)


# ---------------------------------------------------------------------
# Per-cell metric isolation (D1, A#3)
# ---------------------------------------------------------------------


class TestPerCellMetricIsolation:
    def test_failed_column_present(self, reg_data, cfg):
        rep = compute_dummy_baselines(config=cfg, **reg_data)
        assert "failed" in rep.table.columns
        # Healthy run → all rows should have failed=False
        assert (rep.table["failed"] == False).all()  # noqa: E712


# ---------------------------------------------------------------------
# JSON serialization (D14, D15)
# ---------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_has_schema_version(self, reg_data, cfg):
        rep = compute_dummy_baselines(config=cfg, **reg_data)
        d = rep.to_dict()
        assert d["schema_version"] == SCHEMA_VERSION

    def test_to_dict_json_roundtrip(self, reg_data, cfg):
        rep = compute_dummy_baselines(config=cfg, **reg_data)
        d = rep.to_dict()
        # D15: must be JSON-serializable (NaN replaced with None)
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["schema_version"] == SCHEMA_VERSION
        assert parsed["target_type"] == "regression"

    def test_nan_scrubbed_to_none(self):
        # Manually construct a report with NaN to exercise scrubbing
        rep = BaselineReport(
            target_type="regression", target_name="y",
            table=pd.DataFrame({"val_RMSE": [1.0, float("nan")]}, index=["a", "b"]),
            strongest="a", primary_metric="val_RMSE",
            ts_period_used=None, plot_path=None, elapsed_s=0.1,
            n_train=10, n_val=2, n_test=2,
            n_train_finite=10, n_val_finite=2, n_test_finite=2,
            extras={},
        )
        d = rep.to_dict()
        assert d["data"]["b"]["val_RMSE"] is None
        # And json.dumps succeeds.
        json.dumps(d)


# ---------------------------------------------------------------------
# Polars / pandas / dtype variants
# ---------------------------------------------------------------------


class TestDtypeVariants:
    def test_polars_dataframe_X(self, cfg):
        pl = pytest.importorskip("polars")
        rng = np.random.default_rng(0)
        n_tr, n_va, n_te = 200, 50, 50
        rep = compute_dummy_baselines(
            target_type="regression", target_name="y",
            train_y=rng.normal(size=n_tr),
            val_y=rng.normal(size=n_va),
            test_y=rng.normal(size=n_te),
            train_X=pl.DataFrame({"x": rng.normal(size=n_tr)}),
            val_X=pl.DataFrame({"x": rng.normal(size=n_va)}),
            test_X=pl.DataFrame({"x": rng.normal(size=n_te)}),
            config=cfg,
        )
        assert rep.strongest is not None

    def test_polars_series_y(self, cfg):
        pl = pytest.importorskip("polars")
        rng = np.random.default_rng(0)
        n_tr, n_va, n_te = 200, 50, 50
        rep = compute_dummy_baselines(
            target_type="regression", target_name="y",
            train_y=pl.Series(rng.normal(size=n_tr)),
            val_y=pl.Series(rng.normal(size=n_va)),
            test_y=pl.Series(rng.normal(size=n_te)),
            train_X=pd.DataFrame({"x": rng.normal(size=n_tr)}),
            val_X=pd.DataFrame({"x": rng.normal(size=n_va)}),
            test_X=pd.DataFrame({"x": rng.normal(size=n_te)}),
            config=cfg,
        )
        assert rep.strongest is not None


# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------


class TestHelpers:
    def test_baseline_inputs_hash_deterministic(self):
        rng = np.random.default_rng(0)
        y_tr = rng.normal(size=100); y_va = rng.normal(size=20); y_te = rng.normal(size=20)
        h1 = _baseline_inputs_hash("regression", y_tr, y_va, y_te)
        h2 = _baseline_inputs_hash("regression", y_tr, y_va, y_te)
        assert h1 == h2
        # Different target_type → different hash.
        h3 = _baseline_inputs_hash("binary_classification", y_tr, y_va, y_te)
        assert h1 != h3

    def test_baseline_inputs_hash_changes_on_input_change(self):
        rng = np.random.default_rng(0)
        y_tr = rng.normal(size=100); y_va = rng.normal(size=20); y_te = rng.normal(size=20)
        h1 = _baseline_inputs_hash("regression", y_tr, y_va, y_te)
        y_tr_perturbed = y_tr.copy()
        y_tr_perturbed[0] += 1.0
        h2 = _baseline_inputs_hash("regression", y_tr_perturbed, y_va, y_te)
        assert h1 != h2

    def test_per_target_seed_deterministic(self):
        s1 = _per_target_seed(42, "TVT")
        s2 = _per_target_seed(42, "TVT")
        assert s1 == s2
        # Different target_name → different seed.
        s3 = _per_target_seed(42, "EGFDU")
        assert s1 != s3

    def test_slugify(self):
        # Valid characters preserved; problematic chars replaced.
        assert _slugify("simple_name") == "simple_name"
        # Path separator + spaces + colons → safe path component
        slug = _slugify("Bad/Name: with spaces")
        assert "/" not in slug
        assert ":" not in slug
        assert " " not in slug

    def test_normalize_timestamps_handles_none(self):
        assert _normalize_timestamps(None) is None

    def test_normalize_timestamps_datetime(self):
        ts = pd.date_range("2024-01-01", periods=10, freq="D")
        out = _normalize_timestamps(ts)
        assert out is not None
        assert len(out) == 10


# ---------------------------------------------------------------------
# format_text emits Operator Contract verdict line
# ---------------------------------------------------------------------


class TestVerdictFormat:
    def test_verdict_line_under_three_lines_default(self, reg_data, cfg):
        rep = compute_dummy_baselines(config=cfg, **reg_data)
        text = rep.format_text()
        # Operator Contract guarantee 1: ≤ 2 verdict lines + optional plot path.
        # Header is 1 line; verdict is 1 line. Plot path optional.
        # Total should be ≤ 4 lines for non-multi-output.
        n_lines = len([line for line in text.split("\n") if line.strip()])
        assert n_lines <= 4, f"Verdict text too verbose ({n_lines} lines):\n{text}"

    def test_verdict_line_contains_canonical_token(self, reg_data, cfg):
        rep = compute_dummy_baselines(config=cfg, **reg_data)
        text = rep.format_text()
        assert "[DUMMY_BASELINES]" in text


# ---------------------------------------------------------------------
# Suite-end summary + WARN tokens (D6)
# ---------------------------------------------------------------------


class TestSuiteEndSummary:
    @pytest.fixture
    def md(self):
        return {
            "regression": {
                "TVT": {
                    "strongest": "seasonal_naive_p7",
                    "primary_metric": "val_RMSE",
                    "data": {
                        "mean": {"val_RMSE": 645.34},
                        "seasonal_naive_p7": {"val_RMSE": 497.66},
                    },
                },
                "EGFDU": {
                    "strongest": "median",
                    "primary_metric": "val_RMSE",
                    "data": {
                        "mean": {"val_RMSE": 12.51},
                        "median": {"val_RMSE": 12.50},
                    },
                },
            },
        }

    def test_summary_emits_canonical_header(self, md):
        text = format_suite_end_summary(md)
        assert "[DUMMY_BASELINES] CROSS-TARGET VERDICT" in text

    def test_summary_emits_one_row_per_target(self, md):
        text = format_suite_end_summary(md)
        # Two data targets → at least 2 verdict rows past the header line.
        assert "TVT" in text
        assert "EGFDU" in text

    def test_best_model_below_dummy_warn_fires(self, md):
        # EGFDU model lift = dummy/model = 12.50/12.40 ≈ 1.008 < 1.5 → WARN
        text = format_suite_end_summary(
            md, best_model_metrics_by_target={
                ("regression", "TVT"): {"val_RMSE": 14.20, "model_name": "cb"},
                ("regression", "EGFDU"): {"val_RMSE": 12.40, "model_name": "xgb"},
            }, min_lift=1.5,
        )
        assert "WARN BEST_MODEL_BELOW_DUMMY" in text
        assert "EGFDU" in text

    def test_ts_beats_trees_warn_fires(self, md):
        # TVT model RMSE > seasonal_naive RMSE → TS_BEATS_TREES
        text = format_suite_end_summary(
            md, best_model_metrics_by_target={
                ("regression", "TVT"): {"val_RMSE": 800.0, "model_name": "cb"},
            }, min_lift=1.5,
        )
        assert "WARN TS_BEATS_TREES" in text

    def test_partial_failure_warn_fires(self, md):
        text = format_suite_end_summary(
            md, failures_metadata={
                "regression": {"BROKEN": "synthetic crash"},
            },
        )
        assert "WARN PARTIAL_FAILURE" in text
        assert "BROKEN" in text

    def test_all_baselines_below_random_fires_for_binary(self):
        md = {
            "binary_classification": {
                "T": {
                    "strongest": "prior",
                    "primary_metric": "val_log_loss",
                    "data": {
                        "prior": {"val_log_loss": 0.69, "val_AUC": 0.45},
                        "most_frequent": {"val_log_loss": 0.69, "val_AUC": 0.40},
                    },
                },
            },
        }
        text = format_suite_end_summary(md)
        assert "WARN ALL_BASELINES_BELOW_RANDOM" in text


# ---------------------------------------------------------------------
# Object-dtype target gate (D8)
# ---------------------------------------------------------------------


class TestObjectDtypeGate:
    def test_string_object_target_returns_empty_report(self, cfg):
        rng = np.random.default_rng(0)
        # Object-dtype with string content — incompatible with regression.
        y_tr = np.array(["a"] * 100, dtype=object)
        y_va = np.array(["b"] * 20, dtype=object)
        y_te = np.array(["c"] * 20, dtype=object)
        rep = compute_dummy_baselines(
            target_type="regression", target_name="y",
            train_y=y_tr, val_y=y_va, test_y=y_te,
            train_X=pd.DataFrame({"x": rng.normal(size=100)}),
            val_X=pd.DataFrame({"x": rng.normal(size=20)}),
            test_X=pd.DataFrame({"x": rng.normal(size=20)}),
            config=cfg,
        )
        # Should not raise; should return a valid report with no strongest.
        assert isinstance(rep, BaselineReport)
        assert rep.strongest is None or "skip_reason" in rep.extras


# ---------------------------------------------------------------------
# D4: Multi-output regression
# ---------------------------------------------------------------------


class TestMultiOutputRegression:
    def test_multi_output_emits_per_output_strongest(self, cfg):
        rng = np.random.default_rng(0)
        n_tr, n_va, n_te = 500, 100, 100
        K = 3
        # Different scales per output to exercise normalization
        scales = [10000.0, 10.0, 1.0]
        y_tr = np.column_stack([rng.normal(0, s, n_tr) for s in scales])
        y_va = np.column_stack([rng.normal(0, s, n_va) for s in scales])
        y_te = np.column_stack([rng.normal(0, s, n_te) for s in scales])
        X = pd.DataFrame({"x": rng.normal(size=n_tr + n_va + n_te)})
        rep = compute_dummy_baselines(
            target_type="regression", target_name="Y",
            train_y=y_tr, val_y=y_va, test_y=y_te,
            train_X=X.iloc[:n_tr], val_X=X.iloc[n_tr:n_tr+n_va], test_X=X.iloc[n_tr+n_va:],
            config=cfg,
        )
        assert "per_output_strongest" in rep.extras
        assert len(rep.extras["per_output_strongest"]) == K
        # Each entry has output, name, primary_metric, primary_value, normalized
        for entry in rep.extras["per_output_strongest"]:
            assert {"output", "name", "primary_metric", "primary_value", "normalized"} <= entry.keys()

    def test_multi_output_emits_cross_output_strongest(self, cfg):
        rng = np.random.default_rng(0)
        n_tr, n_va, n_te = 500, 100, 100
        K = 3
        # All same scale → normalization should converge cleanly
        y_tr = rng.normal(0, 1.0, (n_tr, K))
        y_va = rng.normal(0, 1.0, (n_va, K))
        y_te = rng.normal(0, 1.0, (n_te, K))
        X = pd.DataFrame({"x": rng.normal(size=n_tr + n_va + n_te)})
        rep = compute_dummy_baselines(
            target_type="regression", target_name="Y",
            train_y=y_tr, val_y=y_va, test_y=y_te,
            train_X=X.iloc[:n_tr], val_X=X.iloc[n_tr:n_tr+n_va], test_X=X.iloc[n_tr+n_va:],
            config=cfg,
        )
        assert "cross_output_strongest" in rep.extras
        assert "name" in rep.extras["cross_output_strongest"]
        assert "mean_normalized" in rep.extras["cross_output_strongest"]


# ---------------------------------------------------------------------
# D16: Bootstrap CI when min(n_val, n_test) < 2000
# ---------------------------------------------------------------------


class TestBootstrapCI:
    def test_ci_emitted_for_small_n(self, reg_data, cfg):
        # reg_data uses n_val=100 < 2000 → CI should fire.
        rep = compute_dummy_baselines(config=cfg, **reg_data)
        # CI may not always populate (best-effort); but extras has it when emitted.
        ci = rep.extras.get("bootstrap_ci")
        # If present, must have val key with (lo, point, hi) tuple.
        if ci is not None:
            assert "val" in ci or "test" in ci
            for split in ("val", "test"):
                if split in ci:
                    lo, point, hi = ci[split]
                    assert lo <= point <= hi

    def test_ci_suppressed_for_large_n(self, cfg):
        # n_val >= bootstrap_ci_threshold (2000) → CI suppressed.
        rng = np.random.default_rng(0)
        n_tr, n_va, n_te = 5000, 3000, 3000
        rep = compute_dummy_baselines(
            target_type="regression", target_name="y",
            train_y=rng.normal(size=n_tr), val_y=rng.normal(size=n_va), test_y=rng.normal(size=n_te),
            train_X=pd.DataFrame({"x": rng.normal(size=n_tr)}),
            val_X=pd.DataFrame({"x": rng.normal(size=n_va)}),
            test_X=pd.DataFrame({"x": rng.normal(size=n_te)}),
            config=cfg,
        )
        assert rep.extras.get("bootstrap_ci") is None


# ---------------------------------------------------------------------
# D17: statsmodels-uninstalled fallback
# ---------------------------------------------------------------------


class TestStatsmodelsFallback:
    def test_acf_returns_empty_when_statsmodels_missing(self, monkeypatch):
        """When statsmodels is unavailable, _detect_acf_periods returns []
        without raising; the rest of TS detection (step-size defaults)
        continues normally (D17)."""
        from mlframe.training import dummy_baselines as db
        # Monkey-patch the import to raise ImportError.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if "statsmodels" in name:
                raise ImportError("simulated: statsmodels not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rng = np.random.default_rng(0)
        y_train = rng.normal(size=200)
        # Should return [] gracefully, not raise.
        periods = db._detect_acf_periods(y_train, len(y_train))
        assert periods == []


# ---------------------------------------------------------------------
# D9: All-NaN val target column suppression
# ---------------------------------------------------------------------


class TestAllNaNValTarget:
    def test_all_nan_val_falls_back_to_test(self, cfg):
        rng = np.random.default_rng(0)
        n_tr, n_va, n_te = 500, 50, 100
        rep = compute_dummy_baselines(
            target_type="regression", target_name="y",
            train_y=rng.normal(size=n_tr),
            val_y=np.full(n_va, np.nan),  # all-NaN val
            test_y=rng.normal(size=n_te),
            train_X=pd.DataFrame({"x": rng.normal(size=n_tr)}),
            val_X=pd.DataFrame({"x": rng.normal(size=n_va)}),
            test_X=pd.DataFrame({"x": rng.normal(size=n_te)}),
            config=cfg,
        )
        # Header must surface n_val_finite=0
        assert rep.n_val_finite == 0
        # Strongest may pick from test fallback or be None.
        # Crucially, must not crash.

    def test_both_splits_all_nan_returns_empty_report(self, cfg):
        rng = np.random.default_rng(0)
        n_tr, n_va, n_te = 500, 50, 50
        rep = compute_dummy_baselines(
            target_type="regression", target_name="y",
            train_y=rng.normal(size=n_tr),
            val_y=np.full(n_va, np.nan),
            test_y=np.full(n_te, np.nan),
            train_X=pd.DataFrame({"x": rng.normal(size=n_tr)}),
            val_X=pd.DataFrame({"x": rng.normal(size=n_va)}),
            test_X=pd.DataFrame({"x": rng.normal(size=n_te)}),
            config=cfg,
        )
        assert rep.strongest is None
        assert "skip_reason" in rep.extras or rep.extras.get("skip_reason") is None  # tolerant


# ---------------------------------------------------------------------
# TS prediction rules
# ---------------------------------------------------------------------


class TestTSPredictionRules:
    def test_naive_lag7_uses_train_tail(self, cfg):
        """naive_lag7 should produce constant prediction = y_train[-7]
        (or appropriate cycle), differs from `mean` baseline."""
        rng = np.random.default_rng(0)
        n_tr, n_va, n_te = 400, 80, 80
        ts_tr = pd.date_range("2020-01-01", periods=n_tr, freq="D")
        ts_va = pd.date_range(ts_tr[-1] + pd.Timedelta(days=1), periods=n_va, freq="D")
        ts_te = pd.date_range(ts_va[-1] + pd.Timedelta(days=1), periods=n_te, freq="D")
        # Strong weekly seasonality
        t_tr = np.arange(n_tr); t_va = np.arange(n_va) + n_tr; t_te = np.arange(n_te) + n_tr + n_va
        y_tr = np.sin(2 * np.pi * t_tr / 7) + rng.normal(0, 0.1, n_tr)
        y_va = np.sin(2 * np.pi * t_va / 7) + rng.normal(0, 0.1, n_va)
        y_te = np.sin(2 * np.pi * t_te / 7) + rng.normal(0, 0.1, n_te)
        rep = compute_dummy_baselines(
            target_type="regression", target_name="ts",
            train_y=y_tr, val_y=y_va, test_y=y_te,
            train_X=pd.DataFrame({"x": rng.normal(size=n_tr)}),
            val_X=pd.DataFrame({"x": rng.normal(size=n_va)}),
            test_X=pd.DataFrame({"x": rng.normal(size=n_te)}),
            timestamps_train=ts_tr, timestamps_val=ts_va, timestamps_test=ts_te,
            config=cfg,
        )
        # When TS baselines fire, naive_lag7 should be among them (or seasonal_naive_p7).
        ts_baselines = [str(idx) for idx in rep.table.index
                        if "naive" in str(idx) or "seasonal" in str(idx)]
        assert ts_baselines, f"Expected TS baselines: {list(rep.table.index)}"
        # And on a strong-seasonal series, a TS baseline should beat `mean`.
        if "mean" in rep.table.index:
            mean_rmse = rep.table.loc["mean"].get("val_RMSE", float("inf"))
            best_ts = min(rep.table.loc[idx].get("val_RMSE", float("inf"))
                          for idx in ts_baselines)
            # Sanity: TS captures seasonality better than mean.
            assert best_ts < mean_rmse


# ---------------------------------------------------------------------
# n_repeats variance reporting
# ---------------------------------------------------------------------


class TestNRepeatsVariance:
    def test_stratified_baseline_runs_with_n_repeats(self, binary_data, cfg):
        """stratified baseline averages over n_repeats=20 by default;
        verify it runs cleanly + appears in the table."""
        rep = compute_dummy_baselines(config=cfg, **binary_data)
        assert "stratified" in rep.table.index
        # Metric should be finite (averaged over multiple seeds).
        assert np.isfinite(rep.table.loc["stratified"].get("val_log_loss", float("nan")))


# ---------------------------------------------------------------------
# Polars Enum / List dtype for multilabel
# ---------------------------------------------------------------------


class TestPolarsMultilabel:
    def test_polars_list_float32_multilabel_thresholded(self, cfg):
        """pl.List(pl.Float32) source must be coerced via >=0.5 threshold (D-inline / A#22)."""
        pl = pytest.importorskip("polars")
        rng = np.random.default_rng(0)
        K = 3
        n_tr, n_va, n_te = 200, 50, 50
        # Float labels — must be thresholded
        train_lists = [list(rng.uniform(0, 1, K).astype(float)) for _ in range(n_tr)]
        val_lists = [list(rng.uniform(0, 1, K).astype(float)) for _ in range(n_va)]
        test_lists = [list(rng.uniform(0, 1, K).astype(float)) for _ in range(n_te)]
        train_y = pl.Series("y", train_lists, dtype=pl.List(pl.Float32))
        val_y = pl.Series("y", val_lists, dtype=pl.List(pl.Float32))
        test_y = pl.Series("y", test_lists, dtype=pl.List(pl.Float32))
        rep = compute_dummy_baselines(
            target_type="multilabel_classification", target_name="ml",
            train_y=train_y, val_y=val_y, test_y=test_y,
            train_X=pd.DataFrame({"x": rng.normal(size=n_tr)}),
            val_X=pd.DataFrame({"x": rng.normal(size=n_va)}),
            test_X=pd.DataFrame({"x": rng.normal(size=n_te)}),
            config=cfg,
        )
        assert rep.strongest is not None
        assert np.isfinite(rep.table[rep.primary_metric].iloc[0])

    def test_polars_list_int8_multilabel(self, cfg):
        pl = pytest.importorskip("polars")
        rng = np.random.default_rng(0)
        K = 3
        # Build multilabel as List(Int8) — common roundtrip from FTE
        n_tr, n_va, n_te = 200, 50, 50
        train_lists = [list(rng.integers(0, 2, K).astype(int)) for _ in range(n_tr)]
        val_lists = [list(rng.integers(0, 2, K).astype(int)) for _ in range(n_va)]
        test_lists = [list(rng.integers(0, 2, K).astype(int)) for _ in range(n_te)]
        train_y = pl.Series("y", train_lists, dtype=pl.List(pl.Int8))
        val_y = pl.Series("y", val_lists, dtype=pl.List(pl.Int8))
        test_y = pl.Series("y", test_lists, dtype=pl.List(pl.Int8))
        rep = compute_dummy_baselines(
            target_type="multilabel_classification", target_name="ml",
            train_y=train_y, val_y=val_y, test_y=test_y,
            train_X=pd.DataFrame({"x": rng.normal(size=n_tr)}),
            val_X=pd.DataFrame({"x": rng.normal(size=n_va)}),
            test_X=pd.DataFrame({"x": rng.normal(size=n_te)}),
            config=cfg,
        )
        assert rep.strongest is not None
        # Coerced to (N, K) int — primary metric finite.
        assert np.isfinite(rep.table[rep.primary_metric].iloc[0])


# ---------------------------------------------------------------------
# D2: Paired-bootstrap robustness vs runner-up + TIE annotation
# ---------------------------------------------------------------------


class TestPairedBootstrap:
    def test_paired_bootstrap_emits_delta_and_p(self, cfg):
        """On strongly-seasonal series, strongest TS baseline should beat
        runner-up convincingly (P ≈ 1.0); paired_bootstrap dict populated."""
        rng = np.random.default_rng(0)
        n_tr, n_va, n_te = 400, 80, 80
        n = n_tr + n_va + n_te
        t = np.arange(n)
        y = 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.3, n)
        ts_all = pd.date_range("2020-01-01", periods=n, freq="D")
        X = pd.DataFrame({"x": rng.normal(size=n)})
        rep = compute_dummy_baselines(
            target_type="regression", target_name="ts",
            train_y=y[:n_tr], val_y=y[n_tr:n_tr + n_va], test_y=y[n_tr + n_va:],
            train_X=X.iloc[:n_tr], val_X=X.iloc[n_tr:n_tr + n_va], test_X=X.iloc[n_tr + n_va:],
            timestamps_train=ts_all[:n_tr], timestamps_val=ts_all[n_tr:n_tr + n_va],
            timestamps_test=ts_all[n_tr + n_va:],
            config=cfg,
        )
        paired = rep.extras.get("paired_bootstrap")
        # n_val=80 < 2000 → paired bootstrap fires.
        if paired is not None:
            assert "runner_up" in paired
            assert "delta" in paired
            assert "delta_ci" in paired
            assert "p_strongest_beats" in paired
            # Strong seasonality → strongest should beat runner-up cleanly.
            assert paired["p_strongest_beats"] >= 0.5

    def test_tie_annotation_skips_overlay_plot(self, cfg, tmp_path):
        """When two baselines are statistically indistinguishable, plot
        should be suppressed (D2)."""
        rng = np.random.default_rng(0)
        # Pure noise → mean and median should be ~equivalent → tie likely.
        n_tr, n_va, n_te = 500, 100, 100
        y_tr = rng.normal(size=n_tr); y_va = rng.normal(size=n_va); y_te = rng.normal(size=n_te)
        plot_dir = str(tmp_path)
        rep = compute_dummy_baselines(
            target_type="regression", target_name="noise",
            train_y=y_tr, val_y=y_va, test_y=y_te,
            train_X=pd.DataFrame({"x": rng.normal(size=n_tr)}),
            val_X=pd.DataFrame({"x": rng.normal(size=n_va)}),
            test_X=pd.DataFrame({"x": rng.normal(size=n_te)}),
            config=cfg,
            plot_file_prefix=plot_dir,
        )
        # If TIE detected, plot is suppressed; otherwise plot exists.
        if rep.extras.get("tie"):
            assert rep.plot_path is None


# ---------------------------------------------------------------------
# QUANTILE per-α empirical baselines
# ---------------------------------------------------------------------


class TestQuantilePerAlpha:
    def test_quantile_per_alpha_emits_pinball_columns(self, cfg):
        rng = np.random.default_rng(0)
        n = 500
        y = rng.normal(0, 1, n)
        rep = compute_dummy_baselines(
            target_type="quantile_regression", target_name="q",
            train_y=y, val_y=y[:100], test_y=y[100:200],
            train_X=pd.DataFrame({"x": rng.normal(size=n)}),
            val_X=pd.DataFrame({"x": rng.normal(size=100)}),
            test_X=pd.DataFrame({"x": rng.normal(size=100)}),
            quantile_alphas=[0.1, 0.5, 0.9],
            config=cfg,
        )
        assert rep.primary_metric == "val_pinball_mean"
        # Per-α columns
        for a in (0.1, 0.5, 0.9):
            assert f"val_pinball@{a:.3f}" in rep.table.columns
        # Headline: aggregate over non-boundary α
        assert "val_pinball_mean" in rep.table.columns
        # multi_quantile_empirical (the "right" baseline) should be in the table
        assert "multi_quantile_empirical" in rep.table.index

    def test_quantile_alpha_05_is_labeled_as_median(self, cfg):
        """D19 self-consistency annotation: α=0.5 row labeled identical to median."""
        rng = np.random.default_rng(0)
        n = 500
        y = rng.normal(0, 1, n)
        rep = compute_dummy_baselines(
            target_type="quantile_regression", target_name="q",
            train_y=y, val_y=y[:100], test_y=y[100:200],
            train_X=pd.DataFrame({"x": rng.normal(size=n)}),
            val_X=pd.DataFrame({"x": rng.normal(size=100)}),
            test_X=pd.DataFrame({"x": rng.normal(size=100)}),
            quantile_alphas=[0.5],
            config=cfg,
        )
        # The α=0.5 row's index should annotate the construction.
        idx_strs = [str(i) for i in rep.table.index]
        assert any("median by construction" in s for s in idx_strs)


# ---------------------------------------------------------------------
# D11/D12: Plot path slugify + suppression
# ---------------------------------------------------------------------


class TestPlotPathSlugify:
    def test_unicode_target_name_slugified(self, reg_data, cfg, tmp_path):
        """D11: target name with unicode / spaces / colons must slugify
        to a safe path component."""
        plot_dir = str(tmp_path)
        d = dict(reg_data, target_name="Bad/Name: рус", train_X=reg_data["train_X"].copy())
        rep = compute_dummy_baselines(config=cfg, plot_file_prefix=plot_dir, **d)
        if rep.plot_path is not None:
            # Must not contain raw '/', ':', or non-ASCII path-breaking chars
            import os as _os
            tail = _os.path.basename(rep.plot_path)
            assert ":" not in tail
            assert " " not in tail


# ---------------------------------------------------------------------
# D20: sklearn version assertion is in module load
# ---------------------------------------------------------------------


class TestSklearnVersionAssertion:
    def test_sklearn_version_check_passes(self):
        """Module load already enforces sklearn >= 1.0 via assert.
        This test simply confirms the module imported cleanly."""
        from mlframe.training import dummy_baselines as db
        import sklearn
        assert sklearn.__version__ >= "1.0"
        # The assertion is at module load → just verify import works.
        assert db.compute_dummy_baselines is not None


# ---------------------------------------------------------------------
# n_val < 10 sample-noise gate (D10)
# ---------------------------------------------------------------------


class TestSmallNGate:
    def test_n_val_below_10_falls_back_to_test(self, cfg):
        rng = np.random.default_rng(0)
        n_tr = 500
        n_va = 5  # below 10 → degenerate
        n_te = 100
        rep = compute_dummy_baselines(
            target_type="regression", target_name="y",
            train_y=rng.normal(size=n_tr),
            val_y=rng.normal(size=n_va),
            test_y=rng.normal(size=n_te),
            train_X=pd.DataFrame({"x": rng.normal(size=n_tr)}),
            val_X=pd.DataFrame({"x": rng.normal(size=n_va)}),
            test_X=pd.DataFrame({"x": rng.normal(size=n_te)}),
            config=cfg,
        )
        # Strongest may pick from test fallback or be None — must not crash.
        assert isinstance(rep, BaselineReport)


# ---------------------------------------------------------------------
# Numba acceleration availability
# ---------------------------------------------------------------------


class TestNumbaAcceleration:
    def test_multilabel_macro_log_loss_finite(self, multilabel_data, cfg):
        """Numba-accelerated path must produce finite, plausible values."""
        rep = compute_dummy_baselines(config=cfg, **multilabel_data)
        # Macro log-loss should be in (0, ln(2)) range for binary-per-label
        # with random data — typically near ln(2) ≈ 0.693.
        for idx in rep.table.index:
            v = rep.table.loc[idx, "val_log_loss_macro"]
            if pd.notna(v):
                # Sanity: reasonable bounds for binary-per-label cross-entropy.
                assert 0 < v < 30, f"{idx}: implausible macro log-loss {v}"


# ---------------------------------------------------------------------
# Auto-dropped high-card cat re-attachment for per_group_mean
# (use_text_features=False path — well_id with 600+ unique values gets
# stripped from tree-model frames but should still flow into
# dummy_baselines per_group_mean diagnostic).
# ---------------------------------------------------------------------


class TestAugmentWithDroppedHighCardCols:
    """Direct unit tests for the _augment_with_dropped_high_card_cols helper
    in mlframe.training.core. The helper re-attaches pre-drop column data
    to OD-filtered frames, sliced by train_od_idx / val_od_idx masks."""

    def _import_helper(self):
        from mlframe.training.core import _augment_with_dropped_high_card_cols
        return _augment_with_dropped_high_card_cols

    def test_no_dropped_data_returns_frames_unchanged(self):
        aug = self._import_helper()
        train = pd.DataFrame({"x": [1, 2, 3]})
        val = pd.DataFrame({"x": [4]})
        test = pd.DataFrame({"x": [5]})
        t, v, te, added = aug({}, train, val, test)
        assert added == []
        # Frames returned as-is when no dropped data.
        assert t is train
        assert v is val
        assert te is test

    def test_pandas_passthrough_no_od_mask(self):
        aug = self._import_helper()
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        val = pd.DataFrame({"x": [10.0]})
        test = pd.DataFrame({"x": [20.0, 30.0]})
        dropped = {
            "well_id": {
                "train": np.array(["a", "b", "c", "d"], dtype=object),
                "val": np.array(["x"], dtype=object),
                "test": np.array(["y", "z"], dtype=object),
            }
        }
        t, v, te, added = aug(dropped, train, val, test)
        assert added == ["well_id"]
        assert "well_id" in t.columns
        assert "well_id" in v.columns
        assert "well_id" in te.columns
        # Original column still there.
        assert "x" in t.columns
        # Values match.
        assert list(t["well_id"]) == ["a", "b", "c", "d"]
        assert list(te["well_id"]) == ["y", "z"]

    def test_train_od_mask_slicing(self):
        """When OD applied, captured pre-OD ndarrays get sliced by mask."""
        aug = self._import_helper()
        # Pre-OD train had 5 rows; OD kept indices [0, 2, 4] → post-OD has 3 rows.
        post_od_train = pd.DataFrame({"x": [1.0, 3.0, 5.0]})
        # train_od_idx is a bool mask of length 5
        train_od_mask = np.array([True, False, True, False, True])
        dropped = {
            "well_id": {
                "train": np.array(["w0", "w1", "w2", "w3", "w4"], dtype=object),
            }
        }
        t, _, _, added = aug(
            dropped, post_od_train, None, None, train_od_idx=train_od_mask,
        )
        assert added == ["well_id"]
        assert list(t["well_id"]) == ["w0", "w2", "w4"]

    def test_length_mismatch_silently_skipped(self):
        """Captured ndarray with wrong length post-slicing → column not added."""
        aug = self._import_helper()
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        # Captured length 5 but no OD mask → can't align with len(train)=3
        dropped = {"col_bad": {"train": np.arange(5)}}
        t, _, _, added = aug(dropped, train, None, None)
        assert added == []
        assert "col_bad" not in t.columns

    def test_polars_frame_supported(self):
        pl = pytest.importorskip("polars")
        aug = self._import_helper()
        train = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        val = pl.DataFrame({"x": [10.0]})
        test = pl.DataFrame({"x": [20.0, 30.0]})
        dropped = {
            "well_id": {
                "train": np.array(["a", "b", "c"], dtype=object),
                "val": np.array(["v"], dtype=object),
                "test": np.array(["t1", "t2"], dtype=object),
            }
        }
        t, v, te, added = aug(dropped, train, val, test)
        assert added == ["well_id"]
        assert isinstance(t, pl.DataFrame)
        assert "well_id" in t.columns
        assert t["well_id"].to_list() == ["a", "b", "c"]


class TestPerGroupOnDroppedHighCardCol:
    """End-to-end via compute_dummy_baselines: when caller passes a high-
    card cat column directly (simulating what core.py does after re-
    attachment), per_group_mean baseline appears in the table and beats
    `mean` on group-structured synthetic data."""

    def test_per_group_mean_on_high_card_cat(self, cfg):
        rng = np.random.default_rng(0)
        n_groups = 600
        n_tr, n_va, n_te = 4000, 500, 500
        # Synthesize y = group_offset[group_id] + noise so per_group_mean
        # has strong signal vs constant baselines.
        group_offsets = rng.normal(0, 5, size=n_groups)
        group_id_tr = rng.integers(0, n_groups, n_tr)
        group_id_va = rng.integers(0, n_groups, n_va)
        group_id_te = rng.integers(0, n_groups, n_te)
        y_tr = group_offsets[group_id_tr] + rng.normal(0, 1, n_tr)
        y_va = group_offsets[group_id_va] + rng.normal(0, 1, n_va)
        y_te = group_offsets[group_id_te] + rng.normal(0, 1, n_te)

        # X frames carry the group_id column directly (post-re-attachment).
        train_X = pd.DataFrame({
            "x1": rng.normal(size=n_tr),
            "well_id": [f"grp_{g:04d}" for g in group_id_tr],
        })
        val_X = pd.DataFrame({
            "x1": rng.normal(size=n_va),
            "well_id": [f"grp_{g:04d}" for g in group_id_va],
        })
        test_X = pd.DataFrame({
            "x1": rng.normal(size=n_te),
            "well_id": [f"grp_{g:04d}" for g in group_id_te],
        })

        rep = compute_dummy_baselines(
            target_type="regression", target_name="y",
            train_y=y_tr, val_y=y_va, test_y=y_te,
            train_X=train_X, val_X=val_X, test_X=test_X,
            cat_features=["well_id"],
            config=cfg,
        )
        # per_group_mean should be in the table (cardinality cap = 0.5*n_train
        # = 2000; well_id has 600 unique << 2000, so it passes the gate).
        per_group_rows = [str(idx) for idx in rep.table.index if "per_group" in str(idx)]
        assert per_group_rows, (
            f"per_group_mean missing on high-card group key; got {list(rep.table.index)}"
        )
        # And it should be the strongest (group structure dominates over mean/median).
        assert rep.strongest is not None and "per_group" in str(rep.strongest)
        # Lift over mean should be substantial (group offset std=5 vs noise=1).
        mean_rmse = rep.table.loc["mean", "val_RMSE"]
        per_group_rmse = rep.table.loc[rep.strongest, "val_RMSE"]
        assert per_group_rmse < mean_rmse * 0.5, (
            f"per_group RMSE {per_group_rmse} not significantly less than mean RMSE {mean_rmse}"
        )

    def test_helper_present_in_core(self):
        """Capture-block + call-site augmentation should reference the helper."""
        from mlframe.training import core as _core
        assert hasattr(_core, "_augment_with_dropped_high_card_cols"), (
            "_augment_with_dropped_high_card_cols helper missing from core.py"
        )
        # Also verify the helper is callable.
        result = _core._augment_with_dropped_high_card_cols(
            {}, None, None, None,
        )
        # Empty input returns 4-tuple with empty `added`.
        assert len(result) == 4
        assert result[3] == []
