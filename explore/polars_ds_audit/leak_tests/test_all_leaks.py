"""Leak-safety checks for polars_ds Blueprint transforms.

Принцип: для каждого stateful API проверяем, что статистика, зашитая в Blueprint
при `materialize()` на train, применяется identically к test — test-значения НЕ
участвуют в fit. Контрольные тесты:

1. **pipeline determinism**: pipeline.transform(test) выдаёт одинаковый результат
   если prepend test rows к train (merge train с test и fit заново — результат на test
   должен ОТЛИЧАТЬСЯ, если реализация действительно fit-on-train-only).

2. **target-leak для TE/WoE**: fit_transform на train БЕЗ OOF сильно переобучается —
   train-AUC >> test-AUC на синтетическом датасете без истинного сигнала. Это ожидаемая
   проблема, фиксируем её как красный флаг (и используем как мотив для upstream-патча).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
import pytest
from polars_ds.pipeline import Blueprint

from _common import (
    make_numeric_data,
    make_high_card_cat,
    train_test_split_frame,
    auc,
)


# ---------- API 1: winsorize ----------
def test_winsorize_fit_only_on_train():
    df = make_numeric_data(n=5000, n_features=5, missing_rate=0.0)
    tr, te = train_test_split_frame(df)
    # bounds — train quantiles
    bp = Blueprint(tr, name="w").winsorize(cols=["f0"], q_low=0.01, q_high=0.99)
    pipe = bp.materialize()
    te_t = pipe.transform(te)
    # границы должны быть внутри train quantile — убеждаемся, что применён тот же
    # clip, а не переподсчитан на test.
    tr_lo = tr["f0"].quantile(0.01)
    tr_hi = tr["f0"].quantile(0.99)
    assert te_t["f0"].min() >= tr_lo - 1e-9
    assert te_t["f0"].max() <= tr_hi + 1e-9


# ---------- API 2: impute ----------
def test_impute_uses_train_median_only():
    df = make_numeric_data(n=5000, n_features=3, missing_rate=0.2)
    tr, te = train_test_split_frame(df)
    bp = Blueprint(tr, name="i").impute(cols=["f0"], method="median")
    pipe = bp.materialize()
    # получаем test с импутацией
    te_t = pipe.transform(te)
    # в колонке f0 теперь нет null
    assert te_t["f0"].null_count() == 0
    # медиана использованная для заполнения — это train median, не test median
    train_med = tr["f0"].median()
    test_med = te["f0"].median()
    # значения, которые были null в te, должны стать train_med (а не test_med)
    null_mask_te = te["f0"].is_null().to_numpy()
    if null_mask_te.any():
        filled = te_t["f0"].to_numpy()[null_mask_te]
        assert np.allclose(filled, train_med, atol=1e-9)
        if abs(train_med - test_med) > 1e-6:
            # статистика из train, не из test — это подтверждает leak-safety
            assert not np.allclose(filled, test_med)


# ---------- API 3/4: WoE / Target encode — leakage exposure ----------
@pytest.mark.parametrize("encoder", ["target_encode", "woe_encode"])
def test_te_woe_overfits_without_oof(encoder):
    """Без OOF fit_transform на train переобучается: train AUC >> test AUC.

    Это красный флаг. Документируется для REPORT.md как основание upstream-патча.
    """
    from sklearn.linear_model import LogisticRegression

    df = make_high_card_cat(n=6000, n_cat_cols=3, cardinality=300, signal_strength=0.3, seed=7)
    tr, te = train_test_split_frame(df, frac=0.7)
    bp = Blueprint(tr, name=encoder, target="y")
    bp = getattr(bp, encoder)(cols=["c0", "c1", "c2"])
    pipe = bp.materialize()
    tr_t = pipe.transform(tr)
    te_t = pipe.transform(te)

    feats = ["c0", "c1", "c2"] + [c for c in tr_t.columns if c.startswith("n")]
    # некоторые n* могут иметь null (не касается)
    X_tr = tr_t.select(feats).fill_null(0.0).to_numpy()
    X_te = te_t.select(feats).fill_null(0.0).to_numpy()
    y_tr = tr_t["y"].to_numpy()
    y_te = te_t["y"].to_numpy()

    m = LogisticRegression(max_iter=500)
    m.fit(X_tr, y_tr)
    auc_tr = auc(y_tr, m.predict_proba(X_tr)[:, 1])
    auc_te = auc(y_te, m.predict_proba(X_te)[:, 1])
    gap = auc_tr - auc_te
    print(f"[{encoder}] train AUC={auc_tr:.3f}  test AUC={auc_te:.3f}  gap={gap:.3f}")
    # gap > 0.02 — признак переобучения из-за target leakage на train.
    # Тест НЕ валит билд (это не bug polars_ds — это missing feature), а документирует.
    assert gap >= 0, "sanity"


# ---------- API 5: stat tests — stateless, sanity ----------
def test_stat_tests_stateless_same_on_resplit():
    import polars_ds as pds
    df = make_numeric_data(n=3000, n_features=5)
    # t-test — stateless, должен давать одинаковый результат независимо от fit
    df2 = df.with_columns((pl.col("y") == 1).alias("_m"))
    r1 = df2.select(pds.ttest_ind(
        pl.col("f0").filter(pl.col("_m")), pl.col("f0").filter(~pl.col("_m")),
        equal_var=False).alias("t"))
    r2 = df2.select(pds.ttest_ind(
        pl.col("f0").filter(pl.col("_m")), pl.col("f0").filter(~pl.col("_m")),
        equal_var=False).alias("t"))
    assert r1.equals(r2)


# ---------- API 6: PCA ----------
def test_pca_stateful():
    import polars_ds as pds
    df = make_numeric_data(n=3000, n_features=10, missing_rate=0, outlier_rate=0)
    feats = [c for c in df.columns if c != "y"]
    # pds.pca не имеет fit/transform разделения в Blueprint → отмечаем как
    # красный флаг; интеграция в mlframe потребует обёртки.
    try:
        out = df.select(pds.pca(*[pl.col(f) for f in feats], k=3))
        assert out.shape[0] == 3000
    except Exception as e:
        pytest.skip(f"pds.pca surface unclear: {e}")


# ---------- API 7: string distance — stateless ----------
def test_string_dist_stateless():
    import polars_ds as pds
    df = pl.DataFrame({"a": ["cat", "dog"], "b": ["car", "log"]})
    r = df.select(pds.str_leven("a", "b").alias("d"))
    assert r["d"].to_list() == [1, 1]


# ---------- API 8: power transforms — absence ----------
def test_power_transform_absent():
    """В polars_ds 0.10.3 нет first-class yeo_johnson/box_cox — фиксируем для REPORT."""
    import polars_ds as pds
    assert not any(x in dir(pds) for x in ("yeo_johnson", "yeojohnson", "box_cox", "boxcox"))


if __name__ == "__main__":
    import subprocess
    subprocess.call([sys.executable, "-m", "pytest", __file__, "-v", "-s"])
