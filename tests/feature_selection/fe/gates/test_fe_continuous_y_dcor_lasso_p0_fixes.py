"""Regression tests for three confirmed P0 feature-selection bugs.

1. Continuous-y MI/CMI gate in the temporal / grouped aggregation FE no longer
   int-truncates y (0.7 -> 0) before binning; it quantile-bins continuous y.
2. distance-correlation scoring guards a non-finite denominator so an all-NaN
   column scores 0 rather than escaping a NaN into the ranking / sort.
3. Lasso |coef| scoring binarises multiclass y one-vs-rest instead of
   regressing on the raw ordinal class integers.
"""

import numpy as np
import pandas as pd
import pytest


def _continuous_y_temporal_frame(n_entities=40, per=12, seed=0):
    """Continuous y temporal frame."""
    rng = np.random.default_rng(seed)
    rows = []
    ys = []
    for e in range(n_entities):
        base = rng.uniform(0.0, 1.0)
        for t in range(per):
            val = base + 0.01 * t + 0.001 * rng.standard_normal()
            rows.append({"entity": e, "time": t, "v": val})
            # Continuous target in [0, 1): int8/int64 truncation collapses it to a
            # SINGLE class (all zeros) -> MI gate sees no signal. The expanding mean
            # of v tracks `base`, which is exactly y here, so a correct quantile-bin
            # gate must keep at least one engineered column.
            ys.append(base)
    X = pd.DataFrame(rows)
    y = np.asarray(ys, dtype=np.float64)
    return X, y


def test_temporal_fe_continuous_y_survives_gate_not_int_truncated():
    """Temporal fe continuous y survives gate not int truncated."""
    from mlframe.feature_selection.filters._temporal_agg_fe import hybrid_temporal_agg_fe

    X, y = _continuous_y_temporal_frame()
    # Sanity: int truncation would collapse y to one class.
    assert np.unique(y.astype(np.int64)).size == 1

    _, appended, _, scores = hybrid_temporal_agg_fe(
        X,
        y,
        entity_cols=["entity"],
        value_cols=["v"],
        time_col="time",
        stats=("mean",),
        lags=(),
        top_k=5,
        min_mi=1e-4,
    )
    assert not scores.empty
    assert float(scores["mi"].max()) > 1e-4, "continuous-y true signal must survive the MI gate; int truncation would zero it"
    assert appended, "at least one engineered column must be appended"


def test_grouped_fe_continuous_y_not_int_truncated():
    """Grouped fe continuous y not int truncated."""
    from mlframe.feature_selection.filters._grouped_agg_fe import (
        score_grouped_agg_by_cmi_uplift,
    )

    rng = np.random.default_rng(1)
    n = 600
    grp = rng.integers(0, 30, size=n)
    # group-mean carries the signal; y is continuous in [0, 1)
    grp_level = rng.uniform(0.0, 1.0, size=30)
    y = grp_level[grp] + 0.001 * rng.standard_normal(n)
    assert np.unique(y.astype(np.int64)).size == 1
    raw_X = pd.DataFrame({"g": grp, "num": rng.standard_normal(n)})
    eng_X = pd.DataFrame({"g__mean_enc": grp_level[grp]})

    res = score_grouped_agg_by_cmi_uplift(
        raw_X,
        eng_X,
        y,
        base_cols=["num"],
        n_bins=10,
        eng_to_source={"g__mean_enc": "num"},
    )
    assert not res.empty
    assert float(res["cmi"].max()) > 1e-3, "continuous-y group signal must survive the CMI gate"


def test_dcor_all_nan_column_scores_zero_not_nan():
    """Dcor all nan column scores zero not nan."""
    from mlframe.feature_selection.filters._orthogonal_dcor_fe import (
        distance_correlation,
        _dcor_batch,
    )

    rng = np.random.default_rng(2)
    n = 400
    y = rng.standard_normal(n)
    nan_col = np.full(n, np.nan)
    score = distance_correlation(nan_col, y, n_sample=200)
    assert np.isfinite(score) and score == 0.0

    inf_col = np.full(n, np.inf)
    assert distance_correlation(inf_col, y, n_sample=200) == 0.0

    # batch: an all-NaN column must not be NaN nor sort to the top.
    good = y + 0.01 * rng.standard_normal(n)
    X = np.column_stack([nan_col, good])
    out = _dcor_batch(X, y, n_sample=200)
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0
    assert int(np.argmax(out)) == 1, "NaN column must not rank above the real signal"


def test_lasso_multiclass_not_driven_by_spurious_ordinal():
    """Lasso multiclass not driven by spurious ordinal."""
    from mlframe.feature_selection.filters._orthogonal_lasso_fe import (
        score_features_by_lasso_coef,
    )

    rng = np.random.default_rng(3)
    n = 900
    cls = rng.integers(0, 3, size=n)  # 3 classes, label integers meaningless

    # `signal` is high iff class == 1 (a NON-monotone relationship in class id):
    # regressing on raw ordinal {0,1,2} sees class 1 as the "middle" value and
    # largely cancels its linear effect, so the spurious column `ordinal_noise`
    # (linear in the raw class integer) wins the |coef| ranking. One-vs-rest
    # binarisation recovers `signal` as the driver. raw_X holds only unrelated
    # baseline-lookup sources so the engineered columns are the sole carriers.
    signal = (cls == 1).astype(np.float64) + 0.05 * rng.standard_normal(n)
    ordinal_noise = cls.astype(np.float64) + 0.05 * rng.standard_normal(n)
    raw_X = pd.DataFrame({"u0": rng.standard_normal(n), "u1": rng.standard_normal(n)})
    eng_X = pd.DataFrame(
        {
            "u0__sig": signal,
            "u1__ord": ordinal_noise,
        }
    )

    res = score_features_by_lasso_coef(raw_X, eng_X, cls, alpha=0.001)
    top = res.iloc[0]["engineered_col"]
    assert top == "u0__sig", (
        f"multiclass selection must be driven by the true class-1 signal, not the spurious ordinal column; got top={top}\n{res}"
    )


def test_ksg_gpu_cpu_parity_on_discrete_ties():
    """Ksg gpu cpu parity on discrete ties."""
    pytest.importorskip("cupy")
    import cupy as cp

    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from mlframe.feature_selection.filters._ksg import mixed_ksg_mi, mixed_ksg_mi_gpu

    rng = np.random.default_rng(7)
    n = 4000
    # discrete y (ties) + continuous x with signal -> exercises the tie-jitter
    # path that the GPU previously skipped.
    x = rng.standard_normal(n)
    y = (x > 0).astype(np.float64) + 0.3 * rng.standard_normal(n)
    y = np.round(y).astype(np.float64)
    cpu = mixed_ksg_mi(x, y, k=5, seed=0)
    gpu = mixed_ksg_mi_gpu(x, y, k=5, seed=0)
    assert abs(cpu - gpu) < 0.05, f"CPU {cpu} vs GPU {gpu} diverge"
