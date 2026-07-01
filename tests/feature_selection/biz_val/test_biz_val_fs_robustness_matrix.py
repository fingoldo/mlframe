"""FS robustness matrix: selector x target-type x (NaN-heavy / mixed-categorical), at small and medium n.

Complements test_biz_val_target_types_coverage.py (MRMR signal recovery across target types) by adding the
NaN-heavy and mixed-categorical robustness dimensions and RFECV coverage. Primary contract: the selector must
NOT crash on 25-40% NaN or on string/category columns, AND must still recover the signal / reject most noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

N_SIGNAL = 3
N_NOISE = 6
_SEEDS = [0, 1, 7]
_KINDS = ["regression", "binary", "multiclass", "count", "ordinal"]



pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)

def _make(kind, n, seed, *, nan_frac=0.0, with_cat=False):
    """3 signal numeric cols (s0..s2) + ``N_NOISE`` noise (n0..) + target by ``kind``; optional NaN injection and
    an informative categorical (``cat_inf``) + high-cardinality noise categorical (``cat_noise``)."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n, N_SIGNAL))
    noise = rng.normal(size=(n, N_NOISE))
    lin = 2.0 * s[:, 0] - 1.3 * s[:, 1] + 0.8 * s[:, 2]
    if kind == "regression":
        y = (lin + 0.3 * rng.normal(size=n)).astype(float)
    elif kind == "binary":
        y = (lin + 0.3 * rng.normal(size=n) > 0).astype(int)
    elif kind == "multiclass":
        y = np.digitize(lin, np.quantile(lin, [1 / 3, 2 / 3])).astype(int)
    elif kind == "count":
        y = rng.poisson(np.exp(0.5 * s[:, 0] + 0.4 * s[:, 1] - 0.3 * s[:, 2])).astype(int)
    elif kind == "ordinal":
        y = np.digitize(lin, np.quantile(lin, [0.2, 0.4, 0.6, 0.8])).astype(int)
    else:
        raise ValueError(kind)
    cols = {f"s{i}": s[:, i] for i in range(N_SIGNAL)}
    cols.update({f"n{i}": noise[:, i] for i in range(N_NOISE)})
    df = pd.DataFrame(cols)
    if nan_frac > 0:
        for c in df.columns:
            idx = rng.choice(n, int(nan_frac * n), replace=False)
            df.loc[df.index[idx], c] = np.nan
    if with_cat:
        df["cat_inf"] = np.where(s[:, 0] > 0, "hi", "lo")           # informative (tracks s0 sign)
        df["cat_noise"] = rng.integers(0, 30, n).astype(str)         # high-cardinality pure noise
        df = df.astype({"cat_inf": "category", "cat_noise": "category"})
    return df, y, [f"s{i}" for i in range(N_SIGNAL)], [f"n{i}" for i in range(N_NOISE)]


def _mrmr_support(df, y, seed):
    from mlframe.feature_selection.filters.mrmr import MRMR

    # This suite asserts the selector RECOVERS the signal raws -- emit_both keeps signal-bearing raw
    # operands of selected engineered features (noise operands gated out), so a fundamentally linear
    # signal folded into nonlinear engineered children is still surfaced as raw support.
    sel = MRMR(verbose=0, use_simple_mode=True, max_runtime_mins=1, n_workers=1, quantization_nbins=8, random_seed=seed, redundancy_policy="emit_both")
    sel.fit(df, pd.Series(y))
    sup = np.asarray(sel.support_)
    cols = list(df.columns)
    return set([cols[i] for i in np.where(sup)[0]] if sup.dtype == bool else [cols[int(i)] for i in sup.tolist()])


def _rfecv_support(df, y, seed, regression):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from mlframe.feature_selection.wrappers import RFECV

    est = HistGradientBoostingRegressor(max_iter=40, random_state=seed) if regression else HistGradientBoostingClassifier(max_iter=40, random_state=seed)
    sel = RFECV(estimator=est, cv=3)
    sel.fit(df, pd.Series(y))
    out = sel.transform(df)
    return set(out.columns) if hasattr(out, "columns") else set(np.asarray(df.columns)[sel.get_support()])


# --- MRMR: NaN-heavy across every target type --------------------------------------------------


@pytest.mark.parametrize("kind", _KINDS)
def test_mrmr_nan_heavy_recovers_signal_all_target_types(kind):
    """30% NaN in every column: MRMR (native NaN bin) must not crash and recover >=2/3 signal on a majority of
    seeds while leaking little noise. Measured: full 3/3 signal, 0 noise on the clean probe."""
    recs, noise_leak = [], []
    for s in _SEEDS:
        df, y, sig, noi = _make(kind, 2000, s, nan_frac=0.3)
        sup = _mrmr_support(df, y, s)
        recs.append(len(set(sig) & sup))
        noise_leak.append(len(set(noi) & sup))
    assert sum(r >= 2 for r in recs) >= 2, f"MRMR NaN/{kind}: expected >=2/3 signal on majority of seeds; per-seed={recs}"
    assert sum(noise_leak) <= 4, f"MRMR NaN/{kind}: too much noise leaked across seeds; got {noise_leak}"


@pytest.mark.parametrize("kind", ["regression", "binary"])
def test_mrmr_nan_heavy_medium_n(kind):
    """Medium n=15000 with 30% NaN: still recovers signal, rejects noise. One seed (cost)."""
    df, y, sig, noi = _make(kind, 15000, 0, nan_frac=0.3)
    sup = _mrmr_support(df, y, 0)
    assert len(set(sig) & sup) >= 2, f"MRMR NaN medium/{kind}: signal lost: {sup}"
    assert len(set(noi) & sup) <= 1, f"MRMR NaN medium/{kind}: noise leaked: {sup}"


def test_mrmr_mixed_categorical_keeps_informative_drops_noise_cat():
    """MRMR consumes string/category columns natively: keeps the informative categorical, drops the
    high-cardinality noise categorical. Majority of seeds."""
    kept_inf, kept_noisecat = 0, 0
    for s in _SEEDS:
        df, y, _sig, _noi = _make("binary", 2000, s, with_cat=True)
        sup = _mrmr_support(df, y, s)
        kept_inf += "cat_inf" in sup
        kept_noisecat += "cat_noise" in sup
    assert kept_inf >= 2, f"MRMR should keep the informative categorical on a majority of seeds; kept {kept_inf}/3"
    assert kept_noisecat <= 1, f"MRMR should drop the noise categorical on a majority of seeds; kept {kept_noisecat}/3"


# --- RFECV: NaN-heavy (HistGradientBoosting handles NaN natively) -------------------------------


@pytest.mark.parametrize("kind", ["regression", "binary"])
def test_rfecv_nan_heavy_recovers_signal(kind):
    """RFECV with HistGradientBoosting (native NaN support): 30% NaN, n=2000 -- robustness contract is no-crash +
    signal recovery on a majority of seeds. Noise pruning is NOT asserted: RFECV's default argmax rule is a weak
    pruner (measured noise-exclusion ~0 here -- it keeps the full set when the CV score plateaus), which is the
    documented behavior, not a regression. The non-crash + signal-recovery on 30% NaN is the robustness win."""
    reg = kind == "regression"
    recs = []
    for s in _SEEDS:
        df, y, sig, _noi = _make(kind, 2000, s, nan_frac=0.3)
        sup = _rfecv_support(df, y, s, reg)
        recs.append(len(set(sig) & sup))
    assert sum(r >= 2 for r in recs) >= 2, f"RFECV NaN/{kind}: expected >=2/3 signal on majority of seeds; per-seed={recs}"


def test_rfecv_nan_heavy_medium_n_regression():
    """RFECV NaN medium n=15000 (regression): trains + recovers signal."""
    df, y, sig, _noi = _make("regression", 15000, 0, nan_frac=0.3)
    sup = _rfecv_support(df, y, 0, regression=True)
    assert len(set(sig) & sup) >= 2, f"RFECV NaN medium: signal lost: {sup}"
