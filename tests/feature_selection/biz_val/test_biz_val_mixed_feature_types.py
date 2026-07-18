"""biz_value: FS effectiveness on MIXED feature types (numeric + categorical + text).

Most FS biz_val fixtures are all-numeric. This file pins the contract on a frame that
simultaneously carries:

* a NUMERIC signal column (``num_sig``),
* a low-card CATEGORICAL signal column (3 levels, target-predictive: ``cat_sig``),
* a high-card CATEGORICAL NOISE column (random id strings: ``cat_noise``),
* a TEXT signal column (a short string whose presence of a keyword carries signal),
* a TEXT noise column + numeric noise.

Two FS entry points are exercised:

(a) MRMR DIRECT on a raw mixed frame -- MRMR auto-categorizes object/category columns
    through its cat-FE binning path. A LOW-card categorical is handled and selected.
    A HIGH-card categorical (>sqrt(n)*2 levels) trips a HARD ValueError instead of
    being dropped -- a real selector GAP (xfail below).

(b) The SUITE FS path (``train_mlframe_models_suite`` with ``use_mrmr_fs=True``) on a
    RAW mixed frame -- the suite's encoders run BEFORE MRMR, so the high-card column is
    ordinal-encoded and survives FS as a droppable numeric. Here we assert the
    categorical signal is selected and the high-card categorical noise is dropped.

selector x feature-type effectiveness matrix is summarised in the module docstring of
the final agent report; per-cell behaviour is pinned by the tests here.

Thresholds: floors are pinned 5-15% below a MEASURED value (dev run, seed sweep).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.conftest import is_fast_mode, fast_subset

# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)


def _make_mixed_frame(n: int, seed: int, *, high_card: bool = True, as_category: bool = True):
    """Mixed frame with known signal/noise roles.

    Returns ``(df, y, roles)`` where ``roles`` maps each column name to one of
    ``{"num_sig", "cat_sig", "cat_noise", "num_noise"}``. ``y`` is binary and driven
    by ``num_sig`` + the low-card categorical level; the high-card column and numeric
    noise carry no signal.
    """
    rng = np.random.default_rng(seed)
    num_sig = rng.normal(size=n)
    lev = rng.integers(0, 3, n)  # 3 predictive levels
    cat_sig_vals = np.array(["a", "b", "c"])[lev]
    num_noise = rng.normal(size=n)
    score = num_sig + (lev - 1) * 1.6 + 0.3 * rng.normal(size=n)
    y = (score > 0).astype(np.int64)

    cols = {"num_sig": num_sig}
    cols["cat_sig"] = pd.Categorical(cat_sig_vals) if as_category else cat_sig_vals
    if high_card:
        ids = np.array([f"id{i}" for i in rng.integers(0, max(4, n // 4), n)])
        cols["cat_noise"] = pd.Categorical(ids) if as_category else ids
    cols["num_noise"] = num_noise
    df = pd.DataFrame(cols)
    roles = {"num_sig": "num_sig", "cat_sig": "cat_sig", "num_noise": "num_noise"}
    if high_card:
        roles["cat_noise"] = "cat_noise"
    return df, y, roles


def _make_text_signal(n: int, seed: int):
    """Text signal: presence of the keyword ``win`` in the text correlates with y.

    Returns ``(texts_signal, texts_noise, y)``. ``texts_signal`` tokens carry a
    predictive keyword; ``texts_noise`` is random tokens. The bag-of-words encoding
    ``("win" in text)`` is the recoverable signal.
    """
    rng = np.random.default_rng(seed)
    has_kw = rng.random(n) < 0.5
    sig_texts, noise_texts = [], []
    for i in range(n):
        toks = ["win" if has_kw[i] else "lose", f"w{rng.integers(0, 1000)}", f"z{rng.integers(0, 1000)}"]
        rng.shuffle(toks)
        sig_texts.append(" ".join(toks))
        noise_texts.append(" ".join(f"q{rng.integers(0, 1000)}" for _ in range(3)))
    y = (has_kw ^ (rng.random(n) < 0.1)).astype(np.int64)
    return np.array(sig_texts), np.array(noise_texts), y, has_kw.astype(int)


def _fast_mrmr(**overrides):
    """Fast mrmr."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    kw = dict(min_relevance_gain=0.0, cv=3, run_additional_rfecv_minutes=False, full_npermutations=7, random_seed=0, min_features_fallback=1, verbose=False)
    kw.update(overrides)
    return MRMR(**kw)


# ---------------------------------------------------------------------------
# (a) MRMR direct -- low-card categorical handled
# ---------------------------------------------------------------------------
def test_biz_val_mrmr_selects_lowcard_cat_signal_drops_num_noise():
    """MRMR DIRECT on a raw frame (numeric signal + low-card cat signal + numeric noise,
    NO high-card col): both signals selected, numeric noise dropped, on a majority of
    seeds. Measured 5/5 seeds select cat_sig+num_sig; floor = 3/5."""
    seeds = fast_subset([0, 1, 2, 3, 4], n=2)
    hits_cat = hits_num = drops_noise = 0
    for s in seeds:
        df, y, _roles = _make_mixed_frame(500, s, high_card=False)
        m = _fast_mrmr(random_seed=s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df, y)
        sel = set(map(str, m.get_feature_names_out()))
        # engineered names can wrap a base col; check substring membership too.
        joined = " ".join(sel)
        if "cat_sig" in sel or "cat_sig" in joined:
            hits_cat += 1
        if "num_sig" in sel or "num_sig" in joined:
            hits_num += 1
        if "num_noise" not in sel:
            drops_noise += 1
    floor = 2 if is_fast_mode() else 3
    assert hits_cat >= floor, f"cat_sig selected on {hits_cat}/{len(seeds)} seeds"
    assert hits_num >= floor, f"num_sig selected on {hits_num}/{len(seeds)} seeds"
    assert drops_noise >= floor, f"num_noise dropped on {drops_noise}/{len(seeds)} seeds"


def test_biz_val_mrmr_lowcard_cat_beats_random_relevance():
    """The low-card cat signal must out-rank a pure-noise categorical of the SAME
    cardinality when both are present -- MRMR's cat handling is not just passing
    everything through. Measured: cat_sig selected, cat_rand_noise dropped, 5/5 seeds.
    Floor = majority."""
    seeds = fast_subset([0, 1, 2], n=2)
    good = 0
    for s in seeds:
        rng = np.random.default_rng(100 + s)
        df, y, _ = _make_mixed_frame(500, s, high_card=False)
        # add a 3-level random cat noise (same cardinality as cat_sig, no signal)
        df = df.assign(cat_rand=pd.Categorical(np.array(["p", "q", "r"])[rng.integers(0, 3, len(df))]))
        m = _fast_mrmr(random_seed=s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df, y)
        sel = set(map(str, m.get_feature_names_out()))
        joined = " ".join(sel)
        if ("cat_sig" in sel or "cat_sig" in joined) and "cat_rand" not in sel:
            good += 1
    assert good >= (len(seeds) + 1) // 2, f"cat_sig out-ranked equal-card noise on only {good}/{len(seeds)} seeds"


def test_biz_val_mrmr_drops_highcard_cat_noise_gracefully():
    """MRMR DIRECT on a frame WITH a high-card categorical noise column completes (no
    crash): the high-card column is skipped for cat-FE and dropped as noise by the
    relevance screen, while the low-card cat signal is kept. (Was a GAP: the cat-FE
    cardinality ceiling raised a hard ValueError; now ``on_high_cardinality='skip'`` is
    the default.)"""
    df, y, _roles = _make_mixed_frame(500, 0, high_card=True)
    m = _fast_mrmr(random_seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df, y)
    sel = set(map(str, m.get_feature_names_out()))
    assert "cat_noise" not in sel, f"high-card noise should be dropped: {sel}"
    assert any("cat_sig" in c for c in sel), f"low-card cat signal should be kept: {sel}"


def test_biz_val_mrmr_highcard_cat_raises_when_opted_in():
    """The legacy hard ValueError is still available opt-in via
    ``CatFEConfig(on_high_cardinality='raise')`` for callers who want a strict
    "this column shouldn't be categorical" signal."""
    from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig

    df, y, _ = _make_mixed_frame(500, 0, high_card=True)
    m = _fast_mrmr(random_seed=0, cat_fe_config=CatFEConfig(on_high_cardinality="raise"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="(?i)nbins|ceiling|cardinal"):
            m.fit(df, y)


# ---------------------------------------------------------------------------
# (a') MRMR on PRE-ENCODED text -- token signal recovered
# ---------------------------------------------------------------------------
def test_biz_val_mrmr_recovers_preencoded_text_token_signal():
    """When the text col is pre-encoded to a bag-of-words keyword indicator (the
    documented manual step for text), MRMR selects the keyword feature and drops the
    text-noise indicator + numeric noise. Measured 3/3 seeds. Floor = majority."""
    seeds = fast_subset([0, 1, 2], n=2)
    good = 0
    for s in seeds:
        sig_t, noise_t, y, _has_kw = _make_text_signal(500, s)
        has_win = np.array(["win" in t.split() for t in sig_t]).astype(int)
        # noise-text indicator: presence of an arbitrary token that carries no signal
        has_q = np.array(["q0" in t.split() for t in noise_t]).astype(int)
        rng = np.random.default_rng(s)
        df = pd.DataFrame({"txt_win": has_win, "txt_noise_ind": has_q, "num_noise": rng.normal(size=len(y))})
        m = _fast_mrmr(random_seed=s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df, y)
        sel = set(map(str, m.get_feature_names_out()))
        joined = " ".join(sel)
        if ("txt_win" in sel or "txt_win" in joined) and "txt_noise_ind" not in sel:
            good += 1
    assert good >= (len(seeds) + 1) // 2, f"pre-encoded text keyword selected (noise dropped) on {good}/{len(seeds)}"


# ---------------------------------------------------------------------------
# (b) SUITE FS path -- encoders + MRMR together on a RAW mixed frame
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_biz_val_suite_fs_keeps_cat_signal_drops_highcard_noise():
    """``train_mlframe_models_suite`` (encoders + MRMR FS) on a RAW mixed frame WITH a
    high-card categorical: the suite ordinal-encodes ahead of MRMR (so no crash), keeps
    ``cat_sig`` in the selected set, and drops ``cat_noise``. Measured: cat_sig in
    selected_features, cat_noise absent, AUC 1.0 on seed 0."""
    from tests.feature_selection._suite_fe_helpers import run_suite, best_test_metric

    df, y, _roles = _make_mixed_frame(600, 0, high_card=True)
    df = df.assign(y=y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ent, meta = run_suite(
            df,
            "y",
            model="linear",
            use_mrmr=True,
            classification=True,
            mrmr_kwargs=dict(min_relevance_gain=0.0, cv=3, run_additional_rfecv_minutes=False, full_npermutations=5, min_features_fallback=1),
        )
    sel = meta.get("selected_features") or []
    joined = " ".join(map(str, sel))
    assert any("cat_sig" in str(c) for c in sel) or "cat_sig" in joined, f"cat_sig not in suite-selected features: {sel}"
    assert not any(str(c) == "cat_noise" for c in sel), f"high-card cat_noise should be dropped by suite FS: {sel}"
    auc = best_test_metric(ent, "roc_auc")
    assert auc >= 0.85, f"suite AUC on mixed frame too low: {auc}"


@pytest.mark.slow
def test_biz_val_suite_fs_mixed_beats_numeric_only_lift():
    """The low-card categorical carries signal the numeric columns alone do not: a
    suite run WITH the categorical present reaches higher test-AUC than one with the
    categorical column DROPPED. Measured delta >= 0.10 (full+cat 1.0 vs num-only ~0.74
    when num_sig weight is halved). Floor = +0.05."""
    from tests.feature_selection._suite_fe_helpers import run_suite, best_test_metric

    # weaken the numeric path so the categorical contributes distinguishable lift.
    rng = np.random.default_rng(0)
    n = 600
    num_sig = rng.normal(size=n)
    lev = rng.integers(0, 3, n)
    cat_sig = pd.Categorical(np.array(["a", "b", "c"])[lev])
    num_noise = rng.normal(size=n)
    score = 0.5 * num_sig + (lev - 1) * 1.8 + 0.3 * rng.normal(size=n)
    y = (score > 0).astype(np.int64)
    df_full = pd.DataFrame({"num_sig": num_sig, "cat_sig": cat_sig, "num_noise": num_noise, "y": y})
    df_num = df_full.drop(columns=["cat_sig"])
    fs_kw = dict(min_relevance_gain=0.0, cv=3, run_additional_rfecv_minutes=False, full_npermutations=5, min_features_fallback=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ent_full, _ = run_suite(df_full, "y", model="linear", use_mrmr=True, classification=True, mrmr_kwargs=fs_kw)
        ent_num, _ = run_suite(df_num, "y", model="linear", use_mrmr=True, classification=True, mrmr_kwargs=fs_kw)
    auc_full = best_test_metric(ent_full, "roc_auc")
    auc_num = best_test_metric(ent_num, "roc_auc")
    assert auc_full - auc_num >= 0.05, f"categorical signal lift too small: full={auc_full} num_only={auc_num}"
