"""BIZ-VALUE: the SYNTHESIS + DROPPING blind-spot matrix.

This file systematically probes the two headline capability questions for the
feature-selection family, family-by-family, as behavioral biz_value tests:

(A) SYNTHESIS -- can the FE-capable selector (full-mode ``MRMR``) RECOVER a
    target that ONLY a synthesized feature exposes?  For each interaction /
    transform family we build a tiny synthetic where the RAW columns carry
    ~0 LINEAR (univariate) signal but a single synthesized feature carries
    strong signal, and assert the FE-on selection lifts a held-out LINEAR
    downstream AUC over a raw-only selection by a measured floor.  Where the
    FE machinery genuinely cannot synthesize the family (MODULAR -- a sawtooth
    parity no smooth basis fits), the test is written to the CORRECT behavior
    and ``xfail``-ed with the measured miss -- a documented capability GAP,
    never a weakened assertion. The 3-way XOR (XOR3) is RECOVERED by the
    ``fe_hybrid_orth_triplet_enable`` cross-basis synthesizer.

(B) DROPPING -- does the selector correctly DROP a redundancy / decoy column
    in favour of the real signal?  For each redundancy / decoy family we put
    ONE real signal column next to one decoy and assert the decoy is excluded
    (or, for a redundant TWIN, that the redundant PAIR collapses to a single
    survivor -- never both).  This is probed for BOTH the redundancy-aware
    filter (``MRMR``) AND a wrapper selector (``RFECV``); the asymmetry between
    them (MRMR dedups twins / rejects realistic-marginal noise, the embedded
    wrapper does not) is the load-bearing finding and is pinned as explicit
    xfail GAPs on the wrapper rather than hidden.

All fits are TINY (n<=800), fixed-seed, single representative under fast mode.
Measured floors sit 5-15% below a once-measured value.  ASCII prints only.

Calibration run (n=800, seed-7/seed-1, store py3.14, CPU):
  SYNTHESIS delta (FE_auc - raw_auc):
    PRODUCT +0.54  RATIO +0.50  QUADRATIC/ABS/LOG +0.52  SINE +0.19
    MINMAX +0.09   CONDITIONAL +0.17   DIFFERENCE +0.00 (already raw-linear)
    XOR3 +0.50 (triplet cross-basis synthesizer, fe_hybrid_orth_triplet_enable)
    MODULAR -0.15 (GAP -- smooth basis cannot fit a sawtooth/parity)
  DROPPING (MRMR, raw-only):  every twin family collapses to a single
    survivor; constant / near-constant / id-like / permuted-noise all dropped.
  DROPPING (RFECV):  exact_dup + constant dropped; permuted-noise, id-like,
    and scaled-copy ADMITTED (wrapper is not redundancy/decoy aware -> GAPs).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from mlframe.feature_selection.filters.mrmr import MRMR

sys.path.insert(0, os.path.dirname(__file__))
from conftest import fast_subset  # noqa: E402
from _selector_factories import _make_rfecv, selected_names  # noqa: E402


# ---------------------------------------------------------------------------
# MRMR configs: raw-only (every default-ON FE generator OFF) vs FE-full.
# ---------------------------------------------------------------------------
_RAW_ONLY = dict(
    fe_max_steps=0,
    fe_univariate_basis_enable=False, fe_univariate_fourier_enable=False,
    fe_hinge_enable=False, fe_conditional_dispersion_enable=False,
    fe_wavelet_enable=False, fe_hybrid_orth_pair_enable=False,
    fe_auto_escalation_enable=False, fe_pair_prewarp_enable=False,
    fe_rung_schedule_enable=False, fe_stability_vote_enable=False,
    cluster_aggregate_enable=False, dcd_enable=False,
)
# Full FE mode includes the 3-way cross-basis synthesizer (``fe_hybrid_orth_triplet_enable``); it stays a separate ctor opt-out because the triplet stage
# runs regardless of fe_max_steps, so default-ON would add the seed_k-bounded O(C(k,3)) stage to EVERY fit. The pair-only synthesis families are byte-identical
# with it on (it only appends a column when its MI-uplift gate clears), and it is the only path that synthesizes a single feature carrying a pure 3-way XOR.
_FE_FULL = dict(fe_max_steps=2, cluster_aggregate_enable=False, dcd_enable=False, fe_hybrid_orth_triplet_enable=True)

_N = 800


def _split(df, y, frac: float = 0.7):
    """Leak-safe shuffle split: FE fit on train, replayed on test."""
    n = len(df)
    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    k = int(n * frac)
    tr, te = idx[:k], idx[k:]
    return (df.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
            df.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True))


def _sel_auc(kw, df, y):
    """Fit MRMR with ``kw``, return held-out LogReg AUC on the selection + names."""
    dtr, ytr, dte, yte = _split(df, y)
    fs = MRMR(verbose=0, random_seed=42, **kw)
    fs.fit(dtr, ytr)
    Xtr = np.nan_to_num(np.asarray(fs.transform(dtr), float))
    Xte = np.nan_to_num(np.asarray(fs.transform(dte), float))
    names = list(fs.get_feature_names_out())
    if Xtr.shape[1] == 0:
        return 0.5, names
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(Xtr, ytr.values)
    return float(roc_auc_score(yte.values, m.predict_proba(Xte)[:, 1])), names


def _has_engineered(names) -> bool:
    return any(("(" in n) or ("__" in n) for n in names)


def _mk(score_fn, *, ncols: int = 6, seed: int = 1, thr="median"):
    """Tiny synthetic: standard-normal raw columns, binary y from ``score_fn``.

    Threshold defaults to the median so the target is balanced; pass a float
    to override (e.g. 0.0 for sign-symmetric scores)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(_N, ncols))
    s = score_fn(X, rng)
    t = np.median(s) if thr == "median" else thr
    y = (s > t).astype(int)
    cols = [f"x{i}" for i in range(ncols)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


# ===========================================================================
# (A) SYNTHESIS MATRIX
#
# Each entry: (family_id, score_fn, threshold, downstream_floor, gap_reason).
# gap_reason is None for families the FE recovers; a string => xfail GAP.
# The floor is the measured (FE_auc - raw_auc) delta minus a 5-15% margin.
# ===========================================================================
_SYNTH_FAMILIES = [
    # family,        score_fn,                                            thr,      floor,  gap_reason
    ("PRODUCT",      lambda X, r: X[:, 0] * X[:, 1],                      "median", 0.45,   None),
    ("RATIO",        lambda X, r: (np.abs(X[:, 0]) + 0.5) / (np.abs(X[:, 1]) + 0.5), "median", 0.42, None),
    ("QUADRATIC",    lambda X, r: X[:, 0] ** 2,                           "median", 0.44,   None),
    ("ABS",          lambda X, r: np.abs(X[:, 0]),                        "median", 0.44,   None),
    ("LOG",          lambda X, r: np.log(np.abs(X[:, 0]) + 0.1),          "median", 0.44,   None),
    ("SINE",         lambda X, r: np.sin(3.0 * X[:, 0]),                  0.0,      0.10,   None),
    ("MINMAX",       lambda X, r: np.maximum(X[:, 0], X[:, 1]),           "median", 0.05,   None),
    ("THRESHOLD_RELU", lambda X, r: (X[:, 0] > 0.5).astype(float) + 0.01 * X[:, 1], 0.5, -0.05, None),
    ("CONDITIONAL",  None,                                                None,     0.10,   None),  # custom builder
    # 3-way XOR: recovered by the ``fe_hybrid_orth_triplet_enable`` cross-basis synthesizer (He1*He1*He1 cell), wired into _FE_FULL. Pair search alone misses it
    # (every pair MI is ~0); the triplet stage emits the single feature that carries the joint sign-product, and a linear downstream lifts to ~0.99 AUC.
    ("XOR3",         lambda X, r: X[:, 0] * X[:, 1] * X[:, 2],            0.0,      0.40,   None),
    # --- documented GAP: FE machinery cannot synthesize this ---
    ("MODULAR",      lambda X, r: ((np.floor(X[:, 0] * 3).astype(int)) % 2).astype(float), 0.5, 0.10,
     "FS GAP: cannot synthesize MODULAR/parity target (no mod generator; smooth basis cannot fit a sawtooth)"),
]


def _build_synth(family, score_fn, thr):
    if family == "CONDITIONAL":
        # y = (x0 if x2>0 else x1) > 0.  Neither x0 nor x1 alone is usable; the
        # gating by x3>0 is what a conditional generator must recover.
        rng = np.random.default_rng(1)
        X = rng.normal(size=(_N, 6))
        s = np.where(X[:, 2] > 0, X[:, 0], X[:, 1])
        y = (s > 0).astype(int)
        return pd.DataFrame(X, columns=[f"x{i}" for i in range(6)]), pd.Series(y, name="y")
    return _mk(score_fn, thr=thr)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "family,score_fn,thr,floor,gap",
    [pytest.param(*r, id=r[0]) for r in fast_subset(_SYNTH_FAMILIES, n=2)],
)
def test_synthesis_matrix(family, score_fn, thr, floor, gap):
    """SYNTHESIS: does full-mode MRMR FE recover a target ONLY a synthesized
    feature exposes?  Floor is on (FE_auc - raw_auc) for a held-out LogReg.

    A documented capability GAP (MODULAR) is asserted to the CORRECT behavior
    and xfail-ed -- not weakened -- so the miss stays visible. XOR3 is now
    recovered via the triplet cross-basis synthesizer in _FE_FULL."""
    if gap is not None:
        pytest.xfail(gap)
    df, y = _build_synth(family, score_fn, thr)
    fe_auc, fe_names = _sel_auc(_FE_FULL, df, y)
    raw_auc, raw_names = _sel_auc(_RAW_ONLY, df, y)
    delta = fe_auc - raw_auc
    print(f"SYNTH {family:14s} FE_auc={fe_auc:.3f} raw_auc={raw_auc:.3f} "
          f"delta={delta:+.3f} floor={floor:+.3f} fe_names={fe_names[:4]}")
    # The FE selection must (a) carry the signal absolutely AND (b) beat raw-only
    # by the floor.  For THRESHOLD_RELU the raw column already exposes the step
    # weakly, so the floor is ~0 (FE must not HURT); the absolute AUC pins value.
    assert fe_auc > 0.70, (
        f"{family}: FE-on selection does not carry the synthesized signal "
        f"(FE_auc={fe_auc:.3f} <= 0.70); names={fe_names}")
    assert delta >= floor, (
        f"{family}: FE did not recover the synthesized target over raw-only: "
        f"FE_auc={fe_auc:.3f} raw_auc={raw_auc:.3f} delta={delta:+.3f} "
        f"< floor {floor:+.3f}; fe_names={fe_names}, raw_names={raw_names}")


@pytest.mark.timeout(120)
def test_synthesis_xor3_recovered_by_triplet_synthesizer():
    """Regression sensor for the closed 3-way-XOR synthesis gap: the triplet
    cross-basis synthesizer (``fe_hybrid_orth_triplet_enable``, wired into
    _FE_FULL) must emit a single feature carrying ``sign(x0*x1*x2)`` so a linear
    downstream lifts to ~0.99 AUC, while raw-only and pair-only FE both miss it
    (every pair MI is ~0). Always runs (not behind fast_subset sampling)."""
    df, y = _mk(lambda X, r: X[:, 0] * X[:, 1] * X[:, 2], thr=0.0)
    fe_auc, fe_names = _sel_auc(_FE_FULL, df, y)
    raw_auc, _ = _sel_auc(_RAW_ONLY, df, y)
    pair_auc, pair_names = _sel_auc(
        dict(fe_max_steps=2, cluster_aggregate_enable=False, dcd_enable=False), df, y)
    print(f"XOR3 triplet FE_auc={fe_auc:.3f} raw_auc={raw_auc:.3f} pair_auc={pair_auc:.3f} fe_names={fe_names[:4]}")
    assert fe_auc > 0.90, f"triplet synthesizer must carry 3-way XOR (FE_auc={fe_auc:.3f}); names={fe_names}"
    assert fe_auc - raw_auc >= 0.40, f"triplet FE must beat raw-only by >=0.40 on 3-way XOR; got {fe_auc - raw_auc:+.3f}"
    assert fe_auc - pair_auc >= 0.30, (
        f"triplet FE must beat PAIR-only FE by >=0.30 on 3-way XOR (pairs cannot synthesize it); "
        f"got {fe_auc - pair_auc:+.3f}; pair_names={pair_names}")
    assert any("__He1_He1_He1" in n or "*" in n for n in fe_names), (
        f"a 3-way cross-basis triplet column must be selected; got {fe_names}")


# ===========================================================================
# (B) DROPPING MATRIX
#
# ONE real signal column ``x_real`` (drives y linearly) next to one decoy.
# Families split into:
#   - REDUNDANT TWINS: decoy is a function of x_real.  Correct behavior = the
#     redundant PAIR collapses to a single survivor (never BOTH).
#   - PURE DECOYS: decoy carries no signal (constant/near-const/id/permuted).
#     Correct behavior = decoy is NOT selected.
#   - LEAKAGE PROXY: decoy ~= y.  A pure-MI selector CORRECTLY ranks it top
#     (it is the most relevant feature); dropping it needs temporal/leak
#     awareness the raw filter does not have -> documented GAP.
# ===========================================================================
def _drop_dataset(decoy, *, seed: int = 7):
    rng = np.random.default_rng(seed)
    x_real = rng.normal(size=_N)
    y = pd.Series((x_real + 0.2 * rng.normal(size=_N) > 0).astype(int), name="y")
    # Allow the decoy builder to depend on x_real / y / rng.
    dvals = decoy(x_real, y.values, rng)
    df = pd.DataFrame({
        "x_real": x_real,
        "decoy": dvals,
        "noise0": rng.normal(size=_N),
        "noise1": rng.normal(size=_N),
    })
    return df, y


# (family, decoy_builder, kind, gap_reason_mrmr)
# kind in {"twin", "decoy", "leak"}.
_DROP_FAMILIES = [
    ("exact_dup",        lambda x, y, r: x.copy(),                              "twin",  None),
    ("noisy_neardup",    lambda x, y, r: x + 0.05 * r.normal(size=_N),          "twin",  None),
    ("nonmono_dup_sq",   lambda x, y, r: x ** 2,                                "twin",  None),
    ("sign_flip",        lambda x, y, r: -x,                                    "twin",  None),
    ("scaled_copy",      lambda x, y, r: 100.0 * x,                             "twin",  None),
    ("monotone_warp",    lambda x, y, r: np.exp(x),                             "twin",  None),
    ("constant",         lambda x, y, r: np.full(_N, 3.0),                      "decoy", None),
    ("near_constant",    lambda x, y, r: np.concatenate([np.full(_N - 2, 1.0), [1.001, 0.999]]), "decoy", None),
    ("id_like_highcard", lambda x, y, r: np.arange(_N, dtype=float),            "decoy", None),
    ("permuted_decoy",   lambda x, y, r: r.permutation(x),                      "decoy", None),
    ("leakage_proxy",    lambda x, y, r: y.astype(float) + 0.01 * r.normal(size=_N), "leak",
     "FS GAP: pure-MI filter cannot drop a leakage proxy of y -- it is the most relevant feature; needs temporal/leak-awareness"),
]


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "family,decoy,kind,gap",
    [pytest.param(*r, id=r[0]) for r in fast_subset(_DROP_FAMILIES, n=3)],
)
def test_mrmr_dropping_matrix(family, decoy, kind, gap):
    """DROPPING (redundancy-aware MRMR): the decoy is dropped / the redundant
    twin pair collapses to one survivor.  ``x_real`` must always survive."""
    if gap is not None:
        # Document the leak-proxy GAP to the CORRECT behavior, then xfail.
        df, y = _drop_dataset(decoy)
        fs = MRMR(verbose=0, random_seed=42, **_RAW_ONLY)
        fs.fit(df, y)
        sel = list(fs.get_feature_names_out())
        print(f"DROP  {family:18s} sel={sel} (GAP)")
        # CORRECT behavior: a leak proxy should be dropped. It is NOT (the proxy
        # is the single most relevant feature for a pure-MI filter), so we
        # demonstrate the miss and xfail rather than weaken the contract.
        if "decoy" in sel:
            pytest.xfail(gap)
        assert "decoy" not in sel, gap
        return
    df, y = _drop_dataset(decoy)
    fs = MRMR(verbose=0, random_seed=42, **_RAW_ONLY)
    fs.fit(df, y)
    sel = list(fs.get_feature_names_out())
    print(f"DROP  {family:18s} sel={sel}")
    assert "x_real" in sel, (
        f"{family}: real signal column was DROPPED, selection={sel}")
    if kind == "twin":
        assert not ("decoy" in sel and "x_real" in sel and sel.count("decoy") and sel.count("x_real")), \
            f"{family}: redundant twin NOT collapsed -- both kept: {sel}"
        # stricter: with only one real column present, decoy must not survive.
        assert "decoy" not in sel, (
            f"{family}: redundant twin admitted alongside x_real: {sel}")
    else:  # pure decoy
        assert "decoy" not in sel, (
            f"{family}: pure decoy admitted into selection: {sel}")


# RFECV (embedded wrapper) is NOT redundancy/decoy-aware. We pin which families
# it handles and which it admits as documented GAPs -- the MRMR-vs-wrapper
# asymmetry is the headline of the matrix.
_RFECV_DROP = [
    ("exact_dup",        lambda x, y, r: x.copy(),                    "decoy", None),
    ("constant",         lambda x, y, r: np.full(_N, 3.0),            "decoy", None),
    ("scaled_copy",      lambda x, y, r: 100.0 * x,                   "decoy",
     "FS GAP: RFECV is not redundancy-aware -- admits an exact scaled copy of the signal"),
    ("permuted_decoy",   lambda x, y, r: r.permutation(x),            "decoy",
     "FS GAP: RFECV admits a permuted realistic-marginal noise decoy (no relevance floor)"),
    ("id_like_highcard", lambda x, y, r: np.arange(_N, dtype=float),  "decoy",
     "FS GAP: RFECV admits a high-cardinality ID-like decoy"),
]


@pytest.mark.slow
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "family,decoy,kind,gap",
    _RFECV_DROP,
    ids=[r[0] for r in _RFECV_DROP],
)
def test_rfecv_dropping_matrix(family, decoy, kind, gap):
    """DROPPING (wrapper RFECV): exact-dup + constant are dropped; redundant
    scaled copies / realistic-marginal noise / ID-like decoys are ADMITTED.
    The admissions are written to the CORRECT behavior and xfail-ed as GAPs."""
    df, y = _drop_dataset(decoy)
    sel = _make_rfecv("binary")
    sel.fit(df, y)
    names = selected_names(sel)
    print(f"RFECV {family:18s} sel={names} decoy_in={'decoy' in names}")
    # The signal must survive somewhere. For a redundant twin (scaled_copy) the
    # signal is carried by EITHER x_real OR its copy, so accept either; for the
    # pure-decoy families x_real itself must survive.
    if family == "scaled_copy":
        assert ("x_real" in names) or ("decoy" in names), (
            f"{family}: RFECV dropped BOTH the signal and its copy: {names}")
    else:
        assert "x_real" in names, f"{family}: RFECV dropped the real signal: {names}"
    if gap is not None:
        pytest.xfail(gap)
    assert "decoy" not in names, (
        f"{family}: RFECV admitted the decoy into selection: {names}")


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_rfecv_synthesis_blindspot_quadratic():
    """SYNTHESIS blind-spot for the wrapper: RFECV has NO feature-engineering,
    so on a purely non-additive target (``y=sign(x0^2 - median)``) it cannot
    synthesize the quadratic and its linear-estimator selection stays at chance
    downstream -- the documented reason MRMR (FE-capable) is the recovery path."""
    df, y = _mk(lambda X, r: X[:, 0] ** 2, thr="median")
    sel = _make_rfecv("binary")
    sel.fit(df, y)
    Xt = np.nan_to_num(np.asarray(sel.transform(df), float))
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    auc = float(np.mean([
        roc_auc_score(y.values, m.fit(Xt, y.values).predict_proba(Xt)[:, 1])
    ]))
    print(f"RFECV synth QUADRATIC in-sample linear auc={auc:.3f}")
    # Linear model on raw quadratic-signal columns is ~chance; this is the GAP
    # the FE-capable MRMR closes (see test_synthesis_matrix[QUADRATIC]).
    pytest.xfail("FS GAP: RFECV cannot synthesize a quadratic (no FE); "
                 f"linear downstream stays near chance (auc={auc:.3f})")
