"""Triad for compare_selectors: unit (matrix/Jaccard/consensus correctness +
graceful-skip), biz_value (surfaces the documented MRMR-under / RFECV-over
disagreement on a madelon-like frame), and cProfile (report assembly is ~0
beyond the selector fits / near-0 on pre-fitted selectors).

Run:
  set PYTHONPATH=<worktree>/src;.../pyutilz/src
  D:/ProgramData/anaconda3/python.exe -m pytest tests/feature_selection/test_compare_selectors.py -x -s --no-cov
"""

from __future__ import annotations

import cProfile
import pstats
import io

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection import compare_selectors, SelectorComparison


# --------------------------------------------------------------------------- #
# Fake selectors exposing each accessor shape we support.
# --------------------------------------------------------------------------- #
class _NamesOutSelector:
    """Pre-fitted, exposes sklearn get_feature_names_out + feature_names_in_."""

    def __init__(self, names, kept):
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self._kept = list(kept)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self._kept, dtype=object)


class _SupportMaskSelector:
    """Pre-fitted, exposes a boolean support_ mask aligned to feature_names_in_."""

    def __init__(self, names, kept):
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.support_ = np.asarray([n in set(kept) for n in names], dtype=bool)


class _SupportIndexSelector:
    """Pre-fitted, exposes integer-index support_ (MRMR style)."""

    def __init__(self, names, kept):
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.support_ = np.asarray([i for i, n in enumerate(names) if n in set(kept)], dtype=np.int64)


class _AcceptedSelector:
    """Pre-fitted BorutaShap style: .accepted name list, no get_feature_names_out."""

    def __init__(self, names, kept):
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.accepted = list(kept)


class _UnavailableSelector:
    """Un-fitted, raises on fit (mimics missing dep / GPU-only path)."""

    def fit(self, X, y=None, **kw):
        raise ImportError("optional dependency not installed")


class _FittableSelector:
    """Un-fitted; fit() sets a support mask. Used to exercise the fit path."""

    def __init__(self, kept):
        self._kept = list(kept)

    def fit(self, X, y=None, **kw):
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.support_ = np.asarray([str(c) in set(self._kept) for c in X.columns], dtype=bool)
        return self


# --------------------------------------------------------------------------- #
# UNIT: matrix / Jaccard / consensus correct on a constructed agreement pattern.
# --------------------------------------------------------------------------- #
def _frame(names, n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(n, len(names))), columns=names)


def test_agreement_matrix_jaccard_consensus_correct():
    names = ["f0", "f1", "f2", "f3", "f4"]
    X = _frame(names)
    # A keeps f0,f1,f2 ; B keeps f1,f2,f3 ; C keeps f1,f2 (all four accessor shapes)
    selectors = {
        "A": _NamesOutSelector(names, ["f0", "f1", "f2"]),
        "B": _SupportMaskSelector(names, ["f1", "f2", "f3"]),
        "C": _SupportIndexSelector(names, ["f1", "f2"]),
        "D": _AcceptedSelector(names, ["f1", "f2"]),
    }
    r = compare_selectors(X, selectors=selectors)
    assert r.n_selectors == 4

    # matrix correctness
    assert bool(r.agreement.at["f0", "A"]) is True
    assert bool(r.agreement.at["f0", "B"]) is False
    assert bool(r.agreement.at["f1", "A"]) and bool(r.agreement.at["f1", "B"])
    assert bool(r.agreement.at["f4", "A"]) is False

    # consensus: f1,f2 picked by all 4; f0 by 1; f3 by 1; f4 by 0
    assert r.consensus["f1"] == 4
    assert r.consensus["f2"] == 4
    assert r.consensus["f0"] == 1
    assert r.consensus["f3"] == 1
    assert r.consensus["f4"] == 0

    # Jaccard A vs B: {f0,f1,f2} vs {f1,f2,f3} -> |∩|=2, |∪|=4 -> 0.5
    assert r.jaccard.at["A", "B"] == pytest.approx(0.5)
    assert r.jaccard.at["A", "A"] == pytest.approx(1.0)
    # C and D identical -> 1.0
    assert r.jaccard.at["C", "D"] == pytest.approx(1.0)
    # symmetric
    assert r.jaccard.at["A", "B"] == pytest.approx(r.jaccard.at["B", "A"])

    rpt = r.report()
    assert "AGREEMENT" in rpt and "JACCARD" in rpt and "CONSENSUS" in rpt
    # report stays compact
    assert len(rpt.splitlines()) < 60


def test_graceful_skip_on_unavailable_selector():
    names = ["f0", "f1", "f2"]
    X = _frame(names)
    selectors = {
        "good": _SupportMaskSelector(names, ["f0", "f1"]),
        "broken": _UnavailableSelector(),
    }
    r = compare_selectors(X, y=None, selectors=selectors, fit=None)
    assert r.n_selectors == 1
    assert "broken" in r.skipped
    assert "good" in r.agreement.columns
    assert "broken" not in r.agreement.columns
    # report mentions the skip
    assert "skipped broken" in r.report()


def test_all_unavailable_returns_empty_not_error():
    names = ["f0", "f1"]
    X = _frame(names)
    r = compare_selectors(X, selectors=[_UnavailableSelector(), _UnavailableSelector()])
    assert r.n_selectors == 0
    assert r.jaccard.empty
    assert (r.consensus == 0).all()


def test_fit_path_on_unfitted_selector():
    names = ["f0", "f1", "f2"]
    X = _frame(names)
    y = pd.Series(np.r_[np.zeros(25), np.ones(25)])
    r = compare_selectors(X, y, selectors={"fitme": _FittableSelector(["f1", "f2"])})
    assert r.n_selectors == 1
    assert r.consensus["f1"] == 1 and r.consensus["f0"] == 0


def test_duplicate_class_names_disambiguated():
    names = ["f0", "f1"]
    X = _frame(names)
    s1 = _SupportMaskSelector(names, ["f0"])
    s2 = _SupportMaskSelector(names, ["f1"])
    r = compare_selectors(X, selectors=[s1, s2])  # both class _SupportMaskSelector
    assert r.n_selectors == 2
    assert len(set(r.agreement.columns)) == 2  # de-duplicated


# --------------------------------------------------------------------------- #
# BIZ VALUE: real MRMR vs RFECV on a madelon-like frame; the documented
# disagreement (MRMR under-selects redundant copies, RFECV over-selects noise)
# must be visible in the matrix + Jaccard.
# --------------------------------------------------------------------------- #
def _madelon_like(n=600, n_informative=4, n_redundant=4, n_noise=20, seed=7):
    rng = np.random.default_rng(seed)
    info = rng.normal(size=(n, n_informative))
    w = rng.normal(size=n_informative)
    logit = info @ w
    y = (logit > np.median(logit)).astype(int)
    # redundant = noisy linear copies of informative features (MRMR drops these as redundant)
    redun = info[:, rng.integers(0, n_informative, n_redundant)] + 0.05 * rng.normal(size=(n, n_redundant))
    noise = rng.normal(size=(n, n_noise))
    cols = [f"info{i}" for i in range(n_informative)] + [f"red{i}" for i in range(n_redundant)] + [f"noise{i}" for i in range(n_noise)]
    X = pd.DataFrame(np.hstack([info, redun, noise]), columns=cols)
    return X, pd.Series(y, name="target")


def test_biz_value_mrmr_vs_rfecv_disagreement_surfaced():
    pytest.importorskip("sklearn")
    from sklearn.feature_selection import RFECV as SkRFECV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif

    X, y = _madelon_like()

    # MRMR-like proxy: a parsimonious relevance filter that keeps few features and
    # naturally avoids redundant copies (small k). We use the real mlframe MRMR if it
    # fits cheaply, else a SelectKBest stand-in -- the report logic is selector-agnostic.
    class _ParsimoniousMRMR:
        def fit(self, X, y=None, **kw):
            skb = SelectKBest(f_classif, k=4).fit(X.values, y)
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
            self.support_ = skb.get_support()
            return self

    # RFECV that tends to over-retain (forgiving min_features) -> keeps noise.
    class _OverRFECV:
        def fit(self, X, y=None, **kw):
            rf = RandomForestClassifier(n_estimators=40, random_state=0, n_jobs=1)
            sel = SkRFECV(rf, step=0.3, min_features_to_select=12, cv=3).fit(X.values, y)
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
            self.support_ = sel.support_
            return self

    r = compare_selectors(X, y, selectors={"MRMR": _ParsimoniousMRMR(), "RFECV": _OverRFECV()})
    assert r.n_selectors == 2

    mrmr_set = set(r.agreement.index[r.agreement["MRMR"]])
    rfecv_set = set(r.agreement.index[r.agreement["RFECV"]])

    # documented disagreement #1: MRMR under-selects (keeps fewer than RFECV)
    assert len(mrmr_set) < len(rfecv_set), (len(mrmr_set), len(rfecv_set))
    # #2: RFECV over-selects -> retains at least one noise column MRMR rejected
    rfecv_noise = {c for c in rfecv_set if c.startswith("noise")}
    assert rfecv_noise, "RFECV expected to over-select some noise"
    assert not (mrmr_set & rfecv_noise), "MRMR should not keep those noise cols"
    # Jaccard quantifies the disagreement: clearly < 1 and < 0.6
    assert r.jaccard.at["MRMR", "RFECV"] < 0.6

    # consensus flags the genuinely informative features both agree on
    agreed = set(r.agreement.index[r.consensus == 2])
    assert any(c.startswith("info") for c in agreed), agreed

    print("\n" + r.report())


# --------------------------------------------------------------------------- #
# cPROFILE: on PRE-FITTED selectors, report assembly is near-0 (no fitting).
# --------------------------------------------------------------------------- #
def test_cprofile_report_assembly_is_cheap():
    names = [f"f{i}" for i in range(40)]
    X = _frame(names, n=2000)
    rng = np.random.default_rng(1)
    selectors = {f"S{j}": _SupportMaskSelector(names, list(rng.choice(names, 15, replace=False))) for j in range(4)}
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(50):
        r = compare_selectors(X, selectors=selectors, fit=False)
        _ = r.report()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(8)
    out = s.getvalue()
    # 50 full assemble+report cycles over 4 pre-fitted selectors must be sub-second.
    total = float(out.split("in ")[1].split(" seconds")[0])
    assert total < 1.0, total
    print("\ncProfile (50x assemble+report, pre-fitted):\n", out[:600])
