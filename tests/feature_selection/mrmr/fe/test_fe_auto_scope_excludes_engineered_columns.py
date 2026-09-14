"""Auto-scoped FE families must never take an engineered column as a source (mrmr_audit_2026-09-14 FEC-1, FEC-2, IMPL-6).

Every recipe is replayed one level deep at transform() time, so a family that builds on an engineered column emits a recipe whose parent
does not exist in the apply-time frame. Three auto-scoping paths still let engineered columns through:

* wavelet and rankgauss (``mid_b``) excluded only the ``hybrid_orth_features_`` ledger, but MI-greedy runs earlier and never merges into
  that ledger, so an MI-greedy column stayed eligible (FEC-1);
* count and frequency encoding (``early_b``) filtered engineered columns on the explicit-config branch only, handing the auto-detect
  branch the whole augmented frame (FEC-2);
* frequency encoding's exclusion set was snapshotted before count encoding ran, so it never contained the count outputs, integer columns
  well inside the auto-detect cardinality window (IMPL-6).

The engineered column is made deterministic by stubbing the MI-greedy constructor to emit one known column, and ``auto_detect_te_cols`` is
stubbed to admit every column, the worst case the scoping must survive. Downstream families are observed through spies that record the
columns they were handed; the count spy runs the real encoder so frequency encoding really sees count outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._count_freq_interaction_fe as cfi
import mlframe.feature_selection.filters._extra_fe_families as extra
import mlframe.feature_selection.filters._mi_greedy_fe as mig
import mlframe.feature_selection.filters._target_encoding_fe as te
import mlframe.feature_selection.filters._wavelet_basis_fe_recipes as wv
from mlframe.feature_selection.filters.mrmr import MRMR

_MIG_COL = "mig_fake_sq"


def _frame(n=600, seed=0):
    """Four raw numeric columns plus an integer column inside the auto-detect cardinality window."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})
    X["k"] = rng.integers(0, 12, size=n)
    y = ((X["x0"] + 0.5 * X["x1"] + 0.2 * rng.normal(size=n)) > 0).astype(int)
    return X, y


def _fit_with_spies(monkeypatch, **params):
    """Fit with the stubs installed; return what each spied family was handed, plus the raw input columns."""
    X, y = _frame()
    seen: dict = {"wavelet": [], "rankgauss": [], "count": [], "freq": [], "count_out": []}

    def _fake_mi_greedy(X_in, y_in, cols=None, **kwargs):
        """Emit exactly one engineered column, so an MI-greedy column is guaranteed to exist downstream."""
        out = X_in.copy()
        out[_MIG_COL] = X_in["x0"].to_numpy() ** 2
        return out, {}, []

    def _spy_wavelet(X_in, y_in, cols=None, **kwargs):
        """Record the wavelet source scope; append nothing."""
        seen["wavelet"].append((list(X_in.columns), list(cols or [])))
        return X_in, [], [], None

    def _spy_rankgauss(X_in, y_in, num_cols=None, **kwargs):
        """Record the rankgauss source scope; append nothing."""
        seen["rankgauss"].append((list(X_in.columns), list(num_cols or [])))
        return X_in, [], [], None

    real_count = cfi.count_encode_with_recipes

    def _spy_count(X_in, cat_cols=None, **kwargs):
        """Record the count-encoding scope, then encode for real so frequency encoding sees count outputs."""
        seen["count"].append((list(X_in.columns), list(cat_cols or [])))
        try:
            X_out, appended, recipes = real_count(X_in, cat_cols=cat_cols, **kwargs)
        except Exception as exc:
            seen["count_err"] = f"{type(exc).__name__}: {exc}"
            raise
        seen["count_out"].extend(appended)
        return X_out, appended, recipes

    def _spy_freq(X_in, cat_cols=None, **kwargs):
        """Record the frequency-encoding scope; append nothing."""
        seen["freq"].append((list(X_in.columns), list(cat_cols or [])))
        return X_in, [], []

    monkeypatch.setattr(mig, "greedy_mi_fe_construct_with_recipes", _fake_mi_greedy)
    monkeypatch.setattr(wv, "hybrid_wavelet_fe_with_recipes", _spy_wavelet)
    monkeypatch.setattr(extra, "hybrid_rankgauss_fe", _spy_rankgauss)
    monkeypatch.setattr(cfi, "count_encode_with_recipes", _spy_count)
    monkeypatch.setattr(cfi, "frequency_encode_with_recipes", _spy_freq)
    monkeypatch.setattr(te, "auto_detect_te_cols", lambda X_in, **kwargs: list(X_in.columns))

    m = MRMR(
        verbose=0,
        random_seed=0,
        max_runtime_mins=2,
        fe_check_pairs_subsample_n=0,
        fe_kfold_te_enable=False,
        fe_mi_greedy_enable=True,
        fe_wavelet_enable=True,
        fe_rankgauss_enable=True,
        fe_count_encoding_enable=True,
        fe_frequency_encoding_enable=True,
        # The local MI gate (default on) drops every count column on this small fixture, and frequency encoding can only leak count
        # outputs that exist. The gate is not what this test observes, so it is off.
        fe_local_mi_gate=False,
        **params,
    )
    # The stubbed MI-greedy column has no recipe, so later end-of-fit stages may reject it; that is outside what this test observes. The
    # scoping decisions under test are all recorded by the spies before any such stage runs, and each assertion below first proves its spy ran.
    try:
        m.fit(X, y)
    except Exception:  # nosec B110 - see the comment above
        pass
    return seen, list(X.columns)


def test_wavelet_and_rankgauss_never_scope_an_mi_greedy_column(monkeypatch):
    """FEC-1: the auto scope handed to wavelet and rankgauss is raw columns only, even with an MI-greedy column in the frame."""
    seen, raw = _fit_with_spies(monkeypatch)
    for family in ("wavelet", "rankgauss"):
        assert seen[family], f"the {family} stage never ran; nothing was observed"
        frame_cols, scope = seen[family][0]
        assert _MIG_COL in frame_cols, f"precondition: the MI-greedy column must be in the frame the {family} stage sees"
        assert _MIG_COL not in scope, f"{family} was handed the MI-greedy column {_MIG_COL!r} as a source"
        assert set(scope) <= set(raw), f"{family} was handed non-raw sources: {sorted(set(scope) - set(raw))}"


def test_count_encoding_auto_scope_excludes_engineered_columns(monkeypatch):
    """FEC-2: with no explicit count columns, the auto-detected scope must still drop engineered columns."""
    seen, _ = _fit_with_spies(monkeypatch)
    assert seen["count"], "the count-encoding stage never ran; nothing was observed"
    frame_cols, scope = seen["count"][0]
    assert _MIG_COL in frame_cols, "precondition: the MI-greedy column must be in the frame count encoding sees"
    assert _MIG_COL not in scope, f"count encoding auto-scoped the engineered column {_MIG_COL!r}"


def test_frequency_encoding_auto_scope_excludes_engineered_and_count_outputs(monkeypatch):
    """FEC-2 + IMPL-6: frequency encoding must drop engineered columns AND the count outputs appended just before it."""
    seen, _ = _fit_with_spies(monkeypatch)
    assert seen["freq"], "the frequency-encoding stage never ran; nothing was observed"
    assert seen["count_out"], f"precondition: count encoding must have appended columns for frequency encoding to see (encoder error: {seen.get('count_err')})"
    frame_cols, scope = seen["freq"][0]
    assert set(seen["count_out"]) <= set(frame_cols), "precondition: the count outputs must be in the frame frequency encoding sees"
    assert _MIG_COL not in scope, f"frequency encoding auto-scoped the engineered column {_MIG_COL!r}"
    leaked = sorted(set(scope) & set(seen["count_out"]))
    assert not leaked, f"frequency encoding auto-scoped count-encoding outputs: {leaked}"
