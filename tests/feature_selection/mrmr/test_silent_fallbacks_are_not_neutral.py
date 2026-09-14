"""Failure fallbacks that substituted a non-neutral value silently (mrmr_audit_2026-09-14 NUM-15, NUM-16, NUM-17, NUM-19, NUM-21).

Each site caught ``Exception``, logged at debug, and substituted a value that is not neutral for its consumer:

* NUM-15: a present-but-unusable ``xxhash`` read as "absent" and pinned the process onto the slower duplicate-column hash for its lifetime;
* NUM-16: a failing Struct-column check substituted "no Struct columns", disarming the validator whose only job is to fail early;
* NUM-17: a failing per-column MI call scored the column 0.0, MRMR's "never select" value, deleting a possibly strong feature;
* NUM-19: one failed chunk of the prevalence auto-debias switched debiasing off for every pair of the fit, loosening the gate;
* NUM-21: a failed capture of the pre-FE raw safety net emptied it, disabling the re-add that stops genuine raws being dropped.
"""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------------------------------------------- NUM-15


def test_xxhash_present_but_unusable_is_warned(monkeypatch, caplog):
    """An installed xxhash missing the expected entry point must warn, not read as 'absent' at debug."""
    from mlframe.feature_selection.filters import _mrmr_degenerate as dg

    monkeypatch.setitem(sys.modules, "xxhash", types.ModuleType("xxhash"))  # importable, but lacks xxh3_64_intdigest
    with caplog.at_level(logging.DEBUG, logger=dg.logger.name):
        assert dg._resolve_xxh3_64() is None
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING and "xxhash" in r.getMessage()]
    assert warned, "a present-but-unusable xxhash downgraded duplicate-column detection silently"
    assert "AttributeError" in warned[0].getMessage()


def test_xxhash_genuinely_absent_stays_quiet(monkeypatch, caplog):
    """Control: a genuine ImportError is the expected optional-dependency case and must not warn."""
    from mlframe.feature_selection.filters import _mrmr_degenerate as dg

    monkeypatch.setitem(sys.modules, "xxhash", None)  # makes ``import xxhash`` raise ImportError
    with caplog.at_level(logging.DEBUG, logger=dg.logger.name):
        assert dg._resolve_xxh3_64() is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_xxhash_healthy_resolves_the_entry_point(monkeypatch):
    """Control: a usable module returns its hash function."""
    from mlframe.feature_selection.filters import _mrmr_degenerate as dg

    fake = types.ModuleType("xxhash")
    fake.xxh3_64_intdigest = lambda buf: 7  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xxhash", fake)
    assert dg._resolve_xxh3_64() is fake.xxh3_64_intdigest


# ---------------------------------------------------------------------------------------------------------------- NUM-16


def test_struct_column_detection_failure_does_not_disarm_the_guard(monkeypatch, caplog):
    """If the Struct-column check itself fails, fit must stop with that error rather than assume there are no Struct columns."""
    pl = pytest.importorskip("polars")
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = pl.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = (rng.normal(size=200) > 0).astype(int)

    def _boom(self):
        """Stand in for a dtype lookup that fails under a polars version change."""
        raise RuntimeError("synthetic dtype lookup failure")

    monkeypatch.setattr(pl.DataFrame, "dtypes", property(_boom))
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="synthetic dtype lookup failure"):
        MRMR(verbose=0, max_runtime_mins=1).fit(X, y)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING and "Struct" in r.getMessage()], "the failed Struct check was not reported"


# ---------------------------------------------------------------------------------------------------------------- NUM-17


def test_sklearn_mi_column_failure_is_raised_not_scored_zero(monkeypatch, caplog):
    """A per-column MI failure is a bug, not a regime: it must surface, not score the column at MRMR's never-select value."""
    import sklearn.metrics

    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_mi_backends import _mi_classif_batch_sklearn

    def _boom(*args, **kwargs):
        """Fail the way a dtype surprise inside mutual_info_score would."""
        raise ValueError("synthetic mutual_info_score failure")

    monkeypatch.setattr(sklearn.metrics, "mutual_info_score", _boom)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] > 0).astype(np.int64)
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="synthetic mutual_info_score failure"):
        _mi_classif_batch_sklearn(X, y, nbins=8)
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING and "column 0" in r.getMessage()]
    assert warned, "the failing column was not named in a warning"


def test_numba_partial_nan_column_failure_is_raised_not_scored_zero(monkeypatch, caplog):
    """The partial-NaN branch of the numba backend swallowed the same dispatcher whose dense-branch failure is already a warning."""
    import mlframe.feature_selection.filters.hermite_fe as hermite_fe
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_mi_backends import _mi_classif_batch_numba

    def _boom(*args, **kwargs):
        """Fail the per-column dispatcher call."""
        raise RuntimeError("synthetic dispatcher failure")

    monkeypatch.setattr(hermite_fe, "plugin_mi_classif_batch_dispatch", _boom)
    rng = np.random.default_rng(0)
    col = rng.normal(size=300)
    col[::7] = np.nan  # partially finite -> the per-column branch, not the dense batch
    X = col.reshape(-1, 1)
    y = (rng.normal(size=300) > 0).astype(np.int64)
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="synthetic dispatcher failure"):
        _mi_classif_batch_numba(X, y, nbins=8)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING and "column 0" in r.getMessage()], "the failing column was not named in a warning"


# ---------------------------------------------------------------------------------------------------------------- NUM-19


def test_prevalence_debias_chunk_failure_is_not_latched_for_the_whole_fit(monkeypatch, caplog):
    """One failed chunk must leave debiasing on for every other chunk's pairs, and say so."""
    import mlframe.feature_selection.filters._mrmr_fe_step._step_pairmi as step_pairmi_mod
    import mlframe.feature_selection.filters._permutation_null as pn

    n, k = 400, 6
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4, size=(n, k + 1)).astype(np.int32)
    nbins = np.array([4] * (k + 1), dtype=np.int32)
    classes_y = data[:, k].astype(np.int32)
    freqs_y = np.bincount(classes_y).astype(np.float64)

    class _Fake:
        """The estimator attributes the pair-MI step reads; the maxT floor is off so the auto-debias fill runs."""

        fe_max_engineered_operands = -1
        fe_escalation_feedforward_enable = True
        _fe_synergy_exhaustive_active_ = False
        fe_pair_maxt_null_permutations = 0
        fe_auto_prevalence_debias_chunk_size = 3
        feature_names_in_ = [f"f{i}" for i in range(k)]

    real_bias = pn.pairwise_mm_joint_bias
    calls = {"n": 0}

    def _fail_first_chunk(*args, **kwargs):
        """Fail the first chunk only; score the rest for real."""
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic chunk failure")
        return real_bias(*args, **kwargs)

    monkeypatch.setattr(pn, "pairwise_mm_joint_bias", _fail_first_chunk)
    with caplog.at_level(logging.WARNING):
        result = step_pairmi_mod.compute_pair_mis_and_floor(
            _Fake(),
            data=data,
            cols=[f"f{i}" for i in range(k)] + ["y"],
            nbins=nbins,
            X=None,
            classes_y=classes_y,
            classes_y_safe=classes_y,
            freqs_y=freqs_y,
            target_indices=(k,),
            cached_MIs={},
            cached_confident_MIs={},
            numeric_vars_to_consider=set(range(k)),
            _prevalence_debias_auto=True,
            n_jobs=1,
            prefetch_factor=2,
            parallel_kwargs={"backend": "threading"},
            fe_min_nonzero_confidence=0.99,
            fe_npermutations=0,
            fe_min_pair_mi=0.001,
            fe_min_pair_mi_prevalence=1.05,
            verbose=0,
        )
    _, _, _, pair_mm_bias, prevalence_debias_auto = result
    n_pairs = k * (k - 1) // 2
    assert calls["n"] >= 2, "the auto-debias fill did not reach a second chunk; the setup observes nothing"
    assert prevalence_debias_auto is True, "one failed chunk switched prevalence debiasing off for the whole fit"
    assert len(pair_mm_bias) == n_pairs - 3, f"expected every pair outside the failed 3-pair chunk to be debiased, got {len(pair_mm_bias)} of {n_pairs}"
    assert [r for r in caplog.records if r.levelno >= logging.WARNING and "debias" in r.getMessage()], "the failed chunk was not reported"


# ---------------------------------------------------------------------------------------------------------------- NUM-21


def test_prefe_safety_net_capture_failure_is_warned(caplog):
    """A failed capture empties the net (nothing to re-add), and that consequence must be visible at warning."""
    from mlframe.feature_selection.filters._mrmr_fe_step._step_core import _capture_prefe_screened_raw

    class _NoNames:
        """An estimator whose feature_names_in_ is not set yet, so the capture fails."""

    est = _NoNames()
    with caplog.at_level(logging.WARNING):
        _capture_prefe_screened_raw(est, ["a", "b"], [0, 1])
    assert est._prefe_screened_raw_ == []
    assert [r for r in caplog.records if r.levelno >= logging.WARNING and "safety net" in r.getMessage()], "the failed capture was logged below warning"


def test_prefe_safety_net_capture_keeps_only_raw_selected_columns():
    """Control: a healthy capture records the selected columns that are raw inputs, in selection order."""
    from mlframe.feature_selection.filters._mrmr_fe_step._step_core import _capture_prefe_screened_raw

    est = types.SimpleNamespace(feature_names_in_=["a", "b", "c"])
    _capture_prefe_screened_raw(est, ["a", "eng_ab", "c", "b"], [2, 1, 0])
    assert est._prefe_screened_raw_ == ["c", "a"]
