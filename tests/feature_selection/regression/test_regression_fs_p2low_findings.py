"""Wave-5a regression sensors for FS P2/Low findings.

Pinned per-finding so a future refactor that drops the fold-in / strengthen-sig / etc.
re-surfaces immediately.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from mlframe.training.pipeline._pipeline_cache import (
    _pipeline_signature_for_cache,
)
from mlframe.feature_selection.filters._mrmr_fingerprints import (
    _content_array_signature,
    _mrmr_compute_y_fingerprint_sample,
)


class _MarkerStep:
    """Stub sklearn-like step with the setattr marker."""

    def get_params(self, deep: bool = False):
        """Return a fixed trivial params dict, enough to satisfy sklearn-step introspection."""
        return {"k": 1}


class _MarkerPipeline:
    """Stub pipeline with a single marked step."""

    def __init__(self, marked: bool):
        step = _MarkerStep()
        step._mlframe_use_sample_weights_in_fs_ = marked
        self.steps = [("sel", step)]


def test_w5_fs_f7_pipeline_signature_folds_sample_weight_marker():
    """Cache must distinguish weight-aware vs weight-blind pipelines."""
    sig_blind = _pipeline_signature_for_cache(_MarkerPipeline(marked=False))
    sig_aware = _pipeline_signature_for_cache(_MarkerPipeline(marked=True))
    assert sig_blind != sig_aware, f"_mlframe_use_sample_weights_in_fs_ marker must affect the signature; got identical signatures: {sig_blind!r}"
    assert "sw=True" in sig_aware
    assert "sw=False" in sig_blind


def test_w5_fs_f8_content_array_signature_1024_strided_sampling():
    """1024-strided sampling discriminates two arrays whose 10-cell signature collides."""
    # n flat = 1024 * 5; old 10-cell stride positions were 0, ~568, ~1137, ..., n-1.
    # New 1024-cell stride hits every 5th flat cell, so any single-cell mutation in the new
    # stride is caught while the old stride missed cells between boundary positions.
    n = 1024
    a = np.zeros((n, 5), dtype=np.int8)
    b = np.zeros((n, 5), dtype=np.int8)
    # Pick a flat-index position that is on the new 1024-stride but NOT on the old 10-stride.
    # Old positions for flat-n=5120: 0, 568, 1137, ..., 5119 -- modulo 5 these are {0, 3, 2, 1, 4, ...}.
    # New 1024-stride hits index 5 (i=1 -> int(5119/1023) = 5). Mutate cell at flat index 5 = row 1 col 0.
    b[1, 0] = 1
    sig_a = _content_array_signature(pd.DataFrame(a))
    sig_b = _content_array_signature(pd.DataFrame(b))
    assert (
        sig_a != sig_b
    ), "1024-strided sampling must catch a single-cell mid-frame difference; pre-fix 10-sample stride missed cells between boundary positions."


def test_w5_fs_f10_y_fingerprint_is_bit_exact_not_6_decimal_rounded():
    """Two y vectors differing only at the 7th decimal must hash to different fingerprints."""
    n = 2000
    y_a = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y_b = y_a.copy()
    # Inject a 7th-decimal difference at the first strided sample position. The
    # pre-fix code rounded to 6 decimals and collided.
    y_b[0] = y_a[0] + 5e-8
    fp_a = _mrmr_compute_y_fingerprint_sample(y_a)
    fp_b = _mrmr_compute_y_fingerprint_sample(y_b)
    assert fp_a != fp_b, "bit-exact y fingerprint must distinguish 1e-7-scale differences; pre-fix rounded to 6 decimals and silently merged distinct targets."


def test_w5_fs_f11_groupkfold_n_groups_floor_enforced():
    """fit() must reject n_groups < 2*cv_n when groups are provided."""
    from sklearn.linear_model import LinearRegression
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    rng = np.random.default_rng(0)
    n, p = 60, 4
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(rng.normal(size=n), name="y")
    # 3 unique groups, cv=5 (n_splits=5) -> 3 < 10, must raise
    groups = np.array([i % 3 for i in range(n)])

    sel = RFECV(estimator=LinearRegression(), cv=5)
    with pytest.raises(ValueError, match=r"n_groups=3.*<.*2 \* cv \(5\)"):
        sel.fit(X, y, groups=groups)


def test_w5_fs_f16_deepcopy_splitter_in_early_stopping_path():
    """copy.copy -> deepcopy: mutating val_cv must not corrupt the caller's splitter."""
    from mlframe.feature_selection.wrappers.rfecv._cv_setup import _resolve_cv_and_val_cv

    class _CustomCV:
        """Splitter without get_params -> hits the deepcopy fallback."""

        def __init__(self):
            self.n_splits = 7
            self._mutable_state = [1, 2, 3]

        def split(self, X, y=None, groups=None):
            """Yield no folds; only invoked to prove the splitter object itself survives the deepcopy."""
            return iter([])

    cv_orig = _CustomCV()
    cv_id = id(cv_orig._mutable_state)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 3)))
    y = pd.Series(rng.normal(size=20))
    _cv_out, val_cv, _ = _resolve_cv_and_val_cv(
        cv=cv_orig,
        cv_shuffle=False,
        random_state=0,
        estimator=None,
        X=X,
        y=y,
        groups=None,
        fit_params={},
        verbose=0,
        early_stopping_val_nsplits=3,
        early_stopping_rounds=None,
        _polars_time_series_hint=False,
    )
    # val_cv must be a distinct object with a distinct _mutable_state
    assert val_cv is not cv_orig
    assert id(val_cv._mutable_state) != cv_id, "copy.deepcopy must isolate val_cv from the caller's cv; copy.copy shared the inner list"
    # cv_orig must be untouched after the val_cv mutation
    assert cv_orig.n_splits == 7
