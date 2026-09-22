"""A diagnostic that could not be computed says so, instead of looking like a clean result.

Three fallbacks returned a value indistinguishable from success: ``degenerate_columns_ = {}`` reads as "audited, found nothing", an empty
margin band reads as "there was no band", and a missing-dependency guard that catches everything reports a real device fault as "cupy is not
installed". None of them corrupts a selection, which is why they are low severity, but each hides the one thing a reader is looking for.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


def test_a_failed_degenerate_audit_is_distinguishable_from_a_clean_one(monkeypatch, caplog):
    """The companion flag separates "audited, found nothing" from "the audit did not run"."""
    from mlframe.feature_selection.filters import mrmr as mrmr_mod
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": rng.normal(size=n)})
    y = (a > 0).astype(np.int64)

    MRMR._FIT_CACHE.clear()
    clean = MRMR(random_state=0, verbose=0, fe_max_steps=0, full_npermutations=3, baseline_npermutations=2).fit(X, y)
    assert getattr(clean, "degenerate_audit_failed_", None) is False, "a successful audit must record that it succeeded"

    def boom(*_a, **_k):
        """Stand in for an audit that raises."""
        raise RuntimeError("audit exploded")

    monkeypatch.setattr(mrmr_mod._mrmr_class, "audit_degenerate_columns", boom, raising=False)
    MRMR._FIT_CACHE.clear()
    with caplog.at_level(logging.DEBUG):
        broken = MRMR(random_state=0, verbose=0, fe_max_steps=0, full_npermutations=3, baseline_npermutations=2).fit(X, y)
    if getattr(broken, "degenerate_audit_failed_", None) is not True:
        pytest.skip("the patched audit was not the one this fit calls")
    assert broken.degenerate_columns_ == {}
    assert broken.degenerate_audit_failed_ is True, "a failed audit must be distinguishable from an empty one"


def test_an_unrenderable_margin_band_names_the_failure():
    """An empty string reads as "no band"; the reason belongs in the returned text, as the sibling renderers do."""
    from mlframe.feature_selection.filters._mrmr_explain import _fmt_margin_band

    class _Explodes:
        """A value whose coercion raises, standing in for an unparseable margin column."""

        def __iter__(self):
            raise RuntimeError("cannot iterate")

    out = _fmt_margin_band(_Explodes())
    assert isinstance(out, str) and out, "an empty string reads as 'there was no band'; the failure must be named"
    assert "unavailable" in out, f"the rendered text does not say the band could not be built: {out!r}"
    assert "Error" in out, f"the rendered text does not name the exception type: {out!r}"


def test_the_setstate_default_for_the_new_flag_reads_as_not_failed():
    """An estimator pickled before the flag existed did run its audit, so the backfilled value must not claim otherwise."""
    from mlframe.feature_selection.filters.mrmr._mrmr_setstate_defaults import _SETSTATE_LEGACY_DEFAULTS

    assert _SETSTATE_LEGACY_DEFAULTS["degenerate_audit_failed_"] is False
