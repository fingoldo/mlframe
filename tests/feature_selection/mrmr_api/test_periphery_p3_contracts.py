"""Periphery P3 contracts.

* ``selection_stability_report`` rejects ``n_boot < 1`` instead of silently running one resample.
* A string-labelled target fingerprints stably; it used to fail the float cast, warn and disable the identity cache on every fit.
* Recipe replay state whose codes disagree with y's length is reported, not silently dropped from the table.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _mrmr_stability_report as sr
from mlframe.feature_selection.filters._mrmr_fingerprints import _mrmr_compute_y_fingerprint_sample


class _Stub:
    """The minimal fitted surface the stability report reads."""

    def __init__(self, state, recipes=()):
        """Hold the replay state and recipes."""
        self._stability_replay_state_ = state
        self._engineered_recipes_ = list(recipes)

    def _effective_random_seed(self):
        """Fixed seed."""
        return 0


def _state(n=200):
    """Replay state with one selected candidate."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=n)
    return {"cand_codes": np.column_stack([y, rng.integers(0, 3, size=n)]), "y_codes": y, "cand_names": ["s", "n"], "selected_mask": np.array([True, False])}


@pytest.mark.parametrize("bad", [0, -3])
def test_selection_stability_report_rejects_nonpositive_n_boot(bad):
    """A non-positive resample count is an input error, not a request for one resample."""
    with pytest.raises(ValueError, match="n_boot"):
        sr.selection_stability_report(_Stub(_state()), n_boot=bad, random_state=0, as_text=False)


def test_y_fingerprint_stable_for_string_labels(caplog):
    """The same string labels hash identically across calls without a warning; different labels hash apart."""
    y1 = pd.Series(["cat", "dog", "cat", "bird"] * 50)
    with caplog.at_level(logging.WARNING):
        a = _mrmr_compute_y_fingerprint_sample(y1)
        b = _mrmr_compute_y_fingerprint_sample(y1.copy())
    assert a == b, "a string target must give a stable fingerprint (it was a fresh never-matching token per call)"
    assert not any("fingerprint failed" in r.getMessage() for r in caplog.records)
    assert _mrmr_compute_y_fingerprint_sample(pd.Series(["cat", "dog", "dog", "bird"] * 50)) != a


def test_recipe_survival_warns_on_stale_replay_state(caplog):
    """A recipe whose stored codes are shorter than y is named in a warning."""
    state = _state()
    state["recipe_replay"] = {"mul(s,n)": {"eng_codes": np.zeros(150, dtype=np.int64), "a_codes": None, "b_codes": None, "alt": False}}
    recipe = type("R", (), {"name": "mul(s,n)", "kind": "unary_binary"})()
    with caplog.at_level(logging.WARNING):
        sr.selection_stability_report(_Stub(state, [recipe]), n_boot=3, random_state=0, as_text=False)
    assert any("mul(s,n)" in r.getMessage() and "150" in r.getMessage() for r in caplog.records)
