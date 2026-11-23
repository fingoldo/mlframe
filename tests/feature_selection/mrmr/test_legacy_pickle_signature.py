"""A pickle written before ``signature`` existed must restore with it defaulted, and the next fit must not raise (AUD23-03)."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _legacy_state(m: MRMR) -> dict:
    """The pickled state of ``m`` with the attribute a legacy pickle lacks removed."""
    state = dict(m.__getstate__())
    state.pop("signature", None)
    return state


def test_setstate_defaults_signature_for_a_legacy_pickle():
    """``signature`` is not a constructor parameter, so the ctor-default injection never covers it: it must be defaulted by name."""
    legacy = MRMR.__new__(MRMR)
    legacy.__setstate__(_legacy_state(MRMR(verbose=0)))
    assert hasattr(legacy, "signature")
    assert legacy.signature is None


def test_a_legacy_pickle_fits_again():
    """The restored estimator fits (the single-target skip check reads ``self.signature``)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((400, 5)), columns=list("abcde"))
    y = (X["a"] + 0.3 * rng.standard_normal(400) > 0).astype(int)
    fitted = MRMR(verbose=0, random_seed=0, fe_max_steps=0).fit(X, y)
    legacy = MRMR.__new__(MRMR)
    legacy.__setstate__(_legacy_state(fitted))
    legacy.skip_retraining_on_same_content = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = legacy.fit(X, y)
    assert list(again.get_feature_names_out())


def test_a_current_pickle_round_trips():
    """Contrast: an up-to-date pickle keeps its signature through pickle."""
    m = MRMR(verbose=0)
    m.signature = ("x",)
    assert pickle.loads(pickle.dumps(m)).signature == ("x",)  # nosec B301 - the bytes were produced one line above, not untrusted input
