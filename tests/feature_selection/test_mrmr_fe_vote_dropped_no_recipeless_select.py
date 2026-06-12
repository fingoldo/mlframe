"""Regression: a cross-fold-stability-vote-rejected engineered feature must NOT
re-enter ``support_`` / discovered without a replayable recipe (BUG2, 2026-06-12).

ROOT CAUSE pinned here. At ``fe_max_steps=2`` the FE step's cross-fold stability
vote (``_run_fe_step``) pops a fold-unstable engineered recipe AND de-selects its
column for that step -- but the materialised bin-code column stays in
``cols``/``data``, so the downstream greedy MRMR screen (the step>1 re-screen /
final selection) re-admits it on its marginal MI. It then arrived at the fit-end
selection finaliser with NO recipe and was SILENTLY DROPPED from ``transform()``
output. ``get_feature_names_out`` advertised it (or ``_engineered_features_`` /
discovered listed it) while ``transform(df)`` never produced the column -- a
select-then-drop contract violation. The prior fix (b6627f85) froze axis params for
EXISTING recipes; it did not touch this MISSING-recipe / re-admission gap.

This test uses the USER'S EXACT reproduction (uniform [0,1] X={a,b,c,d,e} with the
hidden noise var ``f``, the CASE2 target) -- NOT a controlled proxy -- and asserts:

* the "selected without replayable recipe" WARNING does NOT fire, and
* every discovered / advertised feature SURVIVES ``transform(df)`` (no silent drop).

Pre-fix this FAILS (the warning fires and ``add(qubed(a),rint(d))`` is dropped);
post-fix the vote-rejected feature is stripped from the selection so it never
reaches support_ without a recipe, and the surviving roster is fully replayable.

``n`` is reduced to 40000 for test speed; the drop reproduces at this n (verified
against the pre-fix commit) and the user's full n=100000 case validates identically.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _user_case2_df_and_y(n: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    a = rng.rand(n)
    b = rng.rand(n)
    c = rng.rand(n)
    d = rng.rand(n)
    e = rng.rand(n)  # pure noise, IS a column of X
    f = rng.rand(n)  # hidden noise var, NOT a column of X
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    y = 0.2 * a ** 2 / b + f / 5 + np.log(c * 2) * np.sin(d / 3)
    return df, pd.Series(y, name="y")


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.mark.parametrize("n", [40000])
def test_mrmr_fe_vote_dropped_feature_not_select_then_dropped(n):
    df, y = _user_case2_df_and_y(n)

    handler = _CaptureHandler()
    logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")
    prev_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        m = MRMR(max_runtime_mins=5, verbose=1, fe_max_steps=2, random_seed=0)
        m.fit(df, y)
        names_out = list(m.get_feature_names_out())
        discovered = list(getattr(m, "_engineered_features_", []) or [])
        out = m.transform(df)
        out_cols = list(out.columns) if hasattr(out, "columns") else list(names_out)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    # (1) The "selected without replayable recipe" warning must NOT fire.
    recipeless_warnings = [
        msg for msg in handler.messages if "without replayable recipe" in msg
    ]
    assert not recipeless_warnings, (
        "A selected engineered feature was dropped for lack of a replayable recipe "
        f"(BUG2 select-then-drop): {recipeless_warnings}"
    )

    # (2) Every advertised output feature must actually be produced by transform().
    missing_from_transform = [nm for nm in names_out if nm not in out_cols]
    assert not missing_from_transform, (
        "get_feature_names_out advertises feature(s) that transform() does not "
        f"produce: {missing_from_transform} (out_cols={out_cols})"
    )

    # (3) Every discovered engineered feature must survive transform() (no silent drop).
    discovered_missing = [nm for nm in discovered if nm not in out_cols]
    assert not discovered_missing, (
        "Discovered engineered feature(s) silently dropped from transform output: "
        f"{discovered_missing} (out_cols={out_cols})"
    )

    # Sanity: the fit produced SOME usable features (guards against an all-empty
    # regression masking the contract assertions above).
    assert out.shape[1] >= 1
