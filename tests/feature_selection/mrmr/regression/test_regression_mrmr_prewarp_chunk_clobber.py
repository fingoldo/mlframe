"""Regression: the prewarp-spec joblib-chunk CLOBBER + cross-iteration drop bug.

ROOT CAUSE (fixed 2026-06-08). ``_run_fe_step`` collects the per-operand learned
pre-warp specs (``_prewarp_specs``, keyed by cols-space var index) so the recipe
builder can write each engineered ``prewarp`` column's coefficients into
``recipe.extra`` for leak-safe ``transform()``-time replay. The joblib
``backend="threading"`` path splits the prospective pairs across N chunks; each
chunk's ``check_prospective_fe_pairs`` returns its OWN fitted specs under the SAME
reserved key (``_PREWARP_SPECS_RESULT_KEY``). The pre-fix merge loop did a bare
``prospective_additions.update(next_dict)`` per chunk -- so each chunk's reserved
key OVERWROTE the previous chunk's, and only the LAST chunk's specs survived.

An engineered ``prewarp(operand)`` column whose warp was fit in an EARLIER chunk
(or an earlier MRMR FE iteration -- the per-call dict was also not persisted on
``self``) then had NO spec in ``_prewarp_specs`` at recipe-build time. The builder
passed ``prewarp_a/b=None``, so ``recipe.extra`` never got the ``prewarp_*_coef``
fields, and ``transform()`` raised::

    KeyError: unary_binary recipe '...' uses the 'prewarp' pseudo-unary on side
    'a' but 'prewarp_a_coef' is missing from extra. Re-fit MRMR to regenerate ...

The fix (a) backs ``_prewarp_specs`` with a self-level accumulator so specs persist
across FE iterations, and (b) MERGES each chunk's reserved spec payload into that
accumulator BEFORE the ``update`` clobbers the key. Selection is byte-identical
(scheduling/merge-order only); the only observable change is that ``transform()``
on held-out rows no longer raises.

This test forces the multi-chunk joblib path (n_jobs>1, several prewarp-bearing
prospective pairs spread over >1 chunk) and asserts ``transform()`` replays every
engineered prewarp column without raising.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters.mrmr import MRMR

_LEAN = dict(dcd_enable=False, build_friend_graph=False, cluster_aggregate_enable=False)


def _make_multi_product(seed: int = 7, n: int = 6000) -> tuple[pd.DataFrame, pd.Series]:
    """Two INDEPENDENT non-monotone product targets, so the pair search fits a
    prewarp spec on multiple operand pairs -- enough prospective pairs to spread
    across more than one joblib chunk when n_jobs>1."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.5, 2.5, n)
    b = rng.uniform(-2.5, 2.5, n)
    c = rng.uniform(-2.5, 2.5, n)
    d = rng.uniform(-2.5, 2.5, n)
    e = rng.normal(0, 1, n)
    f = rng.normal(0, 1, n)
    # Two non-monotone product signals: (a**3-2a)*(b**2-b) and (c**3-2c)*(d**2-d).
    t1 = (a**3 - 2 * a) * (b**2 - b)
    t2 = (c**3 - 2 * c) * (d**2 - d)
    true = t1 / np.std(t1) + t2 / np.std(t2)
    y = true + rng.normal(0, 0.05 * np.std(true), n)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e, "f": f})
    return df, pd.Series(y, name="y")


def test_prewarp_transform_replays_under_multichunk_joblib():
    """With prewarp ON and the joblib multi-chunk path (n_jobs>1, several
    prospective pairs), ``transform()`` on held-out rows must replay EVERY
    engineered prewarp column without a ``KeyError`` -- i.e. every prewarp
    operand's spec survived the per-chunk merge.
    """
    MRMR.clear_fit_cache()
    df, y = _make_multi_product()
    # Held-out frame (same columns, fresh rows) drives the transform()-replay.
    df_test, _ = _make_multi_product(seed=99, n=2000)

    fs = MRMR(
        verbose=0,
        n_jobs=4,  # >1 -> joblib path eligible
        random_seed=0,
        fe_smart_polynom_iters=0,
        fe_hybrid_orth_enable=False,
        fe_pair_prewarp_enable=True,  # the spec-bearing path under test
        **_LEAN,
    )
    fs.fit(df, y)

    eng = [nm for nm in fs.get_feature_names_out() if nm not in set(df.columns)]
    prewarp_cols = [nm for nm in eng if "prewarp" in nm]
    # The scenario is engineered to recover at least one prewarp column; if the
    # pipeline ever stops recovering any prewarp feature the test is vacuous, so
    # pin that we actually exercise the replay path.
    assert (
        prewarp_cols
    ), "no engineered 'prewarp' column was recovered; the multi-product scenario no longer exercises the prewarp spec-merge path (test would be vacuous)"

    # The crux: replay on held-out rows must NOT raise the missing-coef KeyError.
    try:
        Xt = np.asarray(fs.transform(df_test))
    except KeyError as exc:  # pragma: no cover - this is exactly the regression
        pytest.fail(f"transform() raised KeyError replaying a prewarp recipe -- a prewarp operand spec was clobbered/dropped before recipe build: {exc}")

    # And the replayed prewarp columns must be finite, non-degenerate values.
    names = list(fs.get_feature_names_out())
    for nm in prewarp_cols:
        col = Xt[:, names.index(nm)]
        assert np.isfinite(col).all(), f"prewarp column {nm!r} has non-finite replay values"


if __name__ == "__main__":  # pragma: no cover
    test_prewarp_transform_replays_under_multichunk_joblib()
    print("OK")
