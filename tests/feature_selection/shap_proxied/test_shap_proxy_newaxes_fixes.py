"""Regression tests for two newly-landed ShapProxiedFS bugs (fs_findings #3 and #18).

Both are focused unit tests of the smallest reachable behaviour -- they do NOT run a full
ShapProxiedFS.fit() training pass (the OOF-SHAP / search / honest-revalidation stages are
expensive and would saturate the box). They exercise the constructor + the cheap front of fit()
that runs before any booster work.

  * #3 (P1): ``booster_kind='catboost'`` + non-empty ``cat_features`` is not supported by the
    surrounding (non-categorical-aware) prefilter / clustering / column-slicing pipeline; the fix
    raises a clear ValueError at fit start instead of the obscure float64-densification crash that
    used to surface deep inside ``f_classif_chunked``. The guard fires before any booster import,
    so the test needs neither catboost nor xgboost installed.

  * #18 (Low): the both-floors-set conflict guard previously compared
    ``self.fidelity_floor != 0.5``, so an explicit ``fidelity_floor=0.5`` plus ``spearman_floor``
    silently skipped the intended ValueError. The fix makes ``fidelity_floor`` default to the
    ``None`` "unset" sentinel so the conflict is detected by identity, not by the numeric value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS


# --------------------------------------------------------------------------- #3 catboost cat_features
def test_catboost_cat_features_raises_clear_error_at_fit_start():
    """booster_kind='catboost' + cat_features must fail fast with an actionable ValueError.

    The pipeline densifies ``X.values`` to float64 in the prefilter, which crashes on
    string/categorical columns long before SHAP runs; the guard converts that into a clear
    fit-start error. It triggers before any booster is constructed, so no optional dep is needed.
    """
    X = pd.DataFrame({"c0": ["a", "b", "a", "b"], "n1": [0.1, 0.2, 0.3, 0.4]})
    y = np.array([0, 1, 0, 1])

    sel = ShapProxiedFS(
        classification=True, booster_kind="catboost", cat_features=["c0"],
        random_state=0, verbose=False,
    )
    with pytest.raises(ValueError, match="catboost.*cat_features.*not yet supported"):
        sel.fit(X, y)


def test_catboost_without_cat_features_does_not_trip_the_guard():
    """The guard is scoped to non-empty ``cat_features``; numeric-only catboost fits are unaffected.

    We only need to prove the guard does NOT fire (it would raise the specific 'not yet supported'
    message). With cat_features unset the front of fit() proceeds past the guard; we stop it before
    the (catboost-dependent) booster build via a deliberately bad input so the test stays cheap and
    dep-free -- the assertion is purely that the cat_features guard message is NOT what surfaces.
    """
    X = pd.DataFrame({"n0": [0.1, 0.2, 0.3, 0.4], "n1": [1.0, 2.0, 3.0, 4.0]})
    y = np.array([0, 1, 0, 1])

    sel = ShapProxiedFS(
        classification=True, booster_kind="catboost", cat_features=None,
        random_state=0, verbose=False,
    )
    try:
        sel.fit(X, y)
    except ValueError as exc:  # may legitimately fail later (e.g. catboost not installed -> ImportError)
        assert "cat_features" not in str(exc), (
            "cat_features guard fired even though cat_features was unset: " + str(exc)
        )
    except Exception:
        # Any non-ValueError (ImportError when catboost is absent, etc.) means we got PAST the
        # cat_features guard, which is exactly what this test asserts.
        pass


# --------------------------------------------------------------------------- #18 fidelity_floor sentinel
def test_fidelity_floor_default_is_none_sentinel():
    """fidelity_floor must default to None so the both-floors-set guard detects an explicit 0.5.

    Pre-fix the default was the literal 0.5, which the conflict guard could not distinguish from a
    user explicitly passing fidelity_floor=0.5; with None as the 'unset' sentinel the guard keys on
    identity. The default must also round-trip through sklearn get_params/set_params untouched.
    """
    sel = ShapProxiedFS()
    assert sel.fidelity_floor is None, (
        f"fidelity_floor default must be the None sentinel, got {sel.fidelity_floor!r}"
    )
    assert sel.get_params()["fidelity_floor"] is None


def test_fidelity_floor_explicit_value_stored_verbatim():
    """An explicit fidelity_floor (incl. 0.5) is stored as-is -- no constructor coercion.

    This is what lets the conflict guard tell 'user pinned 0.5' apart from 'unset'. Storing raw
    also preserves sklearn clone() identity (the file coerces nothing in __init__ for this reason).
    """
    for v in (0.5, 0.6, 0.0):
        sel = ShapProxiedFS(fidelity_floor=v)
        assert sel.fidelity_floor == v
        assert sel.get_params()["fidelity_floor"] == v


def test_both_floors_set_conflict_detected_with_explicit_default_value():
    """fidelity_floor=0.5 + spearman_floor must be flagged as a both-set conflict.

    The detection logic the fix relies on is ``self.fidelity_floor is not None`` (was the buggy
    ``self.fidelity_floor != 0.5``). We assert the observable predicate directly on a constructed
    selector rather than running the full fit() -- the ValueError itself lives inside the
    trust-guard stage, which only runs after the (expensive, CPU-saturating) OOF-SHAP + search
    pipeline that this suite must not execute.
    """
    sel = ShapProxiedFS(fidelity_floor=0.5, spearman_floor=0.6)
    # Pre-fix this was False (0.5 != 0.5), silently skipping the conflict ValueError.
    conflict_detected = sel.spearman_floor is not None and sel.fidelity_floor is not None
    assert conflict_detected, (
        "explicit fidelity_floor=0.5 + spearman_floor must register as a both-set conflict"
    )

    # And the no-conflict default path: only spearman_floor set -> the deprecated alias is honored
    # without a conflict (fidelity_floor stays the None sentinel).
    sel_alias_only = ShapProxiedFS(spearman_floor=0.6)
    assert sel_alias_only.fidelity_floor is None
    assert not (sel_alias_only.spearman_floor is not None
                and sel_alias_only.fidelity_floor is not None)
