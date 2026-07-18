"""Regression guard for the cluster_aggregate_mode pickle-default mismatch
(audit 2026-06-03: cluster-aggregate-1).

The MRMR constructor defaults ``cluster_aggregate_mode='replace'`` (the
deliberate Wave-8 fix for the duplicate-vote effect), but ``__setstate__``'s
legacy-default table injected ``'augment'`` for attribute-less (pre-Wave-8)
pickles -- so an old pipeline silently unpickled to the SUPERSEDED behaviour
instead of matching a freshly-constructed estimator. The legacy default is now
'replace' to match the constructor.
"""

from __future__ import annotations

from mlframe.feature_selection.filters.mrmr import MRMR


def test_constructor_default_mode_is_replace():
    """Constructor default mode is replace."""
    assert MRMR().cluster_aggregate_mode == "replace"


def test_legacy_pickle_refits_mode_to_replace():
    # A pre-Wave-8 pickle had no cluster_aggregate_mode attribute at all.
    """Legacy pickle refits mode to replace."""
    m = MRMR.__new__(MRMR)
    m.__setstate__({})  # empty legacy state -> all defaults injected
    assert m.cluster_aggregate_mode == "replace", (
        f"attribute-less legacy pickle must refit to the corrected 'replace' mode, not 'augment'; got {m.cluster_aggregate_mode!r}"
    )


def test_legacy_pickle_preserves_explicit_mode():
    # If the pickle DID carry an explicit mode, __setstate__ must not clobber it.
    """Legacy pickle preserves explicit mode."""
    m = MRMR.__new__(MRMR)
    m.__setstate__({"cluster_aggregate_mode": "augment"})
    assert m.cluster_aggregate_mode == "augment"
