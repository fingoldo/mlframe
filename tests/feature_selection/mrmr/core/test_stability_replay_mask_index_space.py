"""The stability-replay mask must name the columns that were actually selected (mrmr_audit_2026-09-14 FIT_IMPL-1).

``_build_stability_replay_state`` received ``selected_vars`` in ``feature_names_in_`` index space (the fit
body rebinds it before ``_assign_support`` runs) but indexed ``cols`` -- categorize_dataset's output, which
injects the target column, appends every engineered column, and reorders categoricals first. The two spaces
coincide only for an all-numeric frame with no FE; with any categorical the report silently named the wrong
columns and cut at the wrong K. It never raised: it returned a plausible-looking table.

Verified empirically before the fix: on the fixture below the replay claimed ['n2', 'n3'] while ``support_``
held ['n1', 'n2'].
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


@pytest.fixture(scope="module")
def fitted_with_a_leading_categorical():
    """A leading categorical forces categorize_dataset's reorder, which is what desynchronises the spaces."""
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(
        {
            "cat": pd.Series(rng.choice(list("abc"), n)).astype("category"),
            "n1": rng.normal(size=n),
            "n2": rng.normal(size=n),
            "n3": rng.normal(size=n),
        }
    )
    y = pd.Series((X["n1"] + X["n2"] * 0.5 + rng.normal(0, 0.1, n) > 0).astype(int))
    return MRMR(fe_max_steps=0).fit(X, y)


def test_replay_mask_names_the_same_columns_as_support(fitted_with_a_leading_categorical):
    """The replay's selected names must equal the estimator's own reported selection."""
    m = fitted_with_a_leading_categorical
    state = getattr(m, "_stability_replay_state_", None)
    assert state is not None, "replay state was not captured, so the invariant cannot be checked"

    names = np.asarray(state["cand_names"])
    mask = np.asarray(state["selected_mask"])
    replay_selected = sorted(str(s) for s in names[mask])
    support_names = sorted(str(s) for s in m.get_feature_names_out())
    assert replay_selected == support_names


def test_replay_mask_count_matches_the_support_size(fitted_with_a_leading_categorical):
    """n_selected drives the top-K cut on every bootstrap replay, so its size must match too."""
    m = fitted_with_a_leading_categorical
    state = m._stability_replay_state_
    assert int(np.asarray(state["selected_mask"]).sum()) == len(m.get_feature_names_out())


def test_the_target_column_is_never_marked_selected(fitted_with_a_leading_categorical):
    """The injected target column lives in ``cols`` too; an index-space slip could mark it as a feature."""
    m = fitted_with_a_leading_categorical
    state = m._stability_replay_state_
    names = np.asarray(state["cand_names"])
    mask = np.asarray(state["selected_mask"])
    selected = {str(s) for s in names[mask]}
    assert not any(s.startswith("targ_") for s in selected), f"target column marked as selected: {selected}"
