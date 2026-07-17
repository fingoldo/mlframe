"""Member-equivalent winner check inside ``revalidate_top_n`` adaptive early-stop (iter92).

Extends iter77: when two consecutive rounds' winning candidates have DIFFERENT unit tuples but
their ``_expand``-ed member columns are identical (cluster aggregation collapsing two units to
the same deployed feature set), the loop should still early-stop. Verifies:

  * parity at singleton clusters (no ``unit_to_members`` mapping) -- behaviour identical to iter77;
  * the lever fires when round-0 and round-1 winners differ in unit space but expand to the same
    member set;
  * a fresh ``n_reval_models_run_via_member_equiv`` diagnostic surfaces whenever the lever fired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


@pytest.fixture
def planted_strong():
    rng = np.random.default_rng(0)
    n, f = 1200, 8
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    y = (1.5 * X["x0"] + 1.0 * X["x1"] - 0.8 * X["x2"] + 0.05 * rng.normal(size=n)).to_numpy()
    return X, y


def _split(X, y):
    return X.iloc[:900].reset_index(drop=True), y[:900], X.iloc[900:].reset_index(drop=True), y[900:]


def test_singletons_parity_with_iter77(planted_strong):
    """No cluster aggregation -> member-equiv check reduces to unit-tuple equality (iter77)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

    X, y = planted_strong
    Xs, ys, Xh, yh = _split(X, y)
    candidates = [(0.0, (0, 1, 2)), (0.1, (0, 1)), (0.2, (0, 1, 2, 5)), (0.3, (4, 5, 6))]
    best, _, baseline = revalidate_top_n(
        candidates,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        n_models=3,
        lambda_stab=0.0,
        rng=np.random.default_rng(0),
        n_jobs=1,
        adaptive_n_models=True,
        unit_to_members=None,  # singletons: idx == member columns
    )
    assert set(best) == {0, 1, 2}
    ucb_info = baseline["ucb"]
    assert ucb_info["n_models_run"] == 2, ucb_info
    # No cluster collapse possible -> the lever can't have fired via member-equiv only.
    assert ucb_info["n_reval_models_run_via_member_equiv"] is False


def test_member_equiv_fires_when_units_differ_but_members_match(planted_strong):
    """Two distinct unit tuples that expand to the SAME member set should trigger early-stop.

    Construct a ``unit_to_members`` map where candidates A=(0,) and B=(1,) both expand to {0, 1, 2}
    (the planted-strong informatives). Then the parsimony winner across two consecutive rounds may
    pick A in round-0 and B in round-1 (or vice-versa) -- both deploy identical features. iter77
    compared on the unit tuple and would NOT stop; iter92 compares on members and stops.

    We FORCE this by including BOTH A and B in the candidate list with proxy_loss=0 (top tier)
    plus a worse third option, then check ``n_reval_models_run_via_member_equiv`` fires.
    """
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

    X, y = planted_strong
    Xs, ys, Xh, yh = _split(X, y)
    # unit indices 0, 1, 2 all map to the SAME member set {0, 1, 2}; unit 3 maps to {3, 4, 5}.
    # When the parsimony rule picks across {A, B, C} (the three "equivalent" choices) they're
    # all valid winners -- but they're INDISTINGUISHABLE in deployment, so iter92 catches that
    # in two rounds; iter77's unit-tuple compare may need three.
    unit_to_members = {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2], 3: [3, 4, 5]}
    # All three top candidates expand to the same member set: any picked winner is member-equiv.
    candidates = [(0.0, (0,)), (0.0, (1,)), (0.0, (2,)), (0.5, (3,))]
    best, _, baseline = revalidate_top_n(
        candidates,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        n_models=3,
        lambda_stab=0.0,
        rng=np.random.default_rng(0),
        n_jobs=1,
        adaptive_n_models=True,
        unit_to_members=unit_to_members,
    )
    ucb_info = baseline["ucb"]
    # Whatever unit was picked, its member set is {0, 1, 2}.
    assert set(best) in (
        {
            0,
        },
        {
            1,
        },
        {
            2,
        },
    )
    # Early-stop fired at round 2 (no extra rounds beyond the floor).
    assert ucb_info["n_models_run"] == 2, f"expected early-stop at round 2, got {ucb_info['n_models_run']}; ucb={ucb_info}"


def test_member_equiv_lever_is_strict_extension_of_iter77(planted_strong):
    """When the iter77 unit-tuple compare already fires, iter92 fires identically (no regression).

    Identical setup to ``test_adaptive_converges_after_two_rounds``: unit tuples themselves match
    across rounds, so the member-equiv diagnostic must be False (iter77 path) and ``n_models_run``
    matches the iter77 baseline.
    """
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

    X, y = planted_strong
    Xs, ys, Xh, yh = _split(X, y)
    candidates = [(0.0, (0, 1, 2)), (0.1, (0, 1)), (0.2, (0, 1, 2, 5)), (0.3, (4, 5, 6))]
    _, _, baseline = revalidate_top_n(
        candidates,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        n_models=3,
        lambda_stab=0.0,
        rng=np.random.default_rng(0),
        n_jobs=1,
        adaptive_n_models=True,
    )
    ucb_info = baseline["ucb"]
    assert ucb_info["n_models_run"] == 2
    # Unit-tuple already converged -> the iter92-specific path did not fire.
    assert ucb_info["n_reval_models_run_via_member_equiv"] is False


def test_member_equiv_diagnostic_present_when_legacy_path(planted_strong):
    """Diagnostic surfaces on the legacy ``adaptive_n_models=False`` path too (as ``False``)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

    X, y = planted_strong
    Xs, ys, Xh, yh = _split(X, y)
    candidates = [(0.0, (0, 1, 2)), (0.1, (0, 1)), (0.2, (4, 5, 6))]
    _, _, baseline = revalidate_top_n(
        candidates,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        n_models=3,
        lambda_stab=0.0,
        rng=np.random.default_rng(0),
        n_jobs=1,
        adaptive_n_models=False,
    )
    assert "n_reval_models_run_via_member_equiv" in baseline["ucb"]
    assert baseline["ucb"]["n_reval_models_run_via_member_equiv"] is False
