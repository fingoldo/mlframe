"""The adaptive-Fourier and missingness-indicator protections must earn their re-add, like every sibling protection.

Both passes put a column the MRMR screen dropped back into ``support_``. The hinge, orth-basis, raw floor-drop and cat-FE protections each
admit their candidate only when it lifts a held-out linear fit over the design that actually survived; these two re-added unconditionally (the
Fourier pass with no gate at all, the missingness pass on a membership test), while logging that the column was "held-out-validated". The
validation they referred to happened in the generating detector, against raw ``x``, before the screen ran, so a leg fully subsumed by a
surviving composite was still re-added.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._heldout_gate import (
    build_heldout_incr_probe,
    candidate_values,
    coerce_gate_target,
    selected_design_columns,
)


@pytest.fixture
def frame():
    """A target driven entirely by one signal, plus the signal itself as a selected design column."""
    rng = np.random.default_rng(0)
    n = 900
    signal = rng.normal(size=n)
    y = 3.0 * signal + 0.01 * rng.normal(size=n)
    return signal, y


def test_a_candidate_already_spanned_by_the_selected_design_earns_no_gain(frame):
    """A column that is a linear function of an already-selected one adds nothing out of sample, so the gate rejects it."""
    signal, y = frame
    probe = build_heldout_incr_probe(y_gate=y, sel_value_cols=[signal], random_seed=0)
    assert probe(2.0 * signal + 1.0) < 0.003, "a rescaled copy of a selected column must not clear the floor"


def test_a_genuinely_new_signal_earns_gain(frame):
    """The gate is not simply rejecting everything: an independent contributor to the target clears the floor."""
    signal, y = frame
    rng = np.random.default_rng(1)
    extra = rng.normal(size=signal.shape[0])
    probe = build_heldout_incr_probe(y_gate=y + 2.0 * extra, sel_value_cols=[signal], random_seed=0)
    assert probe(extra) >= 0.003


def test_a_leg_pair_is_judged_jointly_not_leg_by_leg():
    """The sin/cos legs of one frequency split the phase between them, so the gate must score them together.

    Each leg alone correlates near zero with the target; together they span it. A per-leg gate would reject the pair this protection exists
    to rescue, which is why the pass groups legs by source before probing.
    """
    rng = np.random.default_rng(2)
    n = 900
    t = np.linspace(0.0, 8.0 * np.pi, n)
    sin_leg, cos_leg = np.sin(t), np.cos(t)
    phase = 1.1
    y = np.cos(phase) * sin_leg + np.sin(phase) * cos_leg + 0.01 * rng.normal(size=n)
    base = [rng.normal(size=n)]
    probe = build_heldout_incr_probe(y_gate=y, sel_value_cols=base, random_seed=0)
    joint = probe(np.column_stack([sin_leg, cos_leg]))
    assert joint >= 0.003, f"the leg pair must clear the floor together, got {joint}"
    assert probe(sin_leg) < joint and probe(cos_leg) < joint, "fixture precondition: neither leg alone should match the pair"


def test_an_unusable_target_leaves_the_gate_open(frame):
    """With no scoreable target the probe cannot decide, and must not silently drop columns: it returns a passing value."""
    signal, _ = frame
    probe = build_heldout_incr_probe(y_gate=None, sel_value_cols=[signal], random_seed=0)
    assert probe(signal) >= 0.003


def test_coerce_gate_target_rejects_a_target_it_cannot_score():
    """A non-finite or wrong-length target is not usable, and says so rather than producing a silent garbage gate."""
    assert coerce_gate_target(np.array([1.0, 2.0, np.nan]), 3) is None
    assert coerce_gate_target(np.array([1.0, 2.0]), 3) is None
    assert coerce_gate_target(np.array([1.0, 2.0, 3.0]), 3) is not None


def test_candidate_values_reads_from_the_snapshot_then_the_frame():
    """Engineered columns come from the fit-time snapshot, raw ones from X, and anything unscoreable returns None."""
    import pandas as pd

    y_ref = np.zeros(4)
    X = pd.DataFrame({"raw": [1.0, 2.0, 3.0, 4.0], "text": list("abcd")})
    snapshot = {"eng": np.array([9.0, 9.0, 9.0, 9.0])}
    assert np.array_equal(candidate_values("eng", X=X, eng_continuous_snapshot=snapshot, y_ref=y_ref), snapshot["eng"])
    assert np.array_equal(candidate_values("raw", X=X, eng_continuous_snapshot={}, y_ref=y_ref), X["raw"].to_numpy())
    assert candidate_values("text", X=X, eng_continuous_snapshot={}, y_ref=y_ref) is None
    assert candidate_values("absent", X=X, eng_continuous_snapshot={}, y_ref=y_ref) is None


def test_selected_design_columns_skips_non_numeric_selected_columns():
    """A selected categorical column is not a regressor for the linear baseline and must be left out of the design."""
    import pandas as pd

    y_ref = np.zeros(4)
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": list("abcd")})
    design = selected_design_columns(X=X, cols=["num", "cat"], selected_vars=[0, 1], eng_continuous_snapshot={}, y_ref=y_ref)
    assert len(design) == 1 and np.array_equal(design[0], X["num"].to_numpy())


class _FourierRecipe:
    """Minimal stand-in for the engineered recipe the pass reads the leg's source column from."""

    def __init__(self, src: str):
        self.src_names = (src,)
        self.name = "leg"


def _run_group1_with_adaptive_leg(subsumed: bool) -> bool:
    """Drive the protection pass with one adaptive-Fourier leg and report whether it ended up in support."""
    import pandas as pd

    from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._group1 import (
        _friend_graph_and_redundancy_passes_group1,
    )
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 900
    src = rng.normal(size=n)
    extra = rng.normal(size=n)
    leg = 2.0 * src + 1.0 if subsumed else extra  # a rescaled copy of the selected column, or a genuinely new signal
    y = 3.0 * src + 0.01 * rng.normal(size=n) + (0.0 if subsumed else 2.0 * extra)

    cols = ["src", "leg"]
    X = pd.DataFrame({"src": src, "leg": leg})
    est = MRMR(random_state=0, verbose=0)
    est._adaptive_fourier_features_ = ["leg"]
    est.missingness_indicator_features_ = []
    est.hybrid_orth_features_ = []

    selected_vars, *_ = _friend_graph_and_redundancy_passes_group1(
        est,
        X=X,
        classes_y=None,
        cols=cols,
        data=np.column_stack([src, leg]),
        nbins=np.array([8, 8], dtype=np.int32),
        target_indices=[],
        y=y,
        verbose=0,
        cached_MIs={},
        engineered_recipes={},
        _eng_continuous_snapshot={"src": src, "leg": leg},
        selected_vars=[0],
        _effective_min_relevance_gain=0.0,
        _hinge_deferred_recipes={},
        _hinge_deferred_values={},
        _hybrid_orth_pre_recipes={"leg": _FourierRecipe("src")},
        _miss_ind_pre_recipes={},
        _persisted_dcd_state=None,
        _y_np=y,
        fe_to_pandas=lambda f: f,
        _fe_family_on=lambda *a, **k: False,
    )
    return 1 in list(selected_vars)


def test_adaptive_fourier_readd_rejects_a_leg_the_selected_design_already_spans():
    """The screen dropped the leg and a surviving column already carries its signal, so the protection must leave it out."""
    assert not _run_group1_with_adaptive_leg(subsumed=True), "a leg subsumed by a selected column was re-added into support"


def test_adaptive_fourier_readd_still_rescues_a_leg_that_adds_signal():
    """The gate must not simply disable the protection: a leg carrying signal nothing else covers is still re-added."""
    assert _run_group1_with_adaptive_leg(subsumed=False), "the protection stopped rescuing a genuinely useful leg"
