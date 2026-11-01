"""Merging the estimator state of concurrently run FE stages gives what the same stages run one after the other would have left."""

import copy

from mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_concurrency import merge_stage_state, private_copy, snapshot


class _Est:
    """A bare estimator with the attribute kinds the stages touch."""

    def __init__(self):
        self.hybrid_orth_features_ = ["a", "b"]
        self._fe_rejection_records_ = [("r", 0)]
        self.mapping_ = {"k": 1}
        self.seen_ = {"x"}
        self.threshold_ = 0.5


def _sequential(stages):
    """Reference: the stages applied one after the other to one estimator."""
    est = _Est()
    for s in stages:
        s(est)
    return est


def _concurrent(stages):
    """The stages applied to private copies of the same starting estimator, then merged in order."""
    est = _Est()
    base = snapshot(est)
    copies = [private_copy(est) for _ in stages]
    for s, c in zip(stages, copies):
        s(c)
    merge_stage_state(est, base, copies)
    return est


def _early_b(e):
    """Create the empty per-family lists, append to the shared ones."""
    e.hybrid_orth_features_.append("eb1")
    e._fe_rejection_records_.append(("eb", 1))
    e.grouped_quantile_features_ = []
    e.modular_features_ = []


def _mid_a(e):
    """Fill two of the per-family lists (created empty by early_b in a sequential run) and append to the shared ones."""
    e.hybrid_orth_features_.extend(["ma1", "ma2"])
    e.grouped_quantile_features_ = ["gq1", "gq2"]
    e.modular_features_ = ["m1"]
    e.mapping_["ma"] = 2


def _mid_b(e):
    """Append to the shared lists, add a set member, change a scalar."""
    e.hybrid_orth_features_.append("mb1")
    e._fe_rejection_records_.append(("mb", 2))
    e.seen_.add("y")
    e.threshold_ = 0.7
    e.lof_features_ = ["l1"]


def test_concurrent_merge_equals_the_sequential_run():
    """Shared lists keep the stage order, per-family lists end filled, dicts, sets and scalars are folded."""
    stages = [_early_b, _mid_a, _mid_b]
    seq, con = _sequential(stages), _concurrent(stages)
    assert vars(con) == vars(seq)
    assert con.hybrid_orth_features_ == ["a", "b", "eb1", "ma1", "ma2", "mb1"]


def test_private_copies_do_not_touch_the_original_or_each_other():
    """Mutating a copy's containers leaves the original and the other copies unchanged."""
    est = _Est()
    c1, c2 = private_copy(est), private_copy(est)
    c1.hybrid_orth_features_.append("x")
    c1.mapping_["z"] = 9
    assert est.hybrid_orth_features_ == ["a", "b"] and c2.hybrid_orth_features_ == ["a", "b"]
    assert "z" not in est.mapping_ and "z" not in c2.mapping_


def test_a_rebuilt_list_is_taken_from_the_last_stage_that_changed_it():
    """A stage that rebinds a shared list to a new list (not an append) wins in stage order."""
    def rebuild(e):
        """Replace the list."""
        e.hybrid_orth_features_ = ["only"]

    est = _Est()
    base = snapshot(est)
    copies = [private_copy(est) for _ in range(2)]
    rebuild(copies[0])
    merge_stage_state(est, base, copies)
    assert est.hybrid_orth_features_ == ["only"]


def test_unchanged_attributes_stay_untouched():
    """No stage changed anything: the original is exactly as it was."""
    est = _Est()
    before = copy.deepcopy(vars(est))
    base = snapshot(est)
    merge_stage_state(est, base, [private_copy(est), private_copy(est)])
    assert vars(est) == before
