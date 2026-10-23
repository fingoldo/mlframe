"""A pair candidate with an inert raw leaf is kept out of the pool; a candidate whose both leaves matter is kept."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters._mrmr_fe_step._step_inert_operands import _raw_leaves, filter_inert_leaf_block


def _setup(candidate_uses_e: bool, n: int = 6000, seed: int = 0):
    """Binned pool of raw ``c`` and ``e`` plus one candidate column: ``rint(c)``-like (``e`` inert) or a function of both."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 8, n)
    e = rng.integers(0, 8, n)
    y = rng.integers(0, 2, n)
    data = np.column_stack([c, e, y]).astype(np.int32)
    cand = (c + e) % 8 if candidate_uses_e else c.copy()
    self = SimpleNamespace(feature_names_in_=np.array(["c", "e"]), random_seed=0)
    cols = ["c", "e", "y"]
    features = [((("c", 0), ("e", 0)), 0)]
    return self, features, cand.astype(np.float64).reshape(-1, 1), ["max(rint(c),neg(e))"], np.array([8]), cols, data


def _identity(values):
    """The candidate values are already codes in this fixture."""
    return values.astype(np.int32)


def test_raw_leaves_reads_the_distinct_raw_names_of_an_engineered_name():
    """Tokens that are raw columns, once each, in order of appearance."""
    assert _raw_leaves("max(rint(c),neg(e))", {"c", "e", "d"}) == ["c", "e"]
    assert _raw_leaves("div(sqr(a),abs(a))", {"a"}) == ["a"]


def test_a_candidate_that_ignores_one_leaf_is_dropped():
    """The candidate equals ``c``: ``e`` adds nothing given ``c``, so the block is emptied."""
    self, features, vals, new_cols, new_nbins, cols, data = _setup(candidate_uses_e=False)
    kept, _vals, kept_cols, _nbins, original = filter_inert_leaf_block(self, features, vals, new_cols, new_nbins, cols, data, _identity, 0)
    assert kept == [] and kept_cols == [] and original == []


def test_a_candidate_that_uses_both_leaves_is_kept():
    """The candidate is ``(c + e) mod 8``: each leaf is informative about it given the other, so nothing is dropped."""
    self, features, vals, new_cols, new_nbins, cols, data = _setup(candidate_uses_e=True)
    kept, _vals, kept_cols, _nbins, _original = filter_inert_leaf_block(self, features, vals, new_cols, new_nbins, cols, data, _identity, 0)
    assert kept == features and kept_cols == new_cols


def test_without_pool_data_the_block_is_returned_unchanged():
    """No data to read the leaves from: the filter does nothing."""
    self, features, vals, new_cols, new_nbins, cols, _data = _setup(candidate_uses_e=False)
    kept, _vals, kept_cols, _nbins, _original = filter_inert_leaf_block(self, features, vals, new_cols, new_nbins, cols, None, _identity, 0)
    assert kept == features and kept_cols == new_cols


def test_the_dropped_candidate_is_written_to_the_rejection_ledger_with_its_numbers():
    """The drop leaves one ``inert_leaf`` row: the candidate, its raw leaves, the CMI that missed and the floor it had to clear."""
    self, features, vals, new_cols, new_nbins, cols, data = _setup(candidate_uses_e=False)
    filter_inert_leaf_block(self, features, vals, new_cols, new_nbins, cols, data, _identity, 0)
    rows = [r for r in self._fe_rejection_records_ if r["gate"] == "inert_leaf"]
    assert len(rows) == 1
    row = rows[0]
    assert row["candidate"] == "max(rint(c),neg(e))" and "e" in row["reason"]
    assert row["observed"] <= row["threshold"]


def test_admitted_and_rejected_pair_records_are_routed_to_their_own_ledgers():
    """An admitted record lands in the admission list with its numbers; any other record is a rejection."""
    from mlframe.feature_selection.filters._fe_rejection_ledger import drain_pair_records, populate_fe_admission_ledger

    holder = SimpleNamespace(feature_names_in_=np.array(["c", "e"]))
    drain_pair_records(
        holder,
        [
            {"verdict": "admitted", "stage": "pair_gate", "candidate": "mul(c,e)", "operands": (0, 1), "operator": "", "observed": 0.97, "best_mi": 0.097, "pair_mi": 0.1, "reason": "passed"},
            {"gate": "marginal_pair_mi_prescreen", "candidate": "(c,e)", "operands": (0, 1), "observed": 0.5, "threshold": 1.05, "reason": "ratio below the bar"},
        ],
        step=2,
    )
    assert [r["candidate"] for r in holder._fe_rejection_records_] == ["(c,e)"]
    populate_fe_admission_ledger(holder)
    frame = holder.fe_admission_ledger_
    assert list(frame["candidate"]) == ["mul(c,e)"] and float(frame["best_mi"].iloc[0]) == 0.097 and int(frame["step"].iloc[0]) == 2
    assert list(frame["operand_names"]) == ["(c, e)"]


def test_provenance_marks_the_rows_that_reached_the_output_and_names_the_gate_that_dropped_the_others():
    """``in_output`` is True for a column of the finished output; a produced column that is not carries the last ledger gate that dropped it."""
    from mlframe.feature_selection.filters._mrmr_fe_provenance import _output_membership

    class _Fitted:
        """Estimator stand-in with an output of two columns and one rejection record."""

        _fe_rejection_records_ = [{"gate": "operand_uplift_floor", "candidate": "x"}, {"gate": "inert_leaf", "candidate": "x"}]

        @staticmethod
        def get_feature_names_out():
            """The finished output."""
            return np.array(["c", "mul(c,d)"])

    names, last_gate = _output_membership(_Fitted())
    assert names == {"c", "mul(c,d)"}
    assert last_gate == {"x": "inert_leaf"}
    names_missing, gates_missing = _output_membership(SimpleNamespace())
    assert names_missing is None and gates_missing == {}
