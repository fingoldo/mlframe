"""The verbose clean-form demotion message must not crash when a correlation could not be measured."""
from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_score_helpers as h


def test_demotion_with_an_unmeasured_prewarp_correlation_formats_the_message():
    """``_config_corr`` returns None when it cannot rebuild a column; the demotion then resolves to the clean form and the message still renders."""
    messages: list = []
    perf = {((("a", "id"), ("b", "id")), "mul", 0): 0.5}
    best_config, best_mi = h._score_one_pair_monotone_equivalent_warp_scores(None, 0.8, "clean_cfg", 0.4, perf, "__prewarp__", True, messages, "pw_cfg", 0.5)
    assert best_config == "clean_cfg" and best_mi == 0.4
    assert messages and "n/a" in messages[0]


def _rejection_gate(gate_ratio: float) -> str:
    """Run the ledger block for a pair no gate accepted and return the gate it is attributed to."""
    records: list = []
    h._score_one_pair_all_values_were_already(False, False, False, False, 0.97, 0, ("cfg", "mul", 0), (0, 1), ["a", "b"], gate_ratio, records, None)
    return records[0]["gate"]


def test_rejection_below_the_prevalence_floor_is_not_labelled_an_operand_uplift_failure():
    """A ratio under the floor misses the prevalence gate; only a ratio above it can have lost to the larger operand."""
    assert _rejection_gate(0.5) == "engineered_mi_prevalence"
    assert _rejection_gate(1.2) == "operand_uplift_floor"
