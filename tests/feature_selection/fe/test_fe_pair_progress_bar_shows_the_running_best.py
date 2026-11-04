"""The pair progress bar reports the best engineered feature of the whole sweep so far, not the best of the pair just evaluated."""

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core_helpers import _check_prospective__no_config_nan_edge


class _Bar:
    """Progress bar stub that keeps the last postfix."""

    def __init__(self):
        """Start with no postfix."""
        self.postfix = None

    def set_postfix(self, pf, refresh=False):
        """Record the postfix."""
        self.postfix = dict(pf)


def test_a_weaker_later_pair_does_not_replace_the_best_shown():
    """The second pair's own best (0.30) is below the first pair's (0.50), so the bar must still show 0.50."""
    bar = _Bar()
    state = {"mi": -1.0, "name": None}
    cols = ["a", "b", "c", "d"]
    _check_prospective__no_config_nan_edge(1, 0.50, (((0, "identity"), (1, "identity")), "mul", 0), state, cols, (0, 1), bar)
    _check_prospective__no_config_nan_edge(1, 0.30, (((2, "identity"), (3, "identity")), "mul", 1), state, cols, (2, 3), bar)
    assert "0.5000" in bar.postfix["best"], bar.postfix
