"""Two metric readers silently found nothing while the metrics were printed a few log lines above.

Both symptoms came from a SHAPE the reader did not handle, not from a missing value:

- ``_choose_ensemble_flavour`` probed 9 metrics across 3 splits for 7 flavours and matched none, so the winning
  ensemble was picked by dict-insertion order instead of by quality. ``score_ensemble`` returns the raw
  ``(namespace, train_df, val_df, test_df)`` tuple, and a tuple has no ``.metrics``.
- the cross-target verdict printed ``best_model=-`` for a binary target. ``_entry_metric`` read
  ``metrics[split][name]`` but classification metrics live one level deeper, under the class key.

That mattered: in the run that surfaced this, RANK_AVERAGE scored test Brier 23.14% against 20.75% for every
other flavour, so a working chooser has real work to do.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour
from mlframe.training.core._misc_helpers import _entry_metric


def _entry(**per_class):
    """A classification entry whose metrics sit under the class-1 key, as every binary run produces."""
    return SimpleNamespace(metrics={"test": {1: dict(per_class)}})


class TestTheEnsembleChooser:
    """It must rank by metric, and it must survive the shape its own caller hands it."""

    def _raw(self):
        """``score_ensemble``'s real return shape: flavour -> 4-tuple."""
        return {
            "arithm": (_entry(roc_auc=0.70, brier_loss=0.21), None, None, None),
            "harm": (_entry(roc_auc=0.78, brier_loss=0.17), None, None, None),
            "rank_average": (_entry(roc_auc=0.71, brier_loss=0.23), None, None, None),
        }

    def test_picks_the_best_flavour_from_the_tuple_shape(self):
        """The defect in one assertion: this used to return 'arithm', the first key, regardless of quality."""
        assert _choose_ensemble_flavour(self._raw()) == "harm"

    def test_bare_namespaces_still_work(self):
        """A caller that already unwrapped must not be broken by the tolerance added for the one that did not."""
        plain = {k: v[0] for k, v in self._raw().items()}
        assert _choose_ensemble_flavour(plain) == "harm"

    def test_the_worst_flavour_is_not_chosen(self):
        """rank_average destroys calibration by construction; a working chooser must never land on it here."""
        assert _choose_ensemble_flavour(self._raw()) != "rank_average"

    def test_no_metrics_anywhere_still_falls_back_deterministically(self):
        """The fallback is legitimate when there really is nothing to rank by -- it must stay deterministic."""
        empty = {"arithm": (SimpleNamespace(metrics={}), None, None, None), "harm": (SimpleNamespace(metrics={}), None, None, None)}
        assert _choose_ensemble_flavour(empty) == "arithm"

    def test_empty_input_returns_none(self):
        """No candidates at all is a different answer from "could not rank them"."""
        assert _choose_ensemble_flavour({}) is None


class TestTheEntryMetricReader:
    """Every metric layout the suite actually produces has to resolve."""

    def test_class_indexed_classification_metrics_resolve(self):
        """The shape that made the verdict table print "-" for a binary target."""
        assert _entry_metric(_entry(log_loss=0.49), "test", "log_loss") == pytest.approx(0.49)

    def test_flat_regression_metrics_still_resolve(self):
        """The layout that already worked must keep working."""
        assert _entry_metric(SimpleNamespace(metrics={"test": {"rmse": 11.63}}), "test", "rmse") == pytest.approx(11.63)

    def test_split_prefixed_metrics_still_resolve(self):
        """A third legacy shape the reader already tolerated."""
        assert _entry_metric(SimpleNamespace(metrics={"test_rmse": 3.5}), "test", "rmse") == pytest.approx(3.5)

    def test_tuple_wrapped_entry_resolves(self):
        """Same unwrap the chooser needed, for the same reason."""
        assert _entry_metric((_entry(roc_auc=0.82), None, None, None), "test", "roc_auc") == pytest.approx(0.82)

    def test_multiclass_picks_a_class_deterministically(self):
        """With several class keys the lowest is taken, matching the order the per-class report prints."""
        entry = SimpleNamespace(metrics={"test": {2: {"roc_auc": 0.6}, 0: {"roc_auc": 0.9}}})
        assert _entry_metric(entry, "test", "roc_auc") == pytest.approx(0.9)

    def test_a_genuine_miss_is_still_nan(self):
        """The tolerance must not start inventing values for a metric nobody computed."""
        assert np.isnan(_entry_metric(_entry(roc_auc=0.8), "test", "not_computed"))

    def test_wrong_split_is_a_miss(self):
        """Reading val when only test exists must not silently fall through to test."""
        assert np.isnan(_entry_metric(_entry(roc_auc=0.8), "val", "roc_auc"))

    def test_a_non_entry_is_a_miss_not_a_crash(self):
        """The reader runs over a heterogeneous model list, so a stray value must not take the summary down."""
        assert np.isnan(_entry_metric("not an entry", "test", "roc_auc"))


class TestRankFusionCompetesOnEveryMetric:
    """A rank blend is NOT excluded from the calibration probes, and that is a deliberate decision.

    Its output is a normalised rank, so its Brier is not the same quantity a probabilistic blend's is -- the
    production run showed rank_average at 23.14% against 20.75% for everything else. Excluding it would remove a
    candidate the caller asked for, so the chooser keeps ranking it and the tradeoff is documented at the top of
    ``_ensemble_chooser``. This test pins the decision so a future "fix" has to argue with it rather than assume it.
    """

    def test_a_rank_blend_can_win_on_a_calibration_metric(self):
        """The rejected alternative would have made this return 'arithm'."""
        pool = {
            "arithm": SimpleNamespace(metrics={"oof": {1: {"brier_loss": 0.20}}}),
            "rank_average": SimpleNamespace(metrics={"oof": {1: {"brier_loss": 0.10}}}),
        }
        assert _choose_ensemble_flavour(pool) == "rank_average"

    def test_auc_is_probed_before_any_calibration_metric(self):
        """Why the tradeoff is near-moot: the rank-invariant metric decides first whenever it is present."""
        from mlframe.training.core._ensemble_chooser import _ENSEMBLE_RANK_METRIC_CANDIDATES

        _oof = [m for s, m, _ in _ENSEMBLE_RANK_METRIC_CANDIDATES if s == "oof"]
        assert _oof.index("roc_auc") < min(_oof.index(m) for m in ("ice", "brier_loss", "log_loss"))

    def test_auc_still_decides_when_both_metrics_are_present(self):
        """With an AUC on the table the calibration keys never get to express the incomparable quantity."""
        pool = {
            "arithm": SimpleNamespace(metrics={"oof": {1: {"roc_auc": 0.78, "brier_loss": 0.20}}}),
            "rank_average": SimpleNamespace(metrics={"oof": {1: {"roc_auc": 0.70, "brier_loss": 0.10}}}),
        }
        assert _choose_ensemble_flavour(pool) == "arithm"
