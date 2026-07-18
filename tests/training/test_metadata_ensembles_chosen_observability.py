"""Wave-8 observability sensor 2: ``metadata["ensembles_chosen"][tt][tname] = flavour_name``.

Contract under test (per Wave-8 observability spec):

  * After ``score_ensemble`` runs inside ``_train_one_target``, the winning flavour name (the dict
    key with the best ranking metric) lands on
    ``metadata["ensembles_chosen"][target_type][target_name]``.
  * Predict-side ``_resolve_chosen_flavour`` (already implemented in core/predict.py) reads the same
    layout, so the train-side stamp + predict-side read are a closed loop.
  * The chooser walks ``oof.integral_error`` -> ``oof.rmse`` -> ``test.*`` -> ``val.*`` (oof is the
    only honest selection surface; val is burned for ES).
  * Confidence-subset variants (``" conf"`` suffix) and side-channel reports (``"_diversity"``) are
    excluded from candidate ranking.
"""

from __future__ import annotations

import pytest


class _FakeEnsResult:
    """Minimal ``ens_result`` stand-in that emulates the ``.metrics`` attribute the chooser reads."""

    def __init__(self, metrics: dict):
        self.metrics = metrics


def test_choose_returns_none_on_empty():
    """No candidates -> None (predict-side fallback fires)."""
    from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour

    assert _choose_ensemble_flavour({}) is None
    assert _choose_ensemble_flavour(None) is None  # type: ignore[arg-type]


def test_choose_picks_lowest_integral_error_on_oof():
    """``oof.integral_error`` is the canonical ranking metric (lower-is-better)."""
    from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour

    ensembles = {
        "arithm": _FakeEnsResult({"oof": {"integral_error": 0.20}}),
        "harm": _FakeEnsResult({"oof": {"integral_error": 0.05}}),
        "geo": _FakeEnsResult({"oof": {"integral_error": 0.15}}),
    }
    assert _choose_ensemble_flavour(ensembles) == "harm"


def test_choose_falls_back_to_oof_rmse():
    """When ``integral_error`` is missing, fall back to ``oof.rmse`` (still lower-is-better)."""
    from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour

    ensembles = {
        "arithm": _FakeEnsResult({"oof": {"rmse": 0.9}}),
        "harm": _FakeEnsResult({"oof": {"rmse": 0.1}}),
        "geo": _FakeEnsResult({"oof": {"rmse": 0.5}}),
    }
    assert _choose_ensemble_flavour(ensembles) == "harm"


def test_choose_skips_conf_variants():
    """``" conf"``-suffixed flavours are NOT independent candidates."""
    from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour

    ensembles = {
        "arithm": _FakeEnsResult({"oof": {"integral_error": 0.20}}),
        "harm": _FakeEnsResult({"oof": {"integral_error": 0.10}}),
        "harm conf": _FakeEnsResult({"oof": {"integral_error": 0.0001}}),  # decoy
        "_diversity": {"foo": "bar"},  # not even an ens_result
    }
    assert _choose_ensemble_flavour(ensembles) == "harm"


def test_choose_skips_underscore_prefixed_side_channels():
    """``_diversity`` is metadata about high-correlation pairs, not a flavour."""
    from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour

    ensembles = {
        "arithm": _FakeEnsResult({"oof": {"rmse": 0.5}}),
        "_diversity": {"high_correlation_pairs": []},
    }
    assert _choose_ensemble_flavour(ensembles) == "arithm"


def test_choose_falls_back_to_test_split_when_oof_missing():
    """No oof present -> fall back to test."""
    from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour

    ensembles = {
        "arithm": _FakeEnsResult({"test": {"rmse": 0.9}}),
        "harm": _FakeEnsResult({"test": {"rmse": 0.1}}),
    }
    assert _choose_ensemble_flavour(ensembles) == "harm"


def test_choose_returns_first_candidate_when_no_metric_present():
    """Deterministic fallback when none of the candidates expose any ranking metric."""
    from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour

    ensembles = {
        "arithm": _FakeEnsResult({}),
        "harm": _FakeEnsResult({}),
    }
    # Must be one of the candidate keys, NOT None.
    assert _choose_ensemble_flavour(ensembles) in ensembles


def test_read_metric_drills_into_class_1_subdict():
    """Classifier metrics often nest under class 1 (positive class). Drill into that level."""
    from mlframe.training.core._ensemble_chooser import _read_ensemble_metric

    ens = _FakeEnsResult({"oof": {1: {"integral_error": 0.042}}})
    assert _read_ensemble_metric(ens, "oof", "integral_error") == pytest.approx(0.042)


def test_read_metric_returns_none_on_inf_or_nan():
    """Non-finite metrics shouldn't game the ranking."""
    from mlframe.training.core._ensemble_chooser import _read_ensemble_metric

    ens_inf = _FakeEnsResult({"oof": {"rmse": float("inf")}})
    ens_nan = _FakeEnsResult({"oof": {"rmse": float("nan")}})
    assert _read_ensemble_metric(ens_inf, "oof", "rmse") is None
    assert _read_ensemble_metric(ens_nan, "oof", "rmse") is None


def test_read_metric_returns_none_for_missing_split():
    """Read metric returns none for missing split."""
    from mlframe.training.core._ensemble_chooser import _read_ensemble_metric

    ens = _FakeEnsResult({"oof": {"integral_error": 0.05}})
    assert _read_ensemble_metric(ens, "val", "integral_error") is None


# ----------------------------------------------------------------------------
# Wave-8 spec test: synthetic suite where harm wins -> metadata stamped "harm".
# This drives the chooser at the same call site _train_one_target uses, then asserts the
# nested key lands on the right metadata path.
# ----------------------------------------------------------------------------


def test_stamping_layout_harm_wins():
    """End-to-end-of-chooser: stamp the same way _train_one_target does and assert the layout."""
    from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour

    ensembles_for_target = {
        "arithm": _FakeEnsResult({"oof": {"integral_error": 0.30}}),
        "harm": _FakeEnsResult({"oof": {"integral_error": 0.05}}),  # winner
        "geo": _FakeEnsResult({"oof": {"integral_error": 0.20}}),
    }
    metadata: dict = {}
    target_type = "binary_classification"
    target_name = "y1"
    _chosen = _choose_ensemble_flavour(ensembles_for_target)
    if _chosen is not None:
        metadata.setdefault("ensembles_chosen", {}).setdefault(target_type, {})[target_name] = _chosen

    assert metadata["ensembles_chosen"][target_type][target_name] == "harm"


def test_predict_read_matches_train_stamp():
    """Round-trip: stamp via the train-side path, read via the predict-side helper."""
    from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour
    from mlframe.training.core.predict import _resolve_chosen_flavour

    ensembles_for_target = {
        "arithm": _FakeEnsResult({"oof": {"integral_error": 0.30}}),
        "harm": _FakeEnsResult({"oof": {"integral_error": 0.05}}),
        "mean": _FakeEnsResult({"oof": {"integral_error": 0.20}}),
    }
    metadata: dict = {}
    target_type = "regression"
    target_name = "rev_y"
    _chosen = _choose_ensemble_flavour(ensembles_for_target)
    metadata.setdefault("ensembles_chosen", {}).setdefault(target_type, {})[target_name] = _chosen

    _resolved = _resolve_chosen_flavour(metadata, target_type=target_type, target_name=target_name)
    assert _resolved == "harm"
