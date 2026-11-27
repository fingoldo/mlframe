"""Cluster 28 regression: the val / test split-metric emitters pass the right arguments to ``_compute_split_metrics`` (audit 2026-11-20, 28-F76, part)."""

from mlframe.training import _trainer_train_and_evaluate_helpers as h


def _capture(monkeypatch):
    """Replace the metrics computation with a recorder of its keyword arguments."""
    calls = []

    def fake(**kwargs):
        """Record the call."""
        calls.append(kwargs)
        return "result"

    monkeypatch.setattr(h, "_compute_split_metrics", fake)
    return calls


def test_val_emitter_is_a_noop_without_a_val_split(monkeypatch):
    """No val configuration: nothing is computed and None is returned (28-F76)."""
    calls = _capture(monkeypatch)
    assert h._run_val_split_metrics(None, {}, True, {}) is None
    assert calls == []


def test_val_emitter_unpacks_the_configuration_and_tunes_the_threshold(monkeypatch):
    """The val split is named, uses its own metrics dict, knows whether a test split exists, and tunes the F1 threshold (28-F76)."""
    calls = _capture(monkeypatch)
    cfg = ("val", "df", "target", "idx", "preds", "probs", "details", "sc")
    metrics = {"val": {"m": 1}}
    out = h._run_val_split_metrics(cfg, metrics, True, {"extra": 7})
    assert out == "result"
    (c,) = calls
    assert c["split_name"] == "val" and c["metrics_dict"] is metrics["val"]
    assert (c["df"], c["target"], c["idx"], c["preds"], c["probs"], c["details"]) == ("df", "target", "idx", "preds", "probs", "details")
    assert c["has_other_splits"] is True and c["tune_f1_threshold"] is True and c["extra"] == 7


def test_test_emitter_is_a_noop_when_the_test_split_is_off(monkeypatch):
    """_run_test false: nothing is computed (28-F76)."""
    calls = _capture(monkeypatch)
    assert h._run_test_split_metrics(False, {}, "d", "t", "i", "p", "pr", "de", {}) is None
    assert calls == []


def test_test_emitter_reuses_the_val_f1_threshold(monkeypatch):
    """The test split is scored at the threshold tuned on val; a missing val dict gives None (28-F76)."""
    calls = _capture(monkeypatch)
    metrics = {"val": {"f1_opt_threshold": 0.37}, "test": {}}
    h._run_test_split_metrics(True, metrics, "d", "t", "i", "p", "pr", "de", {})
    assert calls[-1]["f1_opt_threshold"] == 0.37 and calls[-1]["split_name"] == "test" and calls[-1]["has_other_splits"] is False
    h._run_test_split_metrics(True, {"val": None, "test": {}}, "d", "t", "i", "p", "pr", "de", {})
    assert calls[-1]["f1_opt_threshold"] is None
