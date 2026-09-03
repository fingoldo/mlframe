"""Smoke tests for mlframe.votenrank.utils (E-P1.4)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")


@pytest.mark.fast
def test_import_votenrank_utils_module():
    """Module imports cleanly and exposes expected public names."""
    from mlframe.votenrank import utils as umod

    for name in ("ranking2top", "kendall_tau", "agreement_rate", "tracker_filename"):
        assert callable(getattr(umod, name)), f"{name} not callable"


@pytest.mark.fast
def test_ranking2top_returns_max_indices():
    """ranking2top returns list of index labels where ranking equals the max value."""
    from mlframe.votenrank.utils import ranking2top

    s = pd.Series([1, 3, 3, 2], index=["a", "b", "c", "d"])
    top = ranking2top(s)
    assert set(top) == {"b", "c"}
    assert isinstance(top, list)


@pytest.mark.fast
def test_kendall_tau_identical_rankings_is_one():
    """kendall_tau between AM and an identical ranking is 1.0; each non-AM column gets its own entry."""
    from mlframe.votenrank.utils import kendall_tau

    df = pd.DataFrame(
        {
            "AM": ["1: a", "2: b", "3: c", "4: d"],
            "method1": ["1: a", "2: b", "3: c", "4: d"],
            "method2": ["1: d", "2: c", "3: b", "4: a"],  # fully reversed vs AM
        }
    )
    out = kendall_tau(df)
    assert set(out) == {"method1", "method2"}
    assert out["method1"] == pytest.approx(1.0)
    assert out["method2"] == pytest.approx(-1.0)


@pytest.mark.fast
def test_agreement_rate_top_k_full_overlap():
    """agreement_rate: a method whose top-k models exactly match AM's top-k scores 1.0."""
    from mlframe.votenrank.utils import agreement_rate

    df = pd.DataFrame(
        {
            "AM": ["1: a", "2: b", "3: c", "4: d"],
            "method1": ["1: b", "2: a", "3: d", "4: c"],  # same top-2 set {a,b}, different order
        }
    )
    out = agreement_rate(df, k=2, top_k=True)
    assert out == {"method1": 1.0}


@pytest.mark.fast
def test_agreement_rate_no_overlap():
    """agreement_rate: a method whose top-k models share nothing with AM's top-k scores 0.0."""
    from mlframe.votenrank.utils import agreement_rate

    df = pd.DataFrame(
        {
            "AM": ["1: a", "2: b", "3: c", "4: d"],
            "method1": ["1: d", "2: c", "3: b", "4: a"],  # top-1 is "d", AM top-1 is "a"
        }
    )
    out = agreement_rate(df, k=1, top_k=True)
    assert out == {"method1": 0.0}


@pytest.mark.fast
def test_agreement_rate_bottom_k_and_clamping():
    """agreement_rate: bottom_k mode reads from the tail; k larger than the subset clamps instead of inflating the denominator."""
    from mlframe.votenrank.utils import agreement_rate

    df = pd.DataFrame(
        {
            "AM": ["1: a", "2: b", "3: c"],
            "method1": ["1: c", "2: b", "3: a"],
        }
    )
    # bottom-1: AM's last row is "c", method1's last row is "a" -> no overlap.
    out_bottom = agreement_rate(df, k=1, top_k=False)
    assert out_bottom == {"method1": 0.0}
    # k=10 clamped to the actual subset size (3), not divided by the raw k=10.
    out_clamped = agreement_rate(df, k=10, top_k=True)
    assert out_clamped == {"method1": 1.0}


@pytest.mark.fast
def test_tracker_filename_format():
    """tracker_filename composes model/task/dirpath into expected pattern."""
    from mlframe.votenrank.utils import tracker_filename

    out = tracker_filename(model="lgb", task="auc", dirpath="/exp")
    assert out == "/exp/lgb_auc_0/"


@pytest.mark.fast
def test_parse_tracker_dirname_handles_underscore_in_task_name():
    """VOTENRANK-14: a task name containing its own underscore must not break the model/task split."""
    from mlframe.votenrank.utils import _parse_tracker_dirname

    known_models = ["lgb", "xgb"]
    # tracker_filename("lgb", "roc_auc", dirpath) -> "lgb_roc_auc_0"
    assert _parse_tracker_dirname("lgb_roc_auc_0", known_models) == ("lgb", "roc_auc")


@pytest.mark.fast
def test_parse_tracker_dirname_handles_underscore_in_model_name():
    """VOTENRANK-14: a model name containing its own underscore must not break the model/task split."""
    from mlframe.votenrank.utils import _parse_tracker_dirname

    known_models = ["light_gbm", "xgb"]
    assert _parse_tracker_dirname("light_gbm_auc_0", known_models) == ("light_gbm", "auc")


@pytest.mark.fast
def test_parse_tracker_dirname_simple_case_unchanged():
    """Sanity: the plain no-underscore case still parses exactly as before."""
    from mlframe.votenrank.utils import _parse_tracker_dirname

    assert _parse_tracker_dirname("lgb_auc_0", ["lgb"]) == ("lgb", "auc")


@pytest.mark.fast
def test_parse_tracker_dirname_unknown_model_returns_none():
    """A directory whose model prefix isn't in the known set returns None rather than raising."""
    from mlframe.votenrank.utils import _parse_tracker_dirname

    assert _parse_tracker_dirname("unknown_auc_0", ["lgb", "xgb"]) is None


@pytest.mark.fast
def test_parse_tracker_dirname_malformed_no_numeric_suffix_returns_none():
    """A directory name with no trailing numeric run-index returns None rather than raising ValueError."""
    from mlframe.votenrank.utils import _parse_tracker_dirname

    assert _parse_tracker_dirname("lgb_auc", ["lgb"]) is None
