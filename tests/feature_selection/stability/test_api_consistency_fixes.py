"""Regression tests for public-API consistency / validation fixes in feature_selection.

Covers:
  API1  -- sibling selectors share one random_state default.
  API14 -- HybridSelector classification= disables value-sniffing (2-value float regression).
  API33 -- grok / chatgpt / deepseek MI kernels all validate the [0, 127] bin-code range.
  API34 -- run_efs does not mutate the caller's exclude_columns list and accepts a list.
"""

import inspect

import numpy as np
import pytest


# --------------------------------------------------------------------------- API1
def test_api1_sibling_selectors_share_random_state_default():
    from mlframe.feature_selection.hybrid_selector import HybridSelector
    from mlframe.feature_selection.general import estimate_features_relevancy

    hybrid_default = inspect.signature(HybridSelector.__init__).parameters["random_state"].default
    efs_default = inspect.signature(estimate_features_relevancy).parameters["random_state"].default
    assert hybrid_default == efs_default, (
        f"sibling selectors disagree on random_state default: HybridSelector={hybrid_default!r} vs estimate_features_relevancy={efs_default!r}"
    )


# --------------------------------------------------------------------------- API14
def test_api14_two_value_float_regression_with_classification_false_treated_as_regression():
    """A 2-value FLOAT regression target sniffs as 'binary' via type_of_target. With classification=False the
    selector must reject it as a regression task (not silently feed it to LGBMClassifier)."""
    from mlframe.feature_selection.hybrid_selector import HybridSelector
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 4)), columns=[f"f{i}" for i in range(4)])
    # Exactly two distinct FLOAT values -> type_of_target() == "binary".
    y = np.where(rng.random(40) > 0.5, 1.5, -2.5).astype(np.float64)
    assert np.unique(y).size == 2 and y.dtype.kind == "f"

    sel = HybridSelector(classification=False)
    with pytest.raises(ValueError, match="classification targets only"):
        sel.fit(X, y)


def test_api14_classification_none_warns_on_ambiguous_float_target():
    from mlframe.feature_selection.hybrid_selector import HybridSelector
    import pandas as pd

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    # Float values {1.0, 2.0} sniff as 'binary' via type_of_target -> the ambiguous-float warning must fire.
    y = np.where(rng.random(30) > 0.5, 1.0, 2.0).astype(np.float64)

    sel = HybridSelector(classification=None)
    with pytest.warns(UserWarning, match="float but was value-sniffed"):
        # The warning fires before any heavy fitting; we only assert the warning, swallow downstream errors.
        try:
            sel.fit(X, y)
        except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
            pass


# --------------------------------------------------------------------------- API33
@pytest.mark.parametrize(
    "fn_name",
    [
        "grok_compute_mutual_information",
        "chatgpt_compute_mutual_information",
        "deepseek_compute_mutual_information",
    ],
)
def test_api33_all_mi_entrypoints_reject_out_of_range_bin_codes(fn_name):
    import mlframe.feature_selection.mi as mi

    fn = getattr(mi, fn_name)
    # Bin codes above 127 must raise consistently (int8 kernel would wrap them negative otherwise).
    bad = np.zeros((50, 3), dtype=np.int16)
    bad[0, 0] = 200
    with pytest.raises(ValueError, match=r"\[0, 127\]"):
        fn(data=bad, target_indices=[0], n_bins=15)


@pytest.mark.parametrize(
    "fn_name",
    [
        "grok_compute_mutual_information",
        "chatgpt_compute_mutual_information",
        "deepseek_compute_mutual_information",
    ],
)
def test_api33_valid_input_still_works(fn_name):
    import mlframe.feature_selection.mi as mi

    fn = getattr(mi, fn_name)
    rng = np.random.default_rng(0)
    data = rng.integers(0, 15, size=(200, 4)).astype(np.int8)
    out = fn(data=data, target_indices=np.array([0], dtype=np.int64), n_bins=15)
    assert out.shape == (1, 4)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- API34
def test_api34_run_efs_does_not_mutate_caller_exclude_columns_list():
    pl = pytest.importorskip("polars")
    from mlframe.feature_selection.general import run_efs

    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "t": rng.normal(size=40),
            "f1": rng.normal(size=40),
            "f2": rng.normal(size=40),
            "id": np.arange(40, dtype=float),
        }
    )
    # ``id`` is pre-excluded as a list; the fix must accept the list and not mutate it in place.
    exclude_columns = ["id"]
    exclude_snapshot = list(exclude_columns)

    # use_mis=False keeps the call cheap (no MI kernels) while still exercising the exclude_columns path.
    result = run_efs(
        df=df.lazy(),
        target_columns=["t"],
        exclude_columns=exclude_columns,  # a list -- must be accepted and NOT mutated
        permuted_mutual_informations={},
        binned_targets=None,
        mi_algorithms_ranking=None,
        binning_params={"verbose": 0},
        efs_params={},
        use_mis=False,
    )
    # The caller's list is unchanged (no in-place .update on a list).
    assert exclude_columns == exclude_snapshot
    # The augmented exclusion set is returned (run_efs returns exclude_columns as the 2nd element).
    returned_exclude = result[1]
    assert isinstance(returned_exclude, set)
    assert "id" in returned_exclude
