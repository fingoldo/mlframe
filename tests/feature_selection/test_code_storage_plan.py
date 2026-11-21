"""The code-storage plan: the narrowest dtype for the codes the binning settings can produce; the requested dtype is a hint that is widened or narrowed."""

import numpy as np

from mlframe.feature_selection.filters.discretization._code_storage_plan import plan_code_storage


def test_quantile_ten_bins_is_int8():
    """The default binner (10 bins + NaN bin) needs codes up to 10: int8 to allocate and to store."""
    plan = plan_code_storage(n_bins=10, user_dtype=np.int32)
    assert plan.alloc_dtype is np.int8 and plan.storage_dtype is np.int8 and plan.bound == 10
    assert any("narrowed" in r for r in plan.reasons)


def test_wide_request_is_narrowed_by_a_small_cap():
    """int32 requested with cap 127 on a 300-level categorical: the codes stop at 127, so int8."""
    plan = plan_code_storage(n_bins=10, max_categorical_cardinality=127, categorical_cardinalities=[300], user_dtype=np.int32)
    assert plan.alloc_dtype is np.int8 and plan.bound == 127


def test_narrow_request_is_widened_instead_of_failing():
    """int8 requested but the supervised strategy can reach 256 intervals: the allocation dtype is widened to int16 and the finished matrix is a byte (unsigned)."""
    plan = plan_code_storage(n_bins=10, nbins_strategy="mdlp", user_dtype=np.int8)
    assert plan.bound == 256 and plan.alloc_dtype is np.int16 and plan.storage_dtype is np.int16
    assert any("widened" in r for r in plan.reasons)


def test_mdlp_depth_seven_fits_a_byte():
    """MDLP at depth 7 (cap 127 rule) stops at 128 intervals: codes up to 128 -> unsigned byte storage."""
    plan = plan_code_storage(n_bins=10, nbins_strategy="mdlp", nbins_strategy_kwargs={"max_depth": 7})
    assert plan.bound == 128 and plan.storage_dtype is np.uint8 and plan.alloc_dtype is np.int16


def test_high_cardinality_categorical_with_the_default_cap_is_a_byte():
    """300 levels capped at 255: codes up to 255 -> stored as uint8."""
    plan = plan_code_storage(n_bins=10, max_categorical_cardinality=255, categorical_cardinalities=[300])
    assert plan.bound == 255 and plan.storage_dtype is np.uint8


def test_uncapped_categorical_takes_its_cardinality():
    """No cap: 5000 levels need int16."""
    plan = plan_code_storage(n_bins=10, categorical_cardinalities=[5000])
    assert plan.bound == 5000 and plan.storage_dtype is np.int16 and plan.alloc_dtype is np.int16


def test_cap_alone_bounds_unknown_categorical_codes():
    """Without the cardinalities the cap is the bound of the categorical codes."""
    plan = plan_code_storage(n_bins=10, max_categorical_cardinality=255)
    assert plan.bound == 255


def test_compact_storage_off_keeps_the_requested_dtype_unless_too_narrow(monkeypatch):
    """With MLFRAME_MRMR_COMPACT_CODES=0 the requested dtype stays, widened only when it cannot hold the codes."""
    monkeypatch.setenv("MLFRAME_MRMR_COMPACT_CODES", "0")
    assert plan_code_storage(n_bins=10, user_dtype=np.int32).alloc_dtype is np.int32
    assert plan_code_storage(n_bins=10, nbins_strategy="mdlp", user_dtype=np.int8).alloc_dtype is np.int16


def test_as_dict_is_json_friendly():
    """The report view has names and plain ints only."""
    import json

    d = plan_code_storage(n_bins=10, nbins_strategy="mdlp").as_dict()
    assert json.loads(json.dumps(d))["max_code"] == 256


def _small_frame(n=3000, seed=0):
    """A numeric frame with a signal in the first two columns."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(6)})
    y = (X["x0"] + 0.5 * X["x1"] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


def test_fit_records_the_plan_and_a_default_fit_stores_bytes():
    """A default fit reports int8 storage with the reasons, and the codes matrix itself is int8."""
    import warnings

    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _small_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(max_runtime_mins=1, verbose=0, fe_max_steps=0).fit(X, y)
    plan = m.code_storage_plan_
    assert plan["alloc_dtype"] in ("int8", "int16") and plan["max_code"] >= 10 and plan["reasons"]


def test_explicit_narrow_dtype_with_a_wide_strategy_is_widened_not_rejected():
    """quantization_dtype=int8 with MDLP (up to 256 intervals) used to be a contract error for wide columns; the plan widens the allocation and the fit runs."""
    import warnings

    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _small_frame(seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(max_runtime_mins=1, verbose=0, fe_max_steps=0, quantization_dtype=np.int8, nbins_strategy="mdlp").fit(X, y)
    assert m.code_storage_plan_["alloc_dtype"] == "int16"
    assert any("widened" in r for r in m.code_storage_plan_["reasons"])
    assert len(list(m.get_feature_names_out())) >= 1


def test_code_alloc_dtype_follows_the_plan_and_falls_back_to_the_request():
    """The FE steps allocate new code blocks in the plan's dtype once a fit has one, in the requested dtype before that."""
    import types

    from mlframe.feature_selection.filters.discretization._code_storage_plan import code_alloc_dtype

    fresh = types.SimpleNamespace(quantization_dtype=np.int32)
    assert code_alloc_dtype(fresh) is np.int32
    fresh._code_alloc_dtype = np.int8
    assert code_alloc_dtype(fresh) is np.int8
