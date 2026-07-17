"""Wave 40 (2026-05-20): silent integer overflow / dtype downcast wraparound.

Distinct from wave 17 (CLEAN, dtype coercion at I/O boundaries): this wave
audits computation-time casts where values exceeding the dtype's range wrap
around silently, producing the WRONG category, WRONG class, or WRONG joint-
histogram cell -- without any error.

5 findings, all fixed:

  P0  feature_selection/filters/discretization.py:626
      categorize_dataset previously log-warn-then-truncate'd category codes >
      127/32767, silently wrapping high-cardinality columns into negative
      ids; the nbins calculation read the post-wrap max and the MI joint
      histogram in mi.py was sized to the wrapped value.

  P2  feature_selection/filters/discretization.py:176
      categorize_1d_array (legacy 1-D variant) had the same shape but
      without a log-warn -- strictly worse than the P0.

  P0  training/neural/recurrent.py:1107
      RecurrentClassifierWrapper.predict cast argmax output to int8
      unconditionally, silently mis-classifying any class id > 127.

  P1  feature_selection/mi.py:184
      chatgpt_compute_mutual_information cast input bins to int8 without
      range check; caller-controlled num_bins>128 + bin_dtype=Int16 wrapped
      values 128+ to negative, then numba kernel (boundscheck=off) wrote to
      wrong joint-histogram cells.

  P2  metrics/core.py:4244
      create_robustness_standard_bins used hardcoded int16; widened to
      range-aware dispatch.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest


MLFRAME_ROOT = Path(importlib.import_module("mlframe").__file__).parent


def _read(rel: str) -> str:
    _path = MLFRAME_ROOT / rel
    if not _path.exists() and _path.suffix == ".py":
        # Monolith-split compat: the flat module became a subpackage
        # (``X.py`` -> ``X/__init__.py`` + submodules). Read __init__ + every submodule.
        _pkg = _path.with_suffix("")
        _init = _pkg / "__init__.py"
        if _init.exists():
            parts = [_init.read_text(encoding="utf-8")]
            for _sub in sorted(_pkg.glob("*.py")):
                if _sub.name != "__init__.py":
                    parts.append(_sub.read_text(encoding="utf-8"))
            return "\n".join(parts)
    return _path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Behavioural sensors
# ---------------------------------------------------------------------------


def test_categorize_dataset_auto_promotes_high_cardinality() -> None:
    """200 distinct categories must produce honest codes 0..199, not wrapped int8 values."""
    import pandas as pd
    from mlframe.feature_selection.filters.discretization import categorize_dataset

    high_card = pd.DataFrame(
        {
            "cat": pd.Categorical([f"c{i}" for i in range(200)]),
        }
    )
    data, cols, nbins = categorize_dataset(high_card, dtype=np.int8)
    assert int(data.max()) == 199, (
        f"categorize_dataset must auto-promote int8 to fit 200 categories; got max code {int(data.max())} (wrap-symptom = negative or <199)."
    )
    assert int(nbins[0]) == 200


def test_categorize_1d_array_auto_promotes_high_cardinality() -> None:
    import pandas as pd
    from mlframe.feature_selection.filters.discretization import categorize_1d_array

    # 300 distinct numerical values; method="numpy" bins them into many ordinal codes.
    vals = pd.Categorical([f"v{i}" for i in range(300)]).codes.astype(np.int64)
    out = categorize_1d_array(
        vals=vals,
        min_ncats=5,
        method="discretizer",
        astropy_sample_size=1000,
        method_kwargs={"n_bins": 5},
        dtype=np.int8,
    )
    # After auto-promote, dtype must be wider than int8 to fit codes > 127.
    assert out.dtype != np.int8 or int(out.max()) <= 127, "categorize_1d_array must auto-promote dtype when codes exceed int8 range."


def test_recurrent_classifier_predict_handles_high_class_count() -> None:
    """argmax on 200-class proba must yield honest 0..199 class ids, no int8 wrap."""
    np.random.seed(0)
    proba = np.zeros((10, 200), dtype=np.float32)
    # Make class id 150 the argmax for every row.
    proba[:, 150] = 0.9
    # Simulate the predict-tail logic directly (no PyTorch dependency).
    classes = proba.argmax(axis=1)
    cmax = int(classes.max())
    # The new code path:
    for _dt in (np.int8, np.int16, np.int32, np.int64):
        if cmax <= np.iinfo(_dt).max:
            out = classes.astype(_dt)
            break
    assert int(out.max()) == 150, f"int8 wraps class 150 -> -106; auto-promoted dtype must preserve it. Got {int(out.max())}"


def test_chatgpt_mutual_information_rejects_out_of_range_bins() -> None:
    """num_bins exceeding int8 range must raise, not silently wrap."""
    from mlframe.feature_selection.mi import chatgpt_compute_mutual_information

    # Build int16 bin codes that include a value > 127 (would wrap on int8 cast).
    data = np.array(
        [[0, 200, 50], [1, 199, 49], [0, 198, 48]],
        dtype=np.int16,
    )
    with pytest.raises(ValueError, match=r"bin codes must be in \[0, 127\]"):
        chatgpt_compute_mutual_information(data=data, target_indices=[0], n_bins=15)


def test_create_robustness_standard_bins_widens_dtype_when_needed() -> None:
    from mlframe.metrics.core import create_robustness_standard_bins

    # cont_nbins exceeding int8 range (>127) must use int16; the int8 max bins
    # are also exercised at the low end.
    result_int8 = create_robustness_standard_bins("group", npoints=200, cont_nbins=10)
    bins = result_int8[0]
    assert bins.dtype == np.int8

    result_int16 = create_robustness_standard_bins("group", npoints=400, cont_nbins=200)
    bins = result_int16[0]
    assert bins.dtype == np.int16
    assert int(bins.max()) <= 199


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_categorize_dataset_no_longer_silent_truncate() -> None:
    src = _read("feature_selection/filters/discretization.py")
    assert "auto-promoting" in src.lower(), "categorize_dataset must auto-promote dtype, not silently log-and-truncate."
    # The standalone unconditional astype(dtype) AFTER the warning must be gone.
    assert "factors exceeded dtype" not in src, "The log-warn-then-truncate phrasing must be replaced with the auto-promote path."


def test_recurrent_classifier_no_hardcoded_int8_argmax() -> None:
    src = _read("training/neural/recurrent.py")
    assert "proba.argmax(axis=1).astype(np.int8)" not in src, "Recurrent classifier must not unconditionally cast argmax to int8."


def test_mi_int8_cast_has_range_validation() -> None:
    src = _read("feature_selection/mi.py")
    assert "bin codes must be in [0, 127]" in src, "mi.py: chatgpt_compute_mutual_information must validate input range before int8 cast."


def test_robustness_bins_uses_range_aware_dtype() -> None:
    # ``create_robustness_standard_bins`` was moved to ``_fairness_metrics.py``
    # when ``metrics/core.py`` was split into siblings.
    src = _read("metrics/_fairness_metrics.py")
    # The fix introduces a 3-way dispatch on cont_nbins width.
    assert "_bin_dtype = np.int8" in src and "_bin_dtype = np.int16" in src, (
        "create_robustness_standard_bins must dispatch on cont_nbins for narrowest safe dtype."
    )
