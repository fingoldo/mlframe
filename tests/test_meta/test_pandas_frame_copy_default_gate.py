"""The shared ``pandas_frame_copy_default`` gate over the MRMR filters tree: every ``pandas.DataFrame(data, ...)`` over an existing array passes ``copy=False`` or carries ``# copy-ok: <reason>``.

pandas 3 copies the array (and consolidates a dict of arrays) unless ``copy=False`` is passed; at a few GB that doubles the peak of the step that builds the frame. The gate is a hard failure here, with no baseline: the
count was brought to zero and a new call has to say why it copies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

gate = pytest.importorskip("py_ci_shared.pandas_frame_copy_default", reason="the pinned py-ci-shared release predates the pandas_frame_copy_default gate (1.23.15)")

FILTERS = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "filters"


def test_no_dataframe_call_over_existing_data_copies_it_unannounced():
    """No finding in the MRMR filters tree (``advisory=False``: a finding fails the test)."""
    gate.assert_pandas_frame_copy_default(FILTERS, advisory=False, min_files=100, allow_unparsed=True, use_git=False)
