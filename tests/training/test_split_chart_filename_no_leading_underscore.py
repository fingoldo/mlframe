"""Regression: per-split chart filenames must not start with an underscore (INV-54).

``_setup_model_directories`` returns a chart ``plot_file`` ending in ``os.sep``; the
per-split metrics path then appended the split name. The old ``f"{plot_file}_{split}"``
idiom produced ``<dir>/_val_perfplot.png`` (underscore-prefixed basename, sorts oddly).
The fix uses ``os.path.join(plot_file, split)`` -> ``<dir>/val_perfplot.png``.
"""
from __future__ import annotations

import os

from mlframe.training.core._setup_helpers import _setup_model_directories


def test_split_plot_file_basename_has_no_leading_underscore(tmp_path):
    plot_file, _ = _setup_model_directories(
        target_name="t", model_name="m", target_type="binary",
        cur_target_name="t", data_dir=str(tmp_path), models_dir="models",
        save_charts=True,
    )
    assert plot_file is not None and plot_file.endswith(os.path.sep)

    # Mirror the production join (training/_eval_helpers._compute_split_metrics).
    for split in ("train", "val", "test"):
        split_plot_file = os.path.join(plot_file, split)
        perfplot = split_plot_file + "_perfplot.png"
        base = os.path.basename(perfplot)
        assert not base.startswith("_"), (
            f"split chart basename {base!r} must not start with '_'; the join-based "
            "naming should produce e.g. 'val_perfplot.png' (INV-54)"
        )
        assert base == f"{split}_perfplot.png"
