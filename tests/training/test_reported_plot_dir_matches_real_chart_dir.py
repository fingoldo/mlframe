"""The reported chart directory must be the one charts are actually written to.

``_phase_config_setup`` logs a ``plot_dir=...`` line at suite start; ``_setup_model_directories`` is what
actually creates and returns the chart path. Those two constructions drifted: the log named
``<data_dir>/<models_dir>/<model_name>`` -- the MODELS directory, unslugified and missing the
``target_name`` segment -- while charts go to ``<data_dir>/charts/<target_name>/<model_name>/...``. Anyone
who followed the log line looked somewhere charts are never written and concluded rendering had failed.
"""

from __future__ import annotations

import tempfile
from os.path import join

from mlframe.training.core._setup_helpers import _setup_model_directories, slugify


def test_logged_plot_dir_prefix_is_the_real_chart_dir_prefix():
    """The logged prefix must be a genuine prefix of the path charts are actually created under."""
    target_name, model_name = "text", "prod_jobsdetails_shuffled2"
    with tempfile.TemporaryDirectory() as data_dir:
        plot_file, _models_dir = _setup_model_directories(
            target_name=target_name,
            model_name=model_name,
            target_type="binary_classification",
            cur_target_name="cl_act_total_hired_above_1",
            data_dir=data_dir,
            models_dir="models",
            save_charts=True,
        )
        assert plot_file is not None

        # Exactly the construction _phase_config_setup logs (minus the trailing "..." placeholder for the
        # per-target segments, which are only known inside the per-target loop).
        logged_prefix = f"{data_dir}/charts/{slugify(target_name)}/{slugify(model_name)}"

        assert plot_file.replace("\\", "/").startswith(
            logged_prefix.replace("\\", "/")
        ), f"logged plot_dir prefix {logged_prefix!r} is not a prefix of the real chart dir {plot_file!r}"
        # And pin the specific historical error: charts do not live under <data_dir>/models/.
        assert join(data_dir, "models") not in plot_file
