"""The ACF helpers live in a neutral module; the chart path and feature engineering share them (tracker 2026-11-02 wave 16, IMP16-5)."""

import subprocess
import sys

from mlframe.feature_engineering import acf_lag_selection
from mlframe.reporting.charts import _acf


def test_both_import_paths_resolve_to_the_same_functions():
    """acf_lag_selection used to import them through mlframe.reporting.charts (the plotting stack); the old chart path stays as a re-export."""
    assert acf_lag_selection.pacf_levinson.__module__ == "mlframe.utils.acf_kernels"
    assert _acf.pacf_levinson is acf_lag_selection.pacf_levinson
    assert _acf.significance_band is acf_lag_selection.significance_band


def test_the_kernel_module_needs_neither_reporting_nor_matplotlib():
    """The neutral module is pure numpy."""
    code = "import sys; import mlframe.utils.acf_kernels; print('mlframe.reporting' in sys.modules, 'matplotlib' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=900)
    assert done.returncode == 0, done.stderr[-400:]
    assert done.stdout.split()[-2:] == ["False", "False"]
