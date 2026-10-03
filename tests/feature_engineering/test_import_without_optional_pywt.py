"""``import mlframe`` must work without pywavelets, which only the optional ``signal`` extra installs."""

import subprocess
import sys
import textwrap


def test_feature_engineering_imports_when_pywt_is_absent():
    """A blocked ``pywt`` must not break importing the feature-engineering package (the fs-benchmark nightly installs no ``signal`` extra)."""
    code = textwrap.dedent(
        """
        import sys
        sys.modules["pywt"] = None  # makes ``import pywt`` raise ImportError
        import mlframe.feature_engineering.timeseries  # noqa: F401
        import mlframe.feature_selection  # noqa: F401
        print("imported")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, check=False)  # nosec B603
    assert proc.returncode == 0 and "imported" in proc.stdout, proc.stderr[-2000:]
