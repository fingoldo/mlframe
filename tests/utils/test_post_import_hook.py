"""The CatBoost clone patch is installed when catboost is imported, not when mlframe.metrics is (tracker 2026-11-02 wave 16, IMP16-1)."""

import subprocess
import sys


def _run(code):
    """Run ``code`` in a fresh interpreter and return the last stdout line."""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=1500)
    assert done.returncode == 0, done.stderr[-500:]
    return done.stdout.split()[-1]


def test_importing_metrics_does_not_import_catboost():
    """About 3.2 s of catboost + IPython + xgboost for every importer of mlframe.metrics."""
    assert _run("import sys; import mlframe.metrics.core; print('catboost' in sys.modules)") == "False"


def test_patch_is_present_once_catboost_is_imported_later():
    """Importing catboost after mlframe.metrics still installs __sklearn_clone__ on the estimators."""
    code = "import mlframe.metrics.core; from catboost import CatBoostClassifier; print(hasattr(CatBoostClassifier, '__sklearn_clone__'))"
    assert _run(code) == "True"


def test_patch_is_installed_at_once_when_catboost_was_imported_first():
    """catboost imported before mlframe: the callback runs immediately."""
    code = "from catboost import CatBoostClassifier; import mlframe.metrics.core; print(hasattr(CatBoostClassifier, '__sklearn_clone__'))"
    assert _run(code) == "True"
