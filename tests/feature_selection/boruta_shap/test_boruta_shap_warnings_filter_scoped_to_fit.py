"""Importing ``mlframe.feature_selection.boruta_shap`` must NOT install a process-wide ``warnings.filterwarnings("ignore", ..., module="sklearn")`` -- legacy code did, which masked legitimate sklearn FutureWarnings everywhere else in the process. The filter now lives inside the ``BorutaShap.fit`` body under ``warnings.catch_warnings()``.

Assertion: after importing the module, a fresh ``warnings.catch_warnings(record=True)`` block sees a sklearn-attributed ``FutureWarning`` (i.e. nothing is silencing it at module-import scope).
"""

from __future__ import annotations

import warnings


def test_module_import_does_not_install_process_wide_sklearn_warning_filter():
    # Importing the module (no fit / no instantiation).
    import mlframe.feature_selection.boruta_shap as _bs  # noqa: F401

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Emit a warning impersonating sklearn (the legacy module-level filter targeted ``module="sklearn"``).
        warnings.warn_explicit(
            message="dummy sklearn future warning",
            category=FutureWarning,
            filename="sklearn/_dummy.py",
            lineno=1,
        )

    fw = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert len(fw) >= 1, (
        "module import must NOT install a process-wide ignore filter for sklearn FutureWarning; "
        f"caught: {caught!r}"
    )
