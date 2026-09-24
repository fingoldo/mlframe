"""Suppression used from worker threads must not snapshot and restore the process-global filter list."""

import threading
import warnings

import numpy as np

from mlframe.utils.warning_filters import install_filter_once


def test_a_filter_is_installed_once_and_stays():
    marker = "mlframe-test-unique-message-xyz"
    assert install_filter_once(message=marker) is True
    assert install_filter_once(message=marker) is False, "a repeat install must not stack another filter"
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        install_filter_once(message=marker)
    assert not [w for w in seen if marker in str(w.message)]


def test_numeric_suppression_does_not_restore_another_threads_filters():
    """Two threads inside the suppressor must both stay suppressed; catch_warnings let one undo the other."""
    from mlframe.feature_engineering.numerical import _suppress_numeric_warnings

    inside = threading.Barrier(2, timeout=30)
    errors = []

    def work():
        try:
            with _suppress_numeric_warnings():
                inside.wait()  # both threads are inside the block at the same time
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    np.float64(1.0) / np.float64(0.0)  # numpy divide-by-zero
                errors.extend(str(w.message) for w in seen if "divide" in str(w.message))
        except Exception as e:
            errors.append(repr(e))

    threads = [threading.Thread(target=work) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == [], f"a thread lost its suppression while another was inside the same block: {errors}"


def test_the_numeric_suppressor_restores_the_error_state_afterwards():
    from mlframe.feature_engineering.numerical import _suppress_numeric_warnings

    before = np.geterr()
    with _suppress_numeric_warnings():
        assert np.geterr()["divide"] == "ignore"
    assert np.geterr() == before, "np.errstate is thread-local but still scoped: it must be restored on exit"
