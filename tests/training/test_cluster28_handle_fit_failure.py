"""Cluster 28 regression: ``_handle_fit_failure`` decides which failed fits are retried and how (audit 2026-11-20, 28-F76, part)."""

import pandas as pd
import pytest

from mlframe.training._training_loop_fallback_helpers import _handle_fit_failure


class _Model:
    """Records set_params calls."""

    def __init__(self):
        self.params = {}

    def set_params(self, **kw):
        """Remember the parameters."""
        self.params.update(kw)


def _handle(message, fit_params=None, model=None, model_type="LGBMClassifier", train_df=None):
    """Call the handler from inside an except block, as the training loop does."""
    model = model or _Model()
    try:
        raise RuntimeError(message)
    except RuntimeError as e:
        return _handle_fit_failure(e, model, model, model_type, train_df, None, fit_params if fit_params is not None else {})


def test_keyboard_interrupt_text_becomes_a_real_keyboard_interrupt():
    """A CatBoost-wrapped KeyboardInterrupt stops the suite instead of being treated as a model failure (28-F76)."""
    with pytest.raises(KeyboardInterrupt):
        _handle("CatBoostError: KeyboardInterrupt")


def test_gpu_callbacks_error_retries_without_callbacks():
    """The GPU-callbacks error drops the callbacks and asks for a retry (28-F76)."""
    params = {"callbacks": [object()], "x": 1}
    again, _, out = _handle("User defined callbacks are not supported for GPU", params)
    assert again is True and "callbacks" not in out and out["x"] == 1


def test_cuda_tree_learner_error_switches_lightgbm_to_cpu():
    """A LightGBM build without CUDA is retried on the CPU (28-F76)."""
    model = _Model()
    again, _, _ = _handle("CUDA Tree Learner was not enabled in this build", model=model)
    assert again is True and model.params == {"device_type": "cpu"}


def test_unsupported_pandas_dtype_is_re_raised():
    """The dtype gap is surfaced by re-raising the original error, not hidden behind a retry (28-F76)."""
    with pytest.raises(RuntimeError, match="pandas dtypes must be int, float or bool"):
        _handle("pandas dtypes must be int, float or bool", train_df=pd.DataFrame({"a": ["x"]}))


def test_unexpected_validation_keyword_is_dropped_and_retried():
    """An estimator that rejects X_val / eval_set loses those fit params and is retried (28-F76)."""
    params = {"X_val": 1, "eval_set": 2, "keep": 3}
    again, _, out = _handle("fit() got an unexpected keyword argument 'X_val'", params)
    assert again is True and out == {"keep": 3}


def test_unrecognised_error_is_not_retried():
    """Any other failure returns try_again False so the caller re-raises it (28-F76)."""
    again, _, out = _handle("something else entirely", {"a": 1})
    assert again is False and out == {"a": 1}
