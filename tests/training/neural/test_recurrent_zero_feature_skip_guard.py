"""Regression: recurrent wrappers must SKIP (not crash) when feature selection empties the aux feature frame.

When MRMR / RFECV / constant-column dropping removes every tabular column, the recurrent fit used to hand an
``(n, 0)`` array to ``StandardScaler.fit``, which raises ``ValueError: Found array with 0 sample(s) (shape=(0, 0))``.
The booster path was already guarded (``_trainer_train_and_evaluate`` + ``_training_loop._train_model_with_fallback``),
but the recurrent fit at ``_phase_recurrent.train_recurrent_models`` bypasses both. The wrappers now mirror that
0-feature skip: warn + leave ``self.model = None`` so ``predict`` raises ``NotFittedError`` and the suite degrades
gracefully (the recurrent member is simply dropped from the ensemble) instead of aborting.

SEQUENCE_ONLY carries its signal in ``sequences`` and runs fine with no aux features -- the guard must NOT fire there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural.recurrent import (  # noqa: E402
    RecurrentClassifierWrapper,
    RecurrentRegressorWrapper,
)
from mlframe.training.neural._recurrent_config import (  # noqa: E402
    RecurrentConfig,
    InputMode,
    RNNType,
)

_N = 40


def _zero_col_frame() -> pd.DataFrame:
    return pd.DataFrame(index=range(_N))


def _sequences() -> list[np.ndarray]:
    rs = np.random.RandomState(1)
    return [rs.randn(5, 3).astype(np.float32) for _ in range(_N)]


@pytest.mark.parametrize("mode", [InputMode.FEATURES_ONLY, InputMode.HYBRID])
def test_recurrent_classifier_skips_on_zero_features(mode):
    y = np.array([0, 1] * (_N // 2), dtype=np.int64)
    cfg = RecurrentConfig(input_mode=mode, max_epochs=1, hidden_size=8, rnn_type=RNNType.LSTM)
    w = RecurrentClassifierWrapper(config=cfg)
    fit_kwargs = {"labels": y, "features": _zero_col_frame()}
    if mode == InputMode.HYBRID:
        fit_kwargs["sequences"] = _sequences()
    # Pre-fix this raised ValueError from StandardScaler on the (n, 0) array.
    w.fit(**fit_kwargs)
    assert w.model is None
    with pytest.raises(NotFittedError):
        w.predict_proba(features=_zero_col_frame())


@pytest.mark.parametrize("mode", [InputMode.FEATURES_ONLY, InputMode.HYBRID])
def test_recurrent_regressor_skips_on_zero_features(mode):
    y = np.random.RandomState(0).randn(_N).astype(np.float32)
    cfg = RecurrentConfig(input_mode=mode, max_epochs=1, hidden_size=8, rnn_type=RNNType.LSTM)
    w = RecurrentRegressorWrapper(config=cfg)
    fit_kwargs = {"labels": y, "features": _zero_col_frame()}
    if mode == InputMode.HYBRID:
        fit_kwargs["sequences"] = _sequences()
    w.fit(**fit_kwargs)
    assert w.model is None
    with pytest.raises(NotFittedError):
        w.predict(features=_zero_col_frame())


def test_sequence_only_still_fits_with_no_aux_features():
    """The skip-guard keys on the aux frame being EMPTY for a mode that consumes it; SEQUENCE_ONLY does not."""
    y = np.array([0, 1] * (_N // 2), dtype=np.int64)
    cfg = RecurrentConfig(input_mode=InputMode.SEQUENCE_ONLY, max_epochs=1, hidden_size=8, rnn_type=RNNType.LSTM)
    w = RecurrentClassifierWrapper(config=cfg)
    w.fit(labels=y, sequences=_sequences(), features=None)
    assert w.model is not None
