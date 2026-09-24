"""PreprocessingExtensionsConfig.random_seed inherits the suite's split seed unless the caller pins one."""

from mlframe.training.configs import PreprocessingExtensionsConfig
from mlframe.training.core._phase_helpers_fit_pipeline import extensions_with_split_seed


def test_unset_seed_inherits_the_split_seed():
    assert PreprocessingExtensionsConfig().random_seed is None
    assert extensions_with_split_seed(None, 7).random_seed == 7
    assert extensions_with_split_seed({"pysr_enabled": False}, 8).random_seed == 8
    assert extensions_with_split_seed(PreprocessingExtensionsConfig(), 9).random_seed == 9


def test_explicit_seed_is_kept():
    assert extensions_with_split_seed(PreprocessingExtensionsConfig(random_seed=0), 7).random_seed == 0
    assert extensions_with_split_seed({"random_seed": 3}, 7).random_seed == 3


def test_suite_threads_the_split_seed(monkeypatch):
    """The suite hands _phase_fit_pipeline the inherited seed (checked at the call boundary, before any fitting)."""
    import mlframe.training.core._main_train_suite as suite

    seen = {}

    class _Stop(Exception):
        pass

    def _capture(**kwargs):
        seen["seed"] = kwargs["preprocessing_extensions"].random_seed
        raise _Stop

    monkeypatch.setattr(suite, "_phase_fit_pipeline", _capture)
    import numpy as np
    import pandas as pd

    from mlframe.training.configs import TrainingSplitConfig

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "y": rng.integers(0, 2, 200)})
    try:
        suite.train_mlframe_models_suite(df=df, target_name="y", model_name="m", features_and_targets_extractor=None,
                                         mlframe_models=["linear"], split_config=TrainingSplitConfig(random_seed=11), verbose=0)
    except _Stop:
        pass
    assert seen.get("seed") == 11
