"""Wave 56 (2026-05-20): missing super().__init__() in subclasses.

Audit class: subclass __init__ that omits super().__init__(), leaving parent's
state uninitialised. Symptom is silent today because the parent's __init__ is
empty (torch Dataset / Sampler / Lightning Callback) or attribute-only with
the subclass manually re-assigning every known attr (sklearn TransformedTargetRegressor).
Future-rot risk: any parent attribute added in a minor upstream release silently
drops from get_params/clone introspection.

5 P2 cosmetic fixes applied:

  1. estimators/custom.py:50 (ESTransformedTargetRegressor)
     Forward 5 known attrs via super().__init__(); any new sklearn attrs added
     in minor releases get populated automatically.

  2. training/neural/base.py:784 (AggregatingValidationCallback)
     super().__init__() before store_params_in_object so future Lightning
     Callback state is honoured.

  3. training/neural/data.py:62 (TorchDataset)
     super().__init__() for forward-compat with torch.utils.data.Dataset.

  4. training/neural/ranker.py:187 (GroupBatchSampler)
     super().__init__(data_source=None) for forward-compat with torch Sampler.

  5. training/neural/ranker.py:240 (_RankerDataset)
     Same as #3.

Per the audit, NONE of these cause runtime AttributeError today because each
parent's __init__ is currently empty/lazy; the fixes are pure forward-compat /
convention hardening.
"""

from __future__ import annotations

def _spy_on_base_init(monkeypatch, cls):
    """Replace the direct base class's ``__init__`` by a recorder; return the list of instances it was invoked on."""
    base = cls.__mro__[1]
    seen: list = []

    def spy(self, *args, **kwargs):
        """Record the instance whose base initialiser ran."""
        seen.append(self)

    monkeypatch.setattr(base, "__init__", spy, raising=False)
    return seen


def test_es_transformed_target_regressor_calls_super_init() -> None:
    """The parent initialiser populates every sklearn param, so get_params and clone round-trip them together with the early-stopping one."""
    from sklearn.base import clone
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from mlframe.estimators.custom import ESTransformedTargetRegressor

    est = ESTransformedTargetRegressor(Ridge(alpha=3.0), transformer=StandardScaler(), check_inverse=False, es_fit_param_name="eval_set")
    params = est.get_params(deep=False)
    assert params["check_inverse"] is False
    assert params["es_fit_param_name"] == "eval_set"
    assert params["func"] is None and params["inverse_func"] is None
    cloned = clone(est)
    assert cloned.regressor.alpha == 3.0
    assert cloned.es_fit_param_name == "eval_set"
    assert cloned.get_params(deep=False).keys() == params.keys()


def test_aggregating_validation_callback_calls_super_init(monkeypatch) -> None:
    """Constructing the callback runs the Lightning ``Callback`` base initialiser exactly once, on the new instance."""
    from mlframe.training.neural._base_callbacks import AggregatingValidationCallback

    seen = _spy_on_base_init(monkeypatch, AggregatingValidationCallback)
    cb = AggregatingValidationCallback("auc", lambda y, p: 0.0)
    assert seen == [cb]
    assert cb.metric_name == "auc" and cb.on_epoch is True


def test_torch_dataset_calls_super_init(monkeypatch) -> None:
    """Constructing a TorchDataset runs the torch ``Dataset`` base initialiser exactly once, on the new instance."""
    import numpy as np

    from mlframe.training.neural.data import TorchDataset

    seen = _spy_on_base_init(monkeypatch, TorchDataset)
    ds = TorchDataset(np.zeros((4, 2), dtype=np.float32), np.zeros(4, dtype=np.float32))
    assert seen == [ds]
    assert len(ds) == 4


def test_group_batch_sampler_calls_super_init(monkeypatch) -> None:
    """Constructing a GroupBatchSampler runs the torch ``Sampler`` base initialiser exactly once, with no extra arguments."""
    import numpy as np

    from mlframe.training.neural.ranker import GroupBatchSampler

    calls: list = []

    def spy(self, *args, **kwargs):
        """Record the instance and the arguments the base initialiser received."""
        calls.append((self, args, kwargs))

    monkeypatch.setattr(GroupBatchSampler.__mro__[1], "__init__", spy, raising=False)
    sampler = GroupBatchSampler(np.array([0, 0, 1, 1]), np.array([0.0, 1.0, 0.0, 1.0]), shuffle=False)
    assert calls == [(sampler, (), {})]


def test_ranker_dataset_calls_super_init(monkeypatch) -> None:
    """Constructing a _RankerDataset runs the torch ``Dataset`` base initialiser exactly once, on the new instance."""
    import numpy as np

    from mlframe.training.neural.ranker import _RankerDataset

    seen = _spy_on_base_init(monkeypatch, _RankerDataset)
    ds = _RankerDataset(np.zeros((3, 2), dtype=np.float32), np.zeros(3, dtype=np.float32))
    assert seen == [ds]
