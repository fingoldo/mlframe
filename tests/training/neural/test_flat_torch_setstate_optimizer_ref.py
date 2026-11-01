"""Unpickling the flat torch module resolves only torch optimizer classes, never an arbitrary module attribute (DYN2-1).

The optimizer reference is part of the pickle stream, so resolving whatever module and attribute it names would run getattr on attacker-chosen
names. A reference outside ``torch.optim`` is now kept as the reference tuple.
"""

import pytest

from mlframe.training.neural._flat_torch_module import MLPTorchModel


def _restore(optimizer_ref):
    """Run the unpickle hook on a bare instance, with only the optimizer entry in the state."""
    obj = MLPTorchModel.__new__(MLPTorchModel)
    obj.__setstate__({"optimizer": optimizer_ref})
    return obj.__dict__["optimizer"]


def test_reference_outside_torch_optim_is_not_resolved():
    """A reference to a module attribute such as ``os.system`` is kept as the tuple and never looked up."""
    ref = ("__optimizer_class_ref__", "os", "system")
    assert _restore(ref) == ref


def test_torch_optimizer_reference_still_resolves_to_its_class():
    """The legitimate case, a torch optimizer class, is restored as the class object."""
    torch = pytest.importorskip("torch")
    resolved = _restore(("__optimizer_class_ref__", "torch.optim", "Adam"))
    assert resolved is torch.optim.Adam
