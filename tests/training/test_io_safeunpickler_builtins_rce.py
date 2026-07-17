"""``_SafeUnpickler`` must block code-exec builtins (eval/exec/compile/__import__/...) even though
``builtins`` is allowlisted for data containers. Pre-fix these payloads EXECUTED at load time:
``builtins`` was prefix-allowed wholesale, so a ``(eval, (src,))`` reduce ran ``src``.

``getattr`` is a special case: it is NOT blanket-blocked (legitimate model bundles need it -- CatBoost's
own ``__reduce__`` does ``getattr(<catboost module>, "_setattr")``), but it is routed through a restricted
reconstructor that refuses introspection-escalation attribute names (``__globals__`` / ``__code__`` /
``__subclasses__`` / ...). Its operand always already passed the module/class allowlist, so the classic
``getattr(os, "system")`` gadget is unreachable.
"""

from __future__ import annotations

import io
import os
import pickle

import dill
import pytest

from mlframe.training.io import _SafeUnpickler


class _EvilEval:
    def __reduce__(self):
        return (eval, ("__import__('os').getcwd()",))


class _EvilExec:
    def __reduce__(self):
        return (exec, ("_pwned = 1",))


class _EvilImport:
    def __reduce__(self):
        return (__import__, ("os",))


@pytest.mark.parametrize(
    "obj, name",
    [(_EvilEval(), "eval"), (_EvilExec(), "exec"), (_EvilImport(), "__import__")],
)
def test_code_exec_builtin_blocked(obj, name):
    raw = pickle.dumps(obj)
    with pytest.raises(dill.UnpicklingError, match=rf"builtins\.{name}"):
        _SafeUnpickler(io.BytesIO(raw)).load()


class _EvilGetattrGlobals:
    """getattr escalation: fetch ``__globals__`` off a function to reach its module dict, then any callable."""

    def __reduce__(self):
        return (getattr, (str.upper, "__globals__"))


def test_getattr_escalation_attr_blocked():
    # The restricted reconstructor must refuse dangerous introspection attribute names.
    raw = pickle.dumps(_EvilGetattrGlobals())
    with pytest.raises(dill.UnpicklingError, match=r"Unsafe getattr"):
        _SafeUnpickler(io.BytesIO(raw)).load()


class _BenignGetattr:
    """getattr fetching an ordinary attribute off an allowlisted operand is permitted (CatBoost-style)."""

    def __reduce__(self):
        return (getattr, ("ABC", "lower"))


def test_getattr_benign_attribute_allowed():
    # Operand is a plain str (allowlisted builtins data); ``lower`` is a harmless bound method.
    raw = pickle.dumps(_BenignGetattr())
    loaded = _SafeUnpickler(io.BytesIO(raw)).load()
    assert loaded() == "abc"


class _EvilSetattrDunder:
    """setattr type-confusion: rewrite ``__class__`` on a reconstructed object."""

    def __reduce__(self):
        return (setattr, (_BenignGetattr(), "__class__", int))


def test_setattr_dunder_blocked():
    raw = pickle.dumps(_EvilSetattrDunder())
    with pytest.raises(dill.UnpicklingError, match=r"Unsafe setattr"):
        _SafeUnpickler(io.BytesIO(raw)).load()


def test_legit_data_containers_still_load():
    # The denylist must not break ordinary builtins data structures a model bundle carries.
    payload = {"a": [1, 2, 3], "b": (4, 5), "c": {6, 7}, "d": b"xx", "e": None, "f": True, "g": frozenset({8})}
    raw = dill.dumps(payload)
    loaded = _SafeUnpickler(io.BytesIO(raw)).load()
    assert loaded == payload


def test_eval_payload_does_not_execute(tmp_path):
    # End-to-end proof the exec primitive cannot write a side-effect file through safe load.
    marker = tmp_path / "pwned.txt"
    src = f"open({str(marker)!r}, 'w').write('x')"

    class _Side:
        def __reduce__(self):
            return (exec, (src,))

    raw = pickle.dumps(_Side())
    with pytest.raises(dill.UnpicklingError):
        _SafeUnpickler(io.BytesIO(raw)).load()
    assert not os.path.exists(marker)
