"""Dead-knob pins for PreprocessingBackendConfig.

``fallback_to_sklearn`` is declared and documented on PreprocessingBackendConfig
but is never READ anywhere under ``src/mlframe`` -- the only occurrences are its
own field definition and docstring. The advertised behaviour (fall back to
sklearn when polars-ds lacks an op) is not wired, so flipping it changes nothing.

The strict-xfail below pins that no-op: it asserts the (non-existent) wired
behaviour and is expected to fail. If a future change actually consumes the
field, the xfail flips to XPASS (strict) and fails the suite -- the signal to
delete this pin and write a real biz_value test for the now-live knob.
"""
from __future__ import annotations

import pathlib

import pytest

import mlframe


def _src_root() -> pathlib.Path:
    return pathlib.Path(mlframe.__file__).resolve().parent


def test_fallback_to_sklearn_is_never_consumed():
    """Sensor: grep every src file for ``fallback_to_sklearn`` READ sites
    (anything other than its own definition / docstring). Zero consumers ==
    dead knob. This passes today and starts FAILING the moment someone wires it,
    prompting removal of the dead-knob disposition.
    """
    root = _src_root()
    config_file = root / "training" / "_preprocessing_configs.py"
    consumers = []
    for py in root.rglob("*.py"):
        if py == config_file:
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "fallback_to_sklearn" in text:
            consumers.append(str(py))
    assert consumers == [], (
        "fallback_to_sklearn is now consumed at %r -- it is no longer a dead knob. "
        "Delete this pin and add a real biz_value test for the wired behaviour." % (consumers,)
    )


@pytest.mark.xfail(strict=True, reason="fallback_to_sklearn is a DEAD knob: declared but never read in src; no behaviour to assert")
def test_biz_val_backend_fallback_to_sklearn_changes_behaviour():
    """Expected-fail pin. There is no code path that branches on
    ``fallback_to_sklearn``, so no synthetic can make True vs False diverge.
    Kept as a tripwire: if the field becomes live, this XPASSes (strict) and
    fails -- replace with a genuine biz_value test then.
    """
    raise AssertionError("fallback_to_sklearn has no wired effect to validate")
