"""Shared base-optimizer-delegation members for the ``training/neural`` optimizer wrappers
(``SAM``, ``Lookahead``): both wrap a ``self.base_optimizer`` and were independently
duplicating identical ``param_groups`` / ``state`` / ``zero_grad`` delegation, consolidated
here so a fix can't silently drift out of sync across copies. Bind in a class body as
``param_groups = wrapper_param_groups`` etc. -- these are plain descriptors/functions, not
bound methods, so a class-body assignment (not a post-class-definition attribute set) is
required for ``self``/``property`` semantics to resolve correctly.
"""
from __future__ import annotations

from typing import Any, List


def _get_param_groups(self: Any) -> List[dict]:
    """Delegate to the wrapped base optimizer's ``param_groups`` so schedulers/logging that read it see the real groups."""
    return list(self.base_optimizer.param_groups)


def _set_param_groups(self: Any, value: List[dict]) -> None:
    """Forward a ``param_groups`` assignment to the wrapped base optimizer."""
    self.base_optimizer.param_groups = value


wrapper_param_groups = property(_get_param_groups, _set_param_groups)


def _get_state(self: Any) -> dict:
    """Delegate to the wrapped base optimizer's ``state`` (per-param optimizer state such as momentum buffers)."""
    return dict(self.base_optimizer.state)


def _set_state(self: Any, value: dict) -> None:
    """Forward a ``state`` assignment to the wrapped base optimizer."""
    self.base_optimizer.state = value


wrapper_state = property(_get_state, _set_state)


def wrapper_zero_grad(self: Any, set_to_none: bool = True) -> None:
    """Forward ``zero_grad`` to the wrapped base optimizer."""
    self.base_optimizer.zero_grad(set_to_none=set_to_none)
