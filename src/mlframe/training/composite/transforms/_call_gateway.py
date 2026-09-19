"""Single signature-gated entry point for calling a registry transform's ``fit`` / ``forward`` / ``inverse``.

Optional keyword arguments (``groups``, ``sample_weight``, recurrence ``history_*`` prefixes) are only accepted by some transforms. Call sites that
spelled out ``transform.fit(y, base)`` silently dropped them (a grouped transform refit without its groups, weights ignored by a chain), while a
blanket ``except TypeError`` retry mis-attributes a TypeError raised deep inside a fit to "parameter not supported". :func:`call_transform` passes
each optional argument exactly when the target callable declares it (or takes ``**kwargs``), and raises when a REQUIRED one (``groups`` for a
``requires_groups`` transform) is missing, so every call site gets the same forwarding rule.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

import numpy as np

_OPS = ("fit", "forward", "inverse")


@functools.lru_cache(maxsize=1024)
def _accepted_kwargs(fn: Callable[..., Any]) -> tuple[frozenset[str], bool]:
    """Declared keyword-capable parameter names of ``fn`` and whether it takes ``**kwargs`` (cached per callable)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return frozenset(), True
    names = frozenset(
        name for name, p in sig.parameters.items() if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )
    var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    return names, var_kw


def callable_accepts(fn: Callable[..., Any], name: str) -> bool:
    """True when ``fn`` declares a keyword-capable parameter ``name`` or accepts ``**kwargs``."""
    try:
        names, var_kw = _accepted_kwargs(fn)
    except TypeError:  # unhashable callable: inspect directly
        names, var_kw = _accepted_kwargs.__wrapped__(fn)
    return var_kw or name in names


def call_transform(transform: Any, op: str, *args: Any, **optional: Any) -> Any:
    """Call ``getattr(transform, op)(*args, **kwargs)`` where ``kwargs`` keeps only the non-``None`` optional arguments the callable accepts.

    ``op`` is one of ``fit`` / ``forward`` / ``inverse``; ``args`` are the positional ``(y, base)`` / ``(y, base, params)`` /
    ``(t_hat, base, params)``. A ``requires_groups`` transform called without ``groups`` raises ``ValueError`` instead of failing later inside
    the transform with a less specific message.
    """
    if op not in _OPS:
        raise ValueError(f"call_transform: op must be one of {_OPS}, got {op!r}")
    fn = getattr(transform, op)
    if getattr(transform, "requires_groups", False) and optional.get("groups") is None:
        raise ValueError(f"call_transform: transform {getattr(transform, 'name', transform)!r} requires groups for {op}.")
    kwargs = {k: v for k, v in optional.items() if v is not None and callable_accepts(fn, k)}
    if "groups" in kwargs:
        kwargs["groups"] = np.asarray(kwargs["groups"]).reshape(-1)
    return fn(*args, **kwargs)


__all__ = ["call_transform", "callable_accepts"]
