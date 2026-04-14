"""Reusable resource-logging primitives for mlframe transformers and estimators.

Provides three levels of API for emitting uniform wall-time + ΔRSS log records
around sklearn-compatible ``fit`` / ``transform`` / ``fit_transform`` calls:

1. :func:`log_resources` — a function decorator.
2. :func:`log_methods` — a class decorator that stamps ``log_resources`` onto
   named methods in place.
3. :func:`wrap_with_logging` — an instance factory that returns a thin proxy
   forwarding attribute access to the underlying object while instrumenting
   the requested methods.

Every emitted ``LogRecord`` carries a structured ``extra`` payload with
``stage``, ``cls``, ``dt_s``, ``rss_mb`` and ``d_rss_mb`` — so downstream
handlers (JSON logger, MLflow metric callback, etc.) can filter / route.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Iterable, Optional

import psutil

logger = logging.getLogger("mlframe.training.logging_transformers")


def log_resources(
    *,
    stage: Optional[str] = None,
    level: int = logging.INFO,
    extra_factory: Optional[Callable[..., dict]] = None,
):
    """Decorator logging wall-time and ΔRSS around the wrapped callable.

    Parameters
    ----------
    stage:
        Human label for the log record. Defaults to ``func.__qualname__``.
    level:
        Logging level (default ``logging.INFO``).
    extra_factory:
        Optional callable ``(self, *args, **kwargs) -> dict`` whose result is
        merged into the ``LogRecord.extra`` payload. Useful for adding
        dataset-shape metadata.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            proc = psutil.Process()
            rss0 = proc.memory_info().rss / 1024 ** 2
            t0 = time.perf_counter()
            try:
                return func(self, *args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                rss1 = proc.memory_info().rss / 1024 ** 2
                label = stage or func.__qualname__
                cls_name = type(self).__name__
                extra = {
                    "stage": label,
                    "cls": cls_name,
                    "dt_s": dt,
                    "rss_mb": rss1,
                    "d_rss_mb": rss1 - rss0,
                }
                if extra_factory is not None:
                    try:
                        extra.update(extra_factory(self, *args, **kwargs))
                    except Exception:  # pragma: no cover - defensive
                        logger.debug("extra_factory raised; ignoring", exc_info=True)
                logger.log(
                    level,
                    "[%s] %s: %.2fs, dRSS=%+.1f MB (now %.1f MB)",
                    label,
                    cls_name,
                    dt,
                    rss1 - rss0,
                    rss1,
                    extra=extra,
                )

        return wrapper

    return decorator


def log_methods(*method_names: str, stage_prefix: str = ""):
    """Class decorator applying :func:`log_resources` to named methods in place.

    Missing methods are silently skipped so the decorator is safe to use against
    classes that may or may not define every listed method (e.g. transformers
    without ``fit_transform``).
    """

    def decorator(cls):
        for name in method_names:
            orig = getattr(cls, name, None)
            if orig is None:
                continue
            stage = f"{stage_prefix}.{name}" if stage_prefix else name
            setattr(cls, name, log_resources(stage=stage)(orig))
        return cls

    return decorator


class _LoggingProxy:
    """Thin forwarding proxy emitted by :func:`wrap_with_logging`.

    Forwards unknown attribute access to the wrapped instance via
    ``__getattr__`` so sklearn helpers (``get_params``, ``set_params``,
    ``clone``) keep working transparently.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str):
        # __getattr__ is only invoked when normal lookup fails, so this safely
        # delegates everything not defined on the proxy itself.
        return getattr(self._inner, name)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"_LoggingProxy({self._inner!r})"


def wrap_with_logging(
    obj: Any,
    *,
    stage: Optional[str] = None,
    methods: Iterable[str] = ("fit", "transform", "fit_transform"),
) -> Any:
    """Return a proxy around ``obj`` that logs resources around ``methods``.

    Methods that don't exist on ``obj`` are skipped silently. Attribute access
    not covered by instrumentation is forwarded to the underlying instance.
    """

    label = stage or type(obj).__name__
    # Subclass per-call so instrumented methods don't leak across instances.
    ProxyCls = type("_LoggingProxy", (_LoggingProxy,), {})

    for name in methods:
        orig = getattr(obj, name, None)
        if orig is None:
            continue

        # Bind name/orig per-iteration via default args to avoid the
        # classic lambda-in-loop late-binding pitfall.
        def _make(_name: str):
            def _call(self, *a, **kw):
                return getattr(self._inner, _name)(*a, **kw)

            _call.__name__ = _name
            _call.__qualname__ = f"_LoggingProxy.{_name}"
            return _call

        bound = _make(name)
        wrapped = log_resources(stage=f"{label}.{name}")(bound)
        setattr(ProxyCls, name, wrapped)

    return ProxyCls(obj)


__all__ = [
    "log_resources",
    "log_methods",
    "wrap_with_logging",
]
