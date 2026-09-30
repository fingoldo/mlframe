"""Shared formatting for the one-line "what did this selector keep" INFO summaries emitted at the end of a selector fit."""
from __future__ import annotations

import functools
import inspect
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

MAX_LOGGED_NAMES = 30


def format_name_list(names: Iterable, max_names: int = MAX_LOGGED_NAMES) -> str:
    """Comma-join ``names``, keeping the first ``max_names`` and summarising the rest as ``... (+K more)`` so a 5000-column selection stays one readable line."""
    names = [str(n) for n in names]
    if len(names) <= max_names:
        return ", ".join(names)
    return ", ".join(names[:max_names]) + f", ... (+{len(names) - max_names} more)"


_nested = threading.local()


def log_selection(
    logger: logging.Logger,
    name: str,
    n_selected: int,
    n_total: int,
    names: Optional[Iterable] = None,
    *,
    elapsed: Optional[float] = None,
    extra: Optional[str] = None,
    what: str = "selected",
    respect_quiet: bool = False,
) -> None:
    """Emit the single end-of-fit INFO line ``<name>: <what> K of N features [(+extra)] [in T s]: [first 30 names]``.

    ``respect_quiet=True`` makes a selector that is itself fitted inside a wrapper's :func:`quiet_nested` scope (bootstrap / stability / hybrid
    inner fits) stay silent, so the wrapper's own line is the only one.

    Never raises: a reporting failure must not fail a long fit.
    """
    if respect_quiet and getattr(_nested, "depth", 0) > 0:
        return
    try:
        msg = f"{name}: {what} {int(n_selected):_} of {int(n_total):_} features"
        if extra:
            msg += f" ({extra})"
        if elapsed is not None:
            msg += f" in {float(elapsed):.1f} s"
        if names is not None:
            msg += f": [{format_name_list(names)}]"
        logger.info(msg)
    except Exception as exc:
        logger.debug("%s selection summary failed: %s", name, exc)


@contextmanager
def quiet_nested() -> Iterator[None]:
    """Silence :func:`logs_selection`-decorated functions called inside this scope, so a wrapper (adapter / cascade) emits ONE summary instead of one per layer."""
    _nested.depth = getattr(_nested, "depth", 0) + 1
    try:
        yield
    finally:
        _nested.depth -= 1


def is_quiet() -> bool:
    """True inside a :func:`quiet_nested` scope: a selector that logs its own summary should stay silent there."""
    return getattr(_nested, "depth", 0) > 0


def n_columns(X: Any) -> int:
    """Column count of a pandas / polars / ndarray-like ``X``."""
    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[1])
    return len(getattr(X, "columns", []))


# extractor(result, bound_args) -> (selected names, n_total, extra text or None)
Extractor = Callable[[Any, "inspect.BoundArguments"], Tuple[Iterable, int, Optional[str]]]


def summ_selected_of_x(res: Any, b: "inspect.BoundArguments") -> Tuple[Iterable, int, Optional[str]]:
    """Extractor for a selector returning ``selected`` or ``(selected, ...)`` and taking the frame as its ``X`` argument."""
    sel = res[0] if isinstance(res, tuple) else res
    return sel, n_columns(b.arguments["X"]), None


def logs_selection(name: str, extract: Extractor, *, what: str = "selected") -> Callable[[F], F]:
    """Decorator: after the wrapped selection function returns, emit one INFO summary via :func:`log_selection`.

    Skipped when called inside :func:`quiet_nested`; the wrapped body itself runs inside that scope so functions it calls stay quiet too.
    """

    def deco(fn: F) -> F:
        sig = inspect.signature(fn)
        logger = logging.getLogger(fn.__module__)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            outer_quiet = getattr(_nested, "depth", 0) > 0
            t0 = time.perf_counter()
            with quiet_nested():
                result = fn(*args, **kwargs)
            if not outer_quiet:
                try:
                    selected, n_total, extra = extract(result, sig.bind(*args, **kwargs))
                    selected = list(selected)
                    log_selection(logger, name, len(selected), n_total, selected, elapsed=time.perf_counter() - t0, extra=extra, what=what)
                except Exception as exc:
                    logger.debug("%s selection summary failed: %s", name, exc)
            return result

        return wrapper  # type: ignore[return-value]

    return deco


def quiet_fit(fn: F) -> F:
    """Decorator for a selector ``fit`` that logs its own summary: functions it delegates to stay silent (see :func:`quiet_nested`)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with quiet_nested():
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def logs_fit(name: str, describe: Callable[[Any], Tuple[Any, ...]]) -> Callable[[F], F]:
    """Method decorator for a selector ``fit``: runs it inside :func:`quiet_nested`, then emits one summary from ``describe(self)``.

    ``describe`` returns ``(names, n_total, extra)`` or ``(names, n_total, extra, n_selected)`` when the count differs from ``len(names)`` (e.g. names include engineered columns).

    Nested fits (stability subsamples, hybrid members, ...) stay silent; the outermost selector logs once, and only if it is not itself inside a quiet scope.
    """

    def deco(fn: F) -> F:
        logger = logging.getLogger(fn.__module__)

        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            with quiet_nested():
                result = fn(self, *args, **kwargs)
            try:
                selected, n_total, extra, *rest = describe(self)
                selected = list(selected)
                n_selected = rest[0] if rest else len(selected)
                log_selection(logger, name, n_selected, n_total, selected, elapsed=time.perf_counter() - t0, extra=extra, respect_quiet=True)
            except Exception as exc:
                logger.debug("%s selection summary failed: %s", name, exc)
            return result

        return wrapper  # type: ignore[return-value]

    return deco
