"""A zero sample weight means the row is absent: transform fits drop such rows before fitting.

Weighted fits used the weights in the solve but counted zero-weight rows everywhere else: group sizes and the James-Stein
shrinkage of the grouped linear residual, rank / copula grids, category counts of the target encoding, trimming fractions.
So a fit with half the weights at zero differed from a fit on the kept half (linear_residual_grouped by 8.8 in y). Every
consumer that passes weights means "these rows count this much"; a weight of zero must not move the fit at all.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np


class ZeroWeightDroppingFit:
    """Wraps a transform ``fit(y, base, ..., sample_weight=..., groups=...)``; rows with weight 0 are dropped first.

    A module-level class (not a closure) so a Transform holding it still pickles when its ``fit`` does. When every weight is
    zero nothing is dropped (the fit then sees the weights as given). Positional ``y`` / ``base`` and a length-n ``groups``
    are row-sliced together.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.__wrapped__ = fn
        self.__name__ = getattr(fn, "__name__", "fit")
        self.__doc__ = getattr(fn, "__doc__", None)

    @property
    def __signature__(self) -> inspect.Signature:
        return inspect.signature(self.fn)

    def __call__(self, y: Any, base: Any = None, *args: Any, **kwargs: Any) -> Any:
        sw = kwargs.get("sample_weight")
        if sw is None:
            return self.fn(y, base, *args, **kwargs)
        w = np.asarray(sw, dtype=np.float64).reshape(-1)
        keep = w != 0
        if keep.all() or not keep.any():
            return self.fn(y, base, *args, **kwargs)
        y_arr = np.asarray(y)
        kwargs = dict(kwargs, sample_weight=w[keep])
        g = kwargs.get("groups")
        if g is not None and np.asarray(g).shape[0] == w.shape[0]:
            kwargs["groups"] = np.asarray(g)[keep]
        base_kept = None if base is None else np.asarray(base)[keep]
        return self.fn(y_arr[keep], base_kept, *args, **kwargs)

    def __reduce__(self):
        return (ZeroWeightDroppingFit, (self.fn,))


def with_zero_weight_drop(fn: Callable[..., Any], *, recurrent: bool) -> Callable[..., Any]:
    """``fn`` wrapped when it takes ``sample_weight`` and the transform is order-free (a recurrent fit reads neighbours)."""
    if recurrent or isinstance(fn, ZeroWeightDroppingFit):
        return fn
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn
    return ZeroWeightDroppingFit(fn) if "sample_weight" in params else fn
