"""iter133 bench — MissingAwareComposite fit/predict @10M (measured REJECT).

Surveys ``MissingAwareComposite`` (training/composite/missing.py) as a fresh full-n component:
its fit + predict both impute a NaN base column over n rows, so the question is whether any
plain-python / redundant-numpy work moves the e2e wall at 10M.

Findings (this hardware, store py3.14, single thread):
- The predict frame's own tottime is dominated by *fundamental* numpy ops: ``~np.isfinite(base)``
  (one full-n pass), the mandatory ``base.copy()`` inside ``_impute_inplace_safe`` (we must not
  mutate the caller's column), and the ``pred[...].copy()`` before the masked offset writes.
- The one structural redundancy — ``predict`` calls ``_extract_base`` (line 263) and then
  ``_impute_inplace_safe`` re-extracts the SAME column (line 134) — costs ~0us, because a polars
  float64 column -> numpy is ZERO-COPY (``to_numpy()`` + ``astype(copy=False)`` no-op). Measured:
    extract_base median ~0.002 ms   (zero-copy)
    base.copy()      median ~15.7 ms (unavoidable; we own the mutation)
  So eliminating the redundant extract saves nothing measurable; the cost is the copy, which is
  required for correctness. (For a pandas non-float64 base the re-extract WOULD cost an astype, but
  that is dwarfed by the same mandatory copy, and the prod carrier is polars float64.)

Verdict: REJECT — no clean, measurable e2e win. All costs are mandatory vectorized numpy.

Run:
    python -c "import sys; sys.modules['cupy']=None" ... (see __main__)
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.modules.setdefault("cupy", None)  # avoid cold-import segfault on py3.14


def main() -> None:
    import polars as pl
    from sklearn.base import BaseEstimator, RegressorMixin

    from mlframe.training.composite.estimator import _extract_base
    from mlframe.training.composite.missing import MissingAwareComposite

    n = 10_000_000
    rng = np.random.default_rng(0)
    base = rng.standard_normal(n).astype(np.float64)
    base[rng.random(n) < 0.2] = np.nan
    X = pl.DataFrame({"b": base, "f": rng.standard_normal(n)})
    y = np.where(np.isnan(base), 0.0, base) + rng.standard_normal(n) * 0.1

    # Isolate the (suspected-redundant) extract vs the mandatory copy.
    for _ in range(3):
        _extract_base(X, "b")
    ts = [(_t := time.perf_counter(), _extract_base(X, "b"), time.perf_counter() - _t)[2] for _ in range(7)]
    print(f"extract_base median ms {np.median(ts) * 1000:.4f}  (zero-copy for polars float64)")
    b = _extract_base(X, "b")
    ts = [(_t := time.perf_counter(), b.copy(), time.perf_counter() - _t)[2] for _ in range(7)]
    print(f"base.copy()   median ms {np.median(ts) * 1000:.4f}  (mandatory; wrapper owns the mutation)")

    class FakeComp(BaseEstimator, RegressorMixin):
        base_column = "b"

        def fit(self, X, y, **k):  # noqa: N803
            self.n_ = 1
            return self

        def predict(self, X):  # noqa: N803
            return np.zeros(X.shape[0])

    m = MissingAwareComposite(composite=FakeComp())
    m.fit(X, y)
    m.predict(X)  # warm
    ts = [(_t := time.perf_counter(), m.predict(X), time.perf_counter() - _t)[2] for _ in range(5)]
    print(f"predict median ms {np.median(ts) * 1000:.1f}  (dominated by isfinite + copies)")
    print("VERDICT: REJECT - all cost is mandatory vectorized numpy; redundant extract is zero-copy.")


if __name__ == "__main__":
    main()
