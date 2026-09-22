"""Does allocating the permutation-null matrix transposed actually win, once the MI call is included?

The fill writes a column of a C-order ``(n, n_perm)`` array per permutation, every element strided. Allocating ``(n_perm, n)`` and passing
``.T`` makes the fill contiguous, but hands the MI kernel a Fortran-ordered view, which may force an internal copy and give the win back.
Both halves are timed separately so the answer is attributable.
Result (this host, best-of, warm), measured 2026-09-22:

    n=200k  fill 0.0321 -> 0.0182  mi 0.1387 -> 0.1393  total 0.1708 -> 0.1574  (1.09x)
    n=2M    fill 0.6080 -> 0.4780  mi 1.3463 -> 1.4502  total 1.9544 -> 1.9282  (1.01x)

bench-attempt-rejected: the fill does get faster, 1.76x at 200k and 1.27x at 2M, but the MI kernel is ~8% SLOWER on the transposed view at 2M,
which is the Fortran-order cost this was written to check for. The two cancel, so the end-to-end saving is 1.01x at production scale: inside
noise, and not worth handing every downstream consumer a non-C-contiguous matrix. Values are identical either way.
"""
import sys
import time

sys.path.insert(0, "src")

import numpy as np

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

N_PERM = 12


def fill_c_order(feat, perms):
    """Current form: C-order (n, n_perm), one strided column write per permutation."""
    n = feat.shape[0]
    mat = np.empty((n, N_PERM), dtype=np.float64)
    for i in range(N_PERM):
        mat[:, i] = feat[perms[i]]
    return mat


def fill_transposed(feat, perms):
    """Proposed form: C-order (n_perm, n) filled by rows, handed on as a transposed view."""
    n = feat.shape[0]
    mat_t = np.empty((N_PERM, n), dtype=np.float64)
    for i in range(N_PERM):
        mat_t[i, :] = feat[perms[i]]
    return mat_t.T


def best(fn, args, repeat):
    """Best-of-repeat wall seconds, warmed once."""
    fn(*args)
    out = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        out = min(out, time.perf_counter() - t0)
    return out


def main():
    """Time the fill and the MI call for both layouts across n."""
    print(f"{'n':>9} {'fill C':>10} {'fill T':>10} {'mi C':>10} {'mi T':>10} {'total C':>10} {'total T':>10} {'equal':>6}")
    for n in (200_000, 2_000_000):
        rng = np.random.default_rng(0)
        feat = rng.normal(size=n)
        y = rng.integers(0, 3, size=n).astype(np.int64)
        perms = [rng.permutation(n) for _ in range(N_PERM)]
        repeat = 5 if n <= 200_000 else 3

        t_fill_c = best(fill_c_order, (feat, perms), repeat)
        t_fill_t = best(fill_transposed, (feat, perms), repeat)
        mat_c = fill_c_order(feat, perms)
        mat_t = fill_transposed(feat, perms)
        t_mi_c = best(lambda m, yy: _mi_classif_batch(m, yy), (mat_c, y), repeat)
        t_mi_t = best(lambda m, yy: _mi_classif_batch(m, yy), (mat_t, y), repeat)
        same = np.allclose(_mi_classif_batch(mat_c, y), _mi_classif_batch(mat_t, y), atol=1e-12)
        print(
            f"{n:>9} {t_fill_c:>10.4f} {t_fill_t:>10.4f} {t_mi_c:>10.4f} {t_mi_t:>10.4f} " f"{t_fill_c + t_mi_c:>10.4f} {t_fill_t + t_mi_t:>10.4f} {same!s:>6}"
        )


if __name__ == "__main__":
    main()
