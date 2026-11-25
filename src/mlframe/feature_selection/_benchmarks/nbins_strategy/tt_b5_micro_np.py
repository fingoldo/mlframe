"""ns per element of each numpy float32 step used by tt_b5_cost_np._block (one block of 16 x 2048 float32, min of 30 repeats, wall)."""
import time
import numpy as np
rng = np.random.default_rng(0)
xb = rng.standard_normal((16, 2048)).astype(np.float32)
mu = np.zeros((16, 1), np.float32); isd = np.ones((16, 1), np.float32); eps = np.full((16, 1), 1e-3, np.float32)
out = np.empty_like(xb)
N = xb.size
ax = np.abs(xb)
def t(label, f):
    f(); best = 1e9
    for _ in range(30):
        t0 = time.perf_counter(); f(); best = min(best, time.perf_counter() - t0)
    print(f"{label:34s} {best / N * 1e9:6.2f} ns/elem", flush=True)
t("abs", lambda: np.abs(xb))
t("log(ax+eps)", lambda: np.log(ax + eps, out=out))
t("sqrt", lambda: np.sqrt(ax, out=out))
t("copysign", lambda: np.copysign(ax, xb, out=out))
t("x*ax", lambda: np.multiply(xb, ax, out=out))
t("log+*1/3+exp (cbrt)", lambda: np.exp(np.log(ax + np.float32(1e-30)) * np.float32(1 / 3), out=out))
t("np.cbrt (scalar libm)", lambda: np.cbrt(xb, out=out))
t("tanh", lambda: np.tanh(xb, out=out))
t("exp(clip)", lambda: np.exp(np.clip(xb, -8.0, 8.0), out=out))
t("1/(ax+eps)", lambda: np.float32(1.0) / (ax + eps))
Zv = np.empty((16, 9, 2048), np.float32)
t("einsum sum jkr->jk (float64 acc)", lambda: np.einsum("jkr->jk", Zv, dtype=np.float64))
t("einsum jkr,jkr->jk", lambda: np.einsum("jkr,jkr->jk", Zv, Zv, dtype=np.float64))
t("einsum jkr,jr->jk", lambda: np.einsum("jkr,jr->jk", Zv, xb, dtype=np.float64))
R = rng.standard_normal((2048, 41)).astype(np.float32)
Zt = Zv.reshape(16 * 9, 2048)
t("sgemm Zt@R (B+1 = 41)  per z-element", lambda: Zt @ R)
