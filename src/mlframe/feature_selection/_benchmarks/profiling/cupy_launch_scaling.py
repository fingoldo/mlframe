import threading, time, numpy as np, cupy as cp
x = cp.random.rand(20000).astype(cp.float32)
ek = cp.ElementwiseKernel("float32 x", "float32 y", "y = sqrtf(x * 1.0001f + 0.1f)", "fused_sq")
rk = cp.RawKernel(r'extern "C" __global__ void f(const float* x, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=sqrtf(x[i]*1.0001f+0.1f);}', "f")
out = cp.empty_like(x)
def w_ek(n):
    for _ in range(n): ek(x, out)
    cp.cuda.Stream.null.synchronize()
def w_rk(n):
    for _ in range(n): rk((79,), (256,), (x, out, np.int32(20000)))  # launch-ok: this microbenchmark measures LAUNCH SCALING itself (overhead of many small launches); the fixed geometry is what is being measured, not a kernel to tune
    cp.cuda.Stream.null.synchronize()
def w_ops(n):
    y = x
    for _ in range(n): y = cp.sqrt(y * 1.0001 + 0.1)
    cp.cuda.Stream.null.synchronize()
def run(fn, nthreads, n_iter, per):
    ts = [threading.Thread(target=fn, args=(n_iter,)) for _ in range(nthreads)]
    t = time.perf_counter()
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    return nthreads * n_iter * per / (time.perf_counter() - t)
for name, fn, per in (("3 cupy ops", w_ops, 3), ("1 ElementwiseKernel", w_ek, 1), ("1 RawKernel", w_rk, 1)):
    fn(200)
    a = run(fn, 1, 4000, per); b = run(fn, 3, 4000, per)
    print("%-20s 1 thread %7.0f launches/s (%.1f us each) | 3 threads %7.0f (x%.2f)" % (name, a, 1e6 / a, b, b / a))
