"""H2: O(p) interaction-propensity ranking recall of planted PURE-pair operands."""
import sys, time
import numpy as np

sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe/src")

PROG = r"D:/Temp/synergy_scale_bench/progress.txt"
def ck(msg):
    with open(PROG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " | " + msg + "\n")
    print(msg, flush=True)

N = 8000
NBINS = 8
P = 2000
K = 6              # planted pure pair interactions -> 12 operand cols
SEEDS = [0, 1, 2, 3, 4]
LEAKS = [0.0, 0.1, 0.3]
MS = [100, 250, 500]

try:
    import dcor as _dcor
    HAS_DCOR = True
except Exception:
    HAS_DCOR = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
ck("H2 start dcor=%s lgb=%s" % (HAS_DCOR, HAS_LGB))


def make_frame(p, seed, L):
    """K pure pair interactions. Operands ~0 marginal signal; y = XOR-like sign product.
    L leaks main-effect/asymmetry into operands. Returns X (n,p float), y binary, operand idx set."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, p))
    operands = []
    logit = np.zeros(N)
    # pick 2K distinct operand columns spread across the frame
    cols = rng.choice(p, size=2 * K, replace=False)
    for k in range(K):
        ia, ib = cols[2 * k], cols[2 * k + 1]
        operands += [int(ia), int(ib)]
        a, b = X[:, ia], X[:, ib]
        # pure interaction: product of signs (zero marginal in expectation)
        logit += 1.6 * np.sign(a) * np.sign(b)
        # leakage: add a main effect proportional to L on each operand
        logit += L * 1.6 * (a + b)
    p_y = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(N) < p_y).astype(np.int32)
    return X, y, set(operands)


def bin_codes(x):
    q = np.quantile(x, np.linspace(0, 1, NBINS + 1)[1:-1])
    return np.searchsorted(q, x)


# ---- criteria, each returns score array length p (higher = more interesting) ----

def crit_marginal_mi(X, y):
    n, p = X.shape
    fy = np.bincount(y, minlength=2) / n
    scores = np.empty(p)
    for j in range(p):
        c = bin_codes(X[:, j])
        mi = 0.0
        for vx in range(NBINS):
            mask = c == vx
            nx = mask.sum()
            if nx == 0:
                continue
            px = nx / n
            yy = y[mask]
            for vy in range(2):
                jc = (yy == vy).sum()
                if jc == 0:
                    continue
                jf = jc / n
                if fy[vy] > 0:
                    mi += jf * np.log(jf / (px * fy[vy]))
        scores[j] = mi
    return scores

def crit_dcor(X, y, sample=2000):
    n, p = X.shape
    idx = np.random.default_rng(0).choice(n, size=min(sample, n), replace=False)
    ys = y[idx].astype(float)
    scores = np.empty(p)
    for j in range(p):
        scores[j] = _dcor.distance_correlation(X[idx, j].astype(float), ys)
    return scores

def crit_second_moment(X, y):
    yf = y.astype(float)
    y2 = yf * yf
    p = X.shape[1]
    scores = np.empty(p)
    for j in range(p):
        x = X[:, j]
        x2 = x * x
        c1 = abs(np.corrcoef(x2, yf)[0, 1])
        c2 = abs(np.corrcoef(x, y2)[0, 1])
        scores[j] = (0 if np.isnan(c1) else c1) + (0 if np.isnan(c2) else c2)
    return scores

def crit_cond_resp_var(X, y):
    yf = y.astype(float)
    p = X.shape[1]
    scores = np.empty(p)
    for j in range(p):
        c = bin_codes(X[:, j])
        means = np.array([yf[c == b].mean() if (c == b).any() else 0.0 for b in range(NBINS)])
        scores[j] = means.var()
    return scores

def crit_gbm_splits(X, y):
    p = X.shape[1]
    ds = lgb.Dataset(X, label=y)
    params = dict(objective="binary", num_leaves=31, learning_rate=0.1,
                  verbose=-1, min_child_samples=20, feature_fraction=1.0)
    booster = lgb.train(params, ds, num_boost_round=100)
    imp = booster.feature_importance(importance_type="split")
    return imp.astype(float)


CRITS = [("marginal_MI", crit_marginal_mi),
         ("2nd_moment", crit_second_moment),
         ("cond_resp_var", crit_cond_resp_var)]
if HAS_DCOR:
    CRITS.append(("dcor", crit_dcor))
if HAS_LGB:
    CRITS.append(("gbm_splits", crit_gbm_splits))


def recall(scores, operands, m):
    top = set(np.argsort(scores)[::-1][:m].tolist())
    return len(top & operands) / len(operands)

# recall_table[crit][L][m] = list over seeds
table = {name: {L: {m: [] for m in MS} for L in LEAKS} for name, _ in CRITS}
# random baseline expected recall = m/p
for L in LEAKS:
    for seed in SEEDS:
        X, y, operands = make_frame(P, seed, L)
        for name, fn in CRITS:
            t0 = time.perf_counter()
            sc = fn(X, y)
            dt = time.perf_counter() - t0
            for m in MS:
                table[name][L][m].append(recall(sc, operands, m))
        ck("L=%.1f seed=%d done (lastcrit %.2fs)" % (L, seed, dt))

ck("=== H2 RECALL TABLE (mean over %d seeds; random baseline m/p) ===" % len(SEEDS))
hdr = "%-15s" % "criterion"
for L in LEAKS:
    for m in MS:
        hdr += " L%.1f/m%-4d" % (L, m)
print(hdr)
for name, _ in CRITS:
    row = "%-15s" % name
    for L in LEAKS:
        for m in MS:
            row += " %9.2f" % np.mean(table[name][L][m])
    print(row)
print("%-15s" % "random_base" + "".join(" %9.2f" % (m / P) for L in LEAKS for m in MS))

# ---- timing at p=2000 and p=10000 ----
ck("=== H2 TIMING ===")
for p in (2000, 10000):
    X, y, _ = make_frame(p, 0, 0.1)
    for name, fn in CRITS:
        t0 = time.perf_counter()
        fn(X, y)
        ck("p=%d %s: %.3fs" % (p, name, time.perf_counter() - t0))

ck("H2 done")
