import pickle, pandas as pd, numpy as np, io, bisect
lines = open("D:/Temp/fam_gpu_trace.csv", encoding="utf8", errors="ignore").read().splitlines()
h = next(i for i, l in enumerate(lines) if l.startswith('"Start"'))
df = pd.read_csv(io.StringIO("\n".join([lines[h]] + lines[h + 2:])), low_memory=False)
unit = lines[h + 1].split(",")[0].strip('"')  # unit of Start
mul = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit]
dur_unit = lines[h + 1].split(",")[1].strip('"'); dmul = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[dur_unit]
df["t"] = df["Start"].astype(float) * mul; df["d"] = df["Duration"].astype(float) * dmul
k = df[df["Name"].notna() & df["Grid X"].notna()].copy()   # kernels have grid dims
anc = k[k["Name"].str.contains("anchor_kernel_zz")]
d = pickle.load(open("D:/Temp/fam_gpu_log.pkl", "rb"))
off = anc["t"].iloc[-1] - d["t_anchor"]   # trace time = perf + off (anchor launch precedes t_anchor by tiny)
ivs = [(a + off, b + off, n) for n, a, b in d["log"]]
k = k[k["t"] > anc["t"].iloc[-1]]
ivs.sort(key=lambda r: r[0])
def own(t):
    best = None
    for a, b, n in ivs:
        if a <= t <= b and not n.startswith(("stage:", "gate:")):
            if best is None or (b - a) < best[0]: best = (b - a, n)
    if best: return best[1]
    for a, b, n in ivs:
        if a <= t <= b:
            if best is None or (b - a) < best[0]: best = (b - a, n)
    return best[1] if best else "(none)"
k["fam"] = [own(t) for t in k["t"]]
gpu = k.groupby("fam").agg(gpu_s=("d", "sum"), n_kern=("d", "size"))
wall = {}
for n, a, b in d["log"]: wall[n] = wall.get(n, 0.0) + (b - a)
res = pd.DataFrame({"wall_s": pd.Series(wall)}).join(gpu).fillna(0.0)
res["gpu_pct"] = 100 * res.gpu_s / res.wall_s.clip(lower=1e-9)
res = res.sort_values("wall_s", ascending=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(res.round(3).to_string())
res.to_csv("D:/Temp/fam_gpu_attr.csv")
