"""Per-NVTX-range kernel launches and GPU time from an ``nvprof --print-gpu-summary`` log. Usage: ``python nvprof_range_summary.py out.txt``."""
import re, sys
txt = open(sys.argv[1], encoding="utf-8", errors="ignore").read().splitlines()
def tms(s):
    m = re.match(r"([\d.]+)(ns|us|ms|s)$", s); return float(m.group(1)) * {"ns": 1e-6, "us": 1e-3, "ms": 1, "s": 1000}[m.group(2)]
cur = None; res = {}
for l in txt:
    m = re.search(r'Range "([^"]+)"', l)
    if m: cur = m.group(1); res.setdefault(cur, [0, 0.0, 0, 0.0]); continue
    if cur is None: continue
    l2 = l.replace("GPU activities:", "").strip()
    m = re.match(r"([\d.]+)%\s+(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)", l2)
    if m and "memcpy" not in m.group(7) and "memset" not in m.group(7):
        res[cur][0] += int(m.group(3)); res[cur][1] += tms(m.group(2))
    elif m:
        res[cur][2] += int(m.group(3)); res[cur][3] += tms(m.group(2))
for k, (n, t, mc, mt) in sorted(res.items(), key=lambda x: -x[1][0]):
    if k.startswith("S_") or n > 400: print(f"{k:28s} kernels {n:6d}  gpu {t:8.1f} ms   memcpy {mc:5d} ({mt:6.1f} ms)")
