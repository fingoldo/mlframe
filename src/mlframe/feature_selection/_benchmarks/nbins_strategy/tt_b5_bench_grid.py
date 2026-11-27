"""Run the tt_b5_bench.py cells one process at a time and append the JSON lines to OUT.  python tt_b5_bench_grid.py OUT.jsonl [criteria|baselines|all]"""
import logging
import json, os, subprocess, sys, time

out = sys.argv[1]
which = sys.argv[2] if len(sys.argv) > 2 else "all"
here = os.path.dirname(os.path.abspath(__file__))
done = set()
if os.path.exists(out):
    for l in open(out, encoding="utf-8"):
        try:
            r = json.loads(l); done.add((r["kind"], r["n"], r["p"], r["variant"], r["K"], r["B"]))
        except Exception as exc:
            logging.getLogger(__name__).debug("tt_b5_bench_grid: ignored %s: %s", type(exc).__name__, exc)
            pass
cells = []
NS, PS = (20000, 100000, 1000000), (20, 200, 1000)
if which in ("criteria", "all"):
    for n in NS:
        for p in PS:
            tv = ["full"] if p <= 64 else ["cap64", "full"]
            for v in tv:
                cells.append(("target", n, p, v, 9, 40))
            for v in ("obs_only", "null_sub50k", "null_all"):
                if v == "null_sub50k" and n <= 50000:
                    continue
                if (n, p) == (1000000, 1000):
                    continue  # the numba-kernel version would run ~15 min per cell under the current CPU contention; extrapolated from n=1e5, the numpy version is measured
                if (n, p) == (1000000, 1000):
                    continue  # the numba-kernel version would run ~15 min per cell under the current CPU contention; extrapolated from n=1e5, the numpy version is measured
                cells.append(("feature", n, p, v, 9, 40))
if which == "np":
    for n in NS:
        for p in PS:
            for v in ("obs_only", "null_sub50k", "null_all"):
                if v == "null_sub50k" and n <= 50000:
                    continue
                cells.append(("feature_np", n, p, v, 9, 40))
if which in ("baselines", "all"):
    for n, p in [(20000, 20), (20000, 200), (100000, 20), (100000, 200), (1000000, 20), (20000, 1000), (100000, 1000), (1000000, 200)]:
        cells.append(("ridge", n, p, "", 9, 40))
    for n, p in [(20000, 20), (20000, 200), (100000, 20), (100000, 200), (1000000, 20), (20000, 1000), (100000, 1000), (1000000, 200)]:
        cells.append(("hgb", n, p, "100", 9, 40))
for c in cells:
    key = (c[0], c[1], c[2], c[3], c[4], c[5])
    if key in done:
        continue
    t = time.time()
    try:
        pr = subprocess.run([sys.executable, os.path.join(here, "tt_b5_bench.py"), c[0], str(c[1]), str(c[2]), c[3], str(c[4]), str(c[5])], capture_output=True, text=True, timeout=1500)
        line = [l for l in pr.stdout.splitlines() if l.startswith("{")]
        if line:
            open(out, "a", encoding="utf-8").write(line[-1] + "\n")
            print("done", key, round(time.time() - t, 1), flush=True)
        else:
            print("FAILED", key, pr.stderr[-400:], flush=True)
    except subprocess.TimeoutExpired:
        open(out, "a", encoding="utf-8").write(json.dumps(dict(kind=c[0], n=c[1], p=c[2], variant=c[3], K=c[4], B=c[5], timeout_s=1500)) + "\n")
        print("TIMEOUT", key, flush=True)
