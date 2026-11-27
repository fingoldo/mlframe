"""Run the tt_b5_invariance jobs one MRMR fit at a time: wait until no other tt_b5_invariance.py process is alive, then run the queue sequentially.
    python tt_b5_queue.py OUT.jsonl LOGFILE "steps:group:seeds" ...
"""
import logging
import subprocess, sys, time, os
import psutil

out, log, jobs = sys.argv[1], sys.argv[2], sys.argv[3:]
here = os.path.dirname(os.path.abspath(__file__))


def others():
    me = os.getpid()
    for pr in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            cl = " ".join(pr.info["cmdline"] or [])
        except Exception as exc:
            logging.getLogger(__name__).debug("tt_b5_queue: ignored %s: %s", type(exc).__name__, exc)
            continue
        if "python" not in (pr.info["name"] or "").lower():
            continue  # skip nohup / sh wrappers, which carry the same command line
        if ("tt_b5_invariance.py" in cl or "tt_b5_queue.py" in cl or "tt_b5_mrmr_baseline.py" in cl) and pr.info["pid"] != me and "--wait-ignore" not in cl:
            return True
    return False


while others():
    time.sleep(10)
with open(log, "a", encoding="utf-8") as lf:
    for j in jobs:
        steps, group, seeds = j.split(":")[:3]
        tag = j.split(":")[3] if len(j.split(":")) > 3 else ""
        nj = len(j.split(":")) > 4 and j.split(":")[4] == "nj"
        sch = j.split(":")[5].replace("+", ",") if len(j.split(":")) > 5 else "quantile,uniform,mdlp"
        lf.write(f"## START {j} {time.strftime('%H:%M:%S')}\n"); lf.flush()
        subprocess.run([sys.executable, os.path.join(here, "tt_b5_invariance.py"), out, "--steps", steps, "--group", group, "--seeds", seeds, "--tag", tag] + (["--no-jitter"] if nj else []) + ["--schemes", sch], stdout=lf, stderr=subprocess.STDOUT)
        lf.write(f"## END {j} {time.strftime('%H:%M:%S')}\n"); lf.flush()
