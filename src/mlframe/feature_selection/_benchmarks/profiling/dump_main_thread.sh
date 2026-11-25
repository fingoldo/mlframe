#!/bin/bash
# usage: dump_main_thread.sh PID OUTFILE [PERIOD_S]
# One line per sample until the process ends: the epoch seconds, then the mlframe frames of the MainThread (innermost first) joined by ' > '.
# Read it with stage_dump_profile.py together with the stage ledger of the same run (BENCH_LEDGER of rss_scale_fit.py). Needs py-spy on the PATH; a dump costs the process a few milliseconds.
PID=$1
OUT=$2
PERIOD=${3:-2}
while tasklist //FI "PID eq $PID" 2>/dev/null | grep -q "$PID"; do
  NOW=$(date +%s.%N)
  STACK=$(py-spy dump --pid "$PID" 2>/dev/null | awk '/MainThread/{f=1;next} /^Thread /{f=0} f&&/mlframe/{sub(/^[ \t]+/,""); gsub(/\(mlframe\\feature_selection\\filters\\/,"("); printf "%s > ", $0}')
  echo "$NOW $STACK" >> "$OUT"
  sleep "$PERIOD"
done
echo "$(date +%s.%N) ended" >> "$OUT"
