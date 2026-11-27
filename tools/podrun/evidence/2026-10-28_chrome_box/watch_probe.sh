#!/bin/bash
# Watches the probe run. Deliberately does NOT use pgrep: a pgrep pattern matching
# the launched command also matches the watcher ssh command line (bit me twice today).
# Liveness is probe.out mtime; termination is a marker in probe.out.
LOG=/workspace/probe.out
PEAK=0
for i in $(seq 1 180); do
  A=$(awk "/^anon /{printf \"%d\", \$2/1048576}" /sys/fs/cgroup/memory.stat)
  [ "$A" -gt "$PEAK" ] && PEAK=$A
  AGE=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  if grep -qE "PipelineException|Traceback \(most recent" "$LOG"; then
    echo "[$(date +%H:%M:%S)] TERMINAL: traceback in log"; break
  fi
  if grep -q "Ensemble forecasting completed\|Forecasting completed" "$LOG"; then
    echo "[$(date +%H:%M:%S)] TERMINAL: forecasting completed"; break
  fi
  if [ "$AGE" -gt 900 ]; then
    echo "[$(date +%H:%M:%S)] TERMINAL: log stale ${AGE}s"; break
  fi
  if [ $((i % 6)) -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] alive(age=${AGE}s) anon=${A}MiB peak=${PEAK}MiB"
  fi
  sleep 30
done
echo "=== PEAK anon: ${PEAK} MiB ($((PEAK/1024)) GiB) ; unpatched pool reached 27.8 GiB ==="
echo "=== probe markers ==="
grep "\[probe\]" "$LOG" | tail -20
echo "=== reconcile / target progress ==="
grep -iE "reconcil|target .*lr_|aggregat" "$LOG" | tail -20
echo "=== tail ==="
tail -25 "$LOG"
echo "=== output written? ==="
ls -d /workspace/views-models/ensembles/chrome_box/data/generated/predictions_forecasting_* 2>/dev/null || echo "(none)"
