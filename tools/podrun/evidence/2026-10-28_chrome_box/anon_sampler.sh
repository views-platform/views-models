#!/usr/bin/env bash
# memory.current includes page cache; `anon` is the allocation that actually has to fit.
while :; do
  A=$(awk "/^anon /{print \$2}" /sys/fs/cgroup/memory.stat)
  F=$(awk "/^file /{print \$2}" /sys/fs/cgroup/memory.stat)
  echo "$(date +%H:%M:%S) anon=$(( A/1048576 ))MiB file=$(( F/1048576 ))MiB stage=$(cat /workspace/deliver/chrome_box/STAGE 2>/dev/null)" >> /workspace/deliver/chrome_box/anon.log
  sleep 10
done
