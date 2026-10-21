#!/usr/bin/env bash
# Prove each verify_local guard FIRES. A guard never seen to fail is decorative.
set -u
# Resolve our own directory and work from it: the suite sources ./watch_pod.sh, so running it as
# `bash scratchpad/test_watch_pod_guards.sh` from the repo root failed all 11 checks on a missing
# file. A suite that reports FAIL because it cannot find its subject is the absence-is-not-a-pass
# defect inverted — it must not depend on the caller's cwd.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
[ -f ./watch_pod.sh ] || { echo "⛔ cannot find ./watch_pod.sh next to this suite — refusing to report"; exit 2; }
WATCH_POD_LIB=1
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
result() { if [ "$2" = "$3" ]; then echo "  PASS  $1 (got $3)"; pass=$((pass+1));
           else echo "  FAIL  $1 (expected $2, got $3)"; fail=$((fail+1)); fi; }

setup() {   # build a clean, valid retrieved-artifact tree from the real bronze files
  rm -rf "$TMP/m"; A="$TMP/m/data/generated/pod_delivery/artifact"; mkdir -p "$A"
  SRC=/home/simon/Documents/scripts/views_platform/views-models/models/bronze_kestrel/data/generated/pod_delivery/artifact
  cp "$SRC"/calibration_model_*.pt "$SRC"/calibration_model_*.pt.config.json "$SRC"/calibration_model_*.pt.sha256 "$A/"
  mkdir -p "$TMP/m/data/generated/preds"
  python3 -c "
import numpy as np
np.save('$TMP/m/data/generated/preds/good.npy', np.arange(1000, dtype='float32'))"
}
run() { M=m IP=x PT=1 EXPECTED_VER=0.0.0 L=$TMP D=$TMP/m LOG=$TMP/log ERRLOG=$TMP/err \
        bash -c 'WATCH_POD_LIB=1; source ./watch_pod.sh m x 1 0.0.0; M=m D='"$TMP"'/m LOG='"$TMP"'/log; verify_local >/dev/null 2>&1; echo $?'; }

echo "=== baseline: a complete, valid retrieval ==="
setup; result "intact tree verifies" 0 "$(run)"

echo "=== M1: sidecar removed (load_vehicle would refuse — #549) ==="
setup; rm "$TMP/m/data/generated/pod_delivery/artifact"/*.pt.config.json
result "missing sidecar caught" 1 "$(run)"

echo "=== M2: .pt corrupted, sha256 no longer matches ==="
setup
pt=$(ls "$TMP/m/data/generated/pod_delivery/artifact"/calibration_model_*.pt | head -1)
before=$(sha256sum "$pt" | awk '{print $1}')
printf 'x' | dd of="$pt" bs=1 seek=500 conv=notrunc 2>/dev/null
after=$(sha256sum "$pt" | awk '{print $1}')
if [ "$before" = "$after" ]; then
  echo "  ⛔ MUTATION DID NOT APPLY — test is broken, not the guard"; fail=$((fail+1))
else
  result "sha256 mismatch caught" 1 "$(run)"
fi

echo "=== M3: a .npy truncated mid-transfer — the cobalt failure, exactly ==="
setup; f="$TMP/m/data/generated/preds/good.npy"; truncate -s 200 "$f"
result "truncated npy caught" 1 "$(run)"

echo "=== M4: .sha256 file absent ==="
setup; rm "$TMP/m/data/generated/pod_delivery/artifact"/*.pt.sha256
result "missing checksum caught" 1 "$(run)"

echo "=== M5: no artifact at all ==="
setup; rm "$TMP/m/data/generated/pod_delivery/artifact"/*.pt
result "absent artifact caught" 1 "$(run)"

# ---------------------------------------------------------------------------------------------
# M6/M7 exercise pull(), which NO test reached until /falsify round 5. Its first-ever execution
# -- at 09:25:33Z, the instant pod five's artifact appeared -- hit
#     local tag=${spec%%:*} rest=${spec#*:} src=${rest%%:*} ...
# Bash expands every argument to `local` BEFORE assigning any of them, so `rest` was unset,
# `set -u` terminated the shell, and the watcher died having retrieved nothing and logged nothing.
# Six guard tests all targeted verify_local. A suite that cannot reach the retrieval path cannot
# catch a defect in the retrieval path.
#
# These run pull() for real against a LOCAL fake pod: a stub `ssh` that drops the ssh options and
# the user@host and execs the rest locally, so rsync's remote-shell protocol works over a
# directory. R and O are globals in watch_pod.sh, so the sourced library can be re-pointed.
# ---------------------------------------------------------------------------------------------
mkfakessh() {
  cat > "$TMP/fakessh" <<'EOS'
#!/usr/bin/env bash
# rsync may call this as `-l root host rsync --server ...` or `root@host rsync ...`.
# Consume ssh options (with their arguments), then exactly one hostname, then exec the rest.
while [ $# -gt 0 ]; do
  case "$1" in
    -l|-o|-p|-i|-F|-E) shift 2 ;;
    -*)                shift ;;
    *)                 shift; break ;;
  esac
done
exec "$@"
EOS
  chmod +x "$TMP/fakessh"
}

fakepod() {   # a pod tree pull() can actually read
  rm -rf "$TMP/pod"
  mkdir -p "$TMP/pod/deliver/artifact" "$TMP/pod/deliver/logs" \
           "$TMP/pod/repo/reports/plots/diagnostics" "$TMP/pod/repo/data/generated" \
           "$TMP/pod/repo/data/raw" "$TMP/pod/repo/logs" "$TMP/pod/repo/wandb"
  SRC=/home/simon/Documents/scripts/views_platform/views-models/models/bronze_kestrel/data/generated/pod_delivery/artifact
  cp "$SRC"/calibration_model_*.pt "$SRC"/calibration_model_*.pt.config.json \
     "$SRC"/calibration_model_*.pt.sha256 "$TMP/pod/deliver/artifact/"
  echo stage > "$TMP/pod/deliver/STAGE"
  python3 -c "
import numpy as np
np.save('$TMP/pod/repo/data/generated/good.npy', np.arange(1000, dtype='float32'))"
  echo nohup > "$TMP/pod/m.nohup"
  rm -rf "$TMP/m"; mkdir -p "$TMP/m"
}

runpull() {   # $1 = extra setup sourced inside the subshell (may break a source path)
  bash -c '
    WATCH_POD_LIB=1; source ./watch_pod.sh m x 1 0.0.0
    M=m; D='"$TMP"'/m; LOG='"$TMP"'/plog; ERRLOG='"$TMP"'/perr
    R='"$TMP"'/pod/repo; O='"$TMP"'/pod/deliver
    S="'"$TMP"'/fakessh"
    NOHUP_SRC='"$TMP"'/pod/m.nohup
    '"${1:-}"'
    pull >/dev/null 2>&1; echo $?'
}

echo "=== M6: pull() runs to completion and reports success on a readable pod ==="
mkfakessh; fakepod
: > "$TMP/plog"; : > "$TMP/perr"
# pull() fetches /workspace/$M.nohup by absolute path; point that one rsync at the fake pod too
sed_rc=$(runpull 'eval "$(declare -f pull | sed "s|root@\$IP:/workspace/\$M.nohup|root@\$IP:\$NOHUP_SRC|")"')
result "pull() exits 0 when every transfer succeeds" 0 "$sed_rc"
nlines=$(grep -c "rsync_exit=" "$TMP/plog" 2>/dev/null || echo 0)
result "pull() logged all 8 per-rsync exit codes (0 = it died early)" 8 "$nlines"
result "the artifact actually landed locally" 0 \
  "$(ls "$TMP/m/data/generated/pod_delivery/artifact"/calibration_model_*.pt >/dev/null 2>&1; echo $?)"

echo "=== M7: pull() reports FAILURE when a source is unreadable (F-4: no silent SECURED) ==="
mkfakessh; fakepod; rm -rf "$TMP/pod/repo/data/raw"
: > "$TMP/plog"; : > "$TMP/perr"
rc=$(runpull 'eval "$(declare -f pull | sed "s|root@\$IP:/workspace/\$M.nohup|root@\$IP:\$NOHUP_SRC|")"')
result "pull() exits non-zero when a transfer fails" 1 "$rc"
result "the failure was written to ERRLOG, not /dev/null (F-5)" 0 \
  "$([ -s "$TMP/perr" ] && echo 0 || echo 1)"

echo; echo "passed $pass, failed $fail"; [ "$fail" -eq 0 ]
