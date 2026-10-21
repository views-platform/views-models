#!/usr/bin/env bash
# Watch ONE pod to artifact, retrieve, and verify. Replaces watch_pair.sh.
#
# Fixes four defects found by /falsify rounds 2 and 4:
#   F-1  data/generated/ was in no rsync pattern                -> pulled (kept from watch_pair)
#   F-2  one-shot latch missed dirs that kept growing           -> re-pulls every poll (kept)
#   F-3  $TS was per-loop, so a no-op logged as a transfer      -> stamp + exit per rsync (kept)
#   F-4  SECURED meant "a .pt exists ON THE POD" and read no    -> SECURED now requires every pull
#        exit code; it printed success on all 94 passes while      to have exited 0 AND the local
#        artifact+sidecars reported rsync_exit=1 every time        sha256 to verify
#   F-5  rsync stderr went to /dev/null, so a failure repeated  -> stderr captured per rsync
#        94 times left nothing to diagnose. The cause of that
#        exit=1 died with the pods; this is why.
#   F-6  "identical to the pod" was taken as "intact"; a file   -> integrity scan: .npy/.npz must
#        still being written matched on BOTH sides and was         LOAD, not merely match
#        truncated. Cost: a 149MB array nearly deleted as a
#        duplicate of its own 46MB stub.
#   F-7  the library version was never asserted; 0.1.4 landed   -> EXPECTED_VER checked at
#        1h20m before indigo trained and confounded sigma-hat      verify_env, BEFORE training
#        (C-167)
#
# usage: watch_pod.sh <model> <ip> <port> <expected-hydranet-version>
set -u
M=${1:?model} IP=${2:?ip} PT=${3:?port} EXPECTED_VER=${4:?expected views-hydranet version}
L=/home/simon/Documents/scripts/views_platform/views-models/models
# BASH_SOURCE, not $0: when the test harness sources this file $0 is the interpreter
# (/bin/bash), so the logs went to /bin/ and every diagnostic append failed silently.
SELFDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SELFDIR/watch_${M}.log"
ERRLOG="$SELFDIR/watch_${M}.stderr"
S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PT -i $HOME/.ssh/id_ed25519"
R=/workspace/views-models/models/$M
O=/workspace/deliver/$M
D=$L/$M

say() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$LOG"; }
alarm() { say "⛔⛔ $*"; }

ver_checked=0

check_version() {           # F-7. Runs once, at the first stage that has a venv.
    [ "$ver_checked" = "1" ] && return 0
    local got
    got=$($S root@$IP "/workspace/venv/bin/python -c \
        'import importlib.metadata as m; print(m.version(\"views-hydranet\"))' 2>/dev/null" 2>/dev/null)
    [ -z "$got" ] && return 1          # venv not ready yet; try again next poll
    ver_checked=1
    if [ "$got" = "$EXPECTED_VER" ]; then
        say "✅ views-hydranet==$got  (expected $EXPECTED_VER)"
        return 0
    fi
    alarm "VERSION MISMATCH: pod has views-hydranet==$got, expected $EXPECTED_VER"
    alarm "   pod_run_model.sh:172 pins ~=0.1.1, so it takes whatever PyPI holds today."
    alarm "   A run on the wrong version confounds sigma-hat (C-167). STOP THE RUN."
    return 1
}

pull() {                    # F-3/F-4/F-5: per-rsync stamp, exit code, and stderr kept
    local all_ok=1
    mkdir -p "$D/logs" "$D/reports/plots" "$D/data/generated" "$D/data/raw" \
             "$D/data/generated/pod_delivery/artifact"
    set -- "artifact+sidecars:$O/artifact/:$D/data/generated/pod_delivery/artifact/" \
           "deliver:$O/:$D/data/generated/pod_delivery/" \
           "diagnostics:$R/reports/plots/diagnostics:$D/reports/plots/" \
           "generated:$R/data/generated/:$D/data/generated/" \
           "pipelogs:$R/logs/:$D/logs/" \
           "wandb:$R/wandb:$D/" \
           "raw:$R/data/raw/:$D/data/raw/"
    for spec in "$@"; do
        # one `local` per name: bash expands ALL args to `local` before assigning any,
        # so `src=${rest%%:*}` on the same line reads an unset `rest` and set -u kills
        # the shell. This killed the watcher at 09:25:33Z on its first-ever pull.
        local tag=${spec%%:*}
        local rest=${spec#*:}
        local src=${rest%%:*}
        local dst=${rest#*:}
        local rc
        # stderr to a real file, never /dev/null — F-5
        rsync -a --partial -e "$S" "root@$IP:$src" "$dst" >/dev/null 2>>"$ERRLOG"
        rc=$?
        say "$M  $tag  rsync_exit=$rc"
        [ "$rc" -ne 0 ] && { all_ok=0; alarm "$M  $tag FAILED rc=$rc — see $ERRLOG"; }
    done
    local rc2
    rsync -a -e "$S" "root@$IP:/workspace/$M.nohup" "$D/data/generated/pod_delivery/" \
          >/dev/null 2>>"$ERRLOG"; rc2=$?
    say "$M  nohup  rsync_exit=$rc2"
    [ "$rc2" -ne 0 ] && all_ok=0
    return $((1 - all_ok))
}

verify_local() {            # F-4/F-6: the artifact must be HERE, match its sha, and load
    local A="$D/data/generated/pod_delivery/artifact"
    local pt; pt=$(ls "$A"/calibration_model_*.pt 2>/dev/null | head -1)
    [ -n "$pt" ] || { say "   verify: no .pt locally yet"; return 1; }
    [ -f "$pt.config.json" ] || { alarm "SIDECAR MISSING — load_vehicle refuses weights alone (#549)"; return 1; }
    [ -f "$pt.sha256" ] || { alarm "NO .sha256 — cannot verify"; return 1; }
    local recorded computed
    recorded=$(awk '{print $1}' "$pt.sha256"); computed=$(sha256sum "$pt" | awk '{print $1}')
    if [ "$recorded" != "$computed" ]; then
        alarm "SHA256 MISMATCH on $(basename "$pt") — recorded $recorded computed $computed"
        return 1
    fi
    # F-6: matching the pod is not the same as being intact.
    local bad
    bad=$(python3 - "$D/data/generated" <<'PY' 2>/dev/null
import numpy as np, os, sys
n = 0
for root, _, fs in os.walk(sys.argv[1]):
    for f in fs:
        if not f.endswith((".npy", ".npz")): continue
        p = os.path.join(root, f)
        try:
            np.load(p, mmap_mode="r", allow_pickle=False)
        except Exception:
            try:
                z = np.load(p, allow_pickle=False); list(z.keys()); z.close()
            except Exception:
                n += 1; print(p)
print(f"COUNT={n}")
PY
)
    local nbad; nbad=$(echo "$bad" | sed -n 's/^COUNT=//p')
    if [ "${nbad:-0}" != "0" ]; then
        alarm "$nbad UNREADABLE .npy/.npz — truncated transfer, NOT safe to terminate"
        echo "$bad" | grep -v '^COUNT=' | head -5 | while read -r f; do say "      corrupt: $f"; done
        return 1
    fi
    say "   verify: sha256 OK (${computed:0:16}), sidecar present, all .npy/.npz load"
    return 0
}

# sourced by the test harness: define the functions, run nothing
[ "${WATCH_POD_LIB:-0}" = "1" ] && return 0

say "watching $M at $IP:$PT — expecting views-hydranet==$EXPECTED_VER"
: > "$ERRLOG"

while :; do
    st=$($S root@$IP "cat $O/STAGE 2>/dev/null" 2>/dev/null)
    al=$($S root@$IP 'pgrep -c -f "^/workspace/venv/bin/python main.py"' 2>/dev/null); al=${al:-0}
    ls_=$($S root@$IP "tr '\r' '\n' < /workspace/$M.nohup 2>/dev/null | grep -oE 'Lesson [0-9]+/300' | tail -1" 2>/dev/null)
    pt_=$($S root@$IP "ls $O/artifact/*.pt 2>/dev/null | wc -l" 2>/dev/null); pt_=${pt_:-0}
    say "$M alive=$al stage=${st:-?} ${ls_:-lesson?} artifact=$pt_"

    # F-7 — assert the version BEFORE training gets anywhere, not after
    case "$st" in verify_env|train|collect_artifact|evaluate) check_version ;; esac

    # a run that never reached train cannot be "gone" — every earlier stage has no main.py
    case "$st" in train|collect_artifact|evaluate) past_train=1 ;; *) past_train=0 ;; esac
    if [ "$al" = "0" ] && [ "$pt_" = "0" ] && [ "$past_train" = "1" ]; then
        alarm "$M JOB GONE WITH NO ARTIFACT (stage=$st)"
    fi

    if [ "$pt_" != "0" ]; then
        if pull && verify_local; then
            say "⛔ $M SECURED — every pull exited 0, sha256 verified, no corrupt files."
            say "   Safe to terminate pod $IP:$PT"
        else
            alarm "$M NOT SECURED this pass — do NOT terminate. See above."
        fi
    fi
    sleep 180
done
