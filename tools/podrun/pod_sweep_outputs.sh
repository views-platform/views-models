#!/usr/bin/env bash
# pod_sweep_outputs.sh --since <ref> [--mounts <file>] <model_name>
#
# ┌──────────────────────────────────────────────────────────────────────────────────────┐
# │  Answers ONE question, on the pod, before the pod is destroyed:                       │
# │  "which files did this run touch, and are any of them outside the collected bundle?"  │
# │  It REFUSES to answer rather than answering wrongly. An empty result is never a pass  │
# │  until a positive control proves the sweep could see something on that mount.          │
# └──────────────────────────────────────────────────────────────────────────────────────┘
#
# Why this exists (risk register C-168, C-169, C-170, C-171 — Cluster G):
#
#   On 2026-10-25 a heavy_freighter control run was declared ready to terminate. The bundle
#   had been verified 2754/2754 byte-identical against $OUT — and that proved nothing,
#   because $OUT is the runner's OWN declaration of the deliverable. Five classes of output
#   lived outside it: the computed metricframes, the W&B evaluation tables, the by_* gate
#   posterior, the training input with its queryset digest, and the pipeline logs.
#
#   The omission sweep written to catch exactly that returned EMPTY, and the empty result
#   was read as a clean bill of health. It was `find / -xdev`, and /workspace on RunPod is
#   a network mount (mfs#...runpod.net) on a different device, so -xdev excluded the one
#   place the data was. 43 files counted while one directory held 156 draw files and 13 GB.
#
# So this script does three things the broken one-liner did not:
#
#   1. Sweeps EACH MOUNT SEPARATELY, so -xdev is correct instead of silently excluding.
#   2. Plants a sentinel on each mount and requires the sweep to FIND it before an empty
#      result from that mount is allowed to mean anything. No sentinel -> UNVERIFIED -> exit 1.
#   3. Computes every timestamp HERE, on the pod. C-171's lesson: a watcher with a
#      hardcoded epoch was a year off, so a stale file passed as fresh.
#
# It is read-only with respect to the run: it writes nothing but its own sentinels, and
# removes each one it writes. It does not touch pod_run_model.sh, which is read-only.
#
# Exit codes — the whole point is that these are distinguishable:
#   0  swept, verified, and every touched file is inside $OUT          -> safe to destroy
#   2  swept, verified, and files were touched OUTSIDE $OUT            -> collect them first
#   1  the sweep could not be verified on at least one mount           -> DO NOT DESTROY;
#                                                                         this is "I could
#                                                                         not look", not
#                                                                         "nothing is there"
#   3  usage error
#
# Usage:
#   ./pod_sweep_outputs.sh --since @1792924998            heavy_freighter
#   ./pod_sweep_outputs.sh --since 2026-10-25T10:43:00    heavy_freighter
#   ./pod_sweep_outputs.sh --since /workspace/deliver/heavy_freighter/.lessons heavy_freighter
#
#   --since takes an @epoch, an ISO-8601 timestamp, or a file whose mtime is the run start.
#           `$OUT/.lessons` is usually the right reference: pod_run_model.sh clears it at
#           startup and the lesson loop rewrites it, so its mtime brackets training.
#   --mounts overrides /proc/mounts. A test seam, so the sweep can be exercised off a pod.

set -uo pipefail

ROOT="${PODRUN_ROOT:-/workspace}"
MOUNTS_FILE=/proc/mounts
SINCE_REF=""
MODEL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --since)  SINCE_REF="${2:-}"; shift 2 ;;
        --mounts) MOUNTS_FILE="${2:-}"; shift 2 ;;
        -*)       echo "unknown option: $1" >&2; exit 3 ;;
        *)        MODEL="$1"; shift ;;
    esac
done

[ -n "$MODEL" ]     || { echo "usage: $0 --since <ref> [--mounts <file>] <model_name>" >&2; exit 3; }
[ -n "$SINCE_REF" ] || { echo "--since is required: an @epoch, an ISO timestamp, or a file" >&2; exit 3; }
[ -r "$MOUNTS_FILE" ] || { echo "cannot read mount table: $MOUNTS_FILE" >&2; exit 3; }

OUT=$ROOT/deliver/$MODEL

# ---- resolve --since to an epoch, HERE, on this host (C-171) -------------------------
case "$SINCE_REF" in
    @*) SINCE=${SINCE_REF#@} ;;
    *)  if [ -e "$SINCE_REF" ]; then
            SINCE=$(stat -c %Y "$SINCE_REF") || exit 3
        else
            SINCE=$(date -d "$SINCE_REF" +%s 2>/dev/null) || {
                echo "--since '$SINCE_REF' is neither an @epoch, a readable path, nor a date" >&2
                exit 3
            }
        fi ;;
esac
case "$SINCE" in ''|*[!0-9]*) echo "resolved --since is not an epoch: '$SINCE'" >&2; exit 3 ;; esac

NOW=$(date +%s)
[ "$SINCE" -lt "$NOW" ] || { echo "--since resolves to $SINCE, which is not before now ($NOW)" >&2; exit 3; }

echo "sweep root-per-mount | since $SINCE ($(date -d @"$SINCE" -Iseconds)) | now $NOW | \$OUT=$OUT"
echo

# ---- the mounts worth sweeping -------------------------------------------------------
# Pseudo filesystems hold no run output and would drown the result. Everything else is
# swept, including read-only mounts — a read-only mount still has to be VERIFIED, because
# "I cannot write a sentinel here" is not "nothing was written here".
PSEUDO='^(proc|sysfs|devpts|devtmpfs|tmpfs|cgroup|cgroup2|securityfs|debugfs|tracefs|pstore|bpf|configfs|fusectl|mqueue|hugetlbfs|autofs|binfmt_misc|efivarfs|ramfs|nsfs|squashfs|overlay)$'

MOUNTPOINTS=$(awk '{print $2, $3}' "$MOUNTS_FILE" \
    | grep -vE " ($(echo "$PSEUDO" | sed 's/^\^(//; s/)\$$//'))\$" \
    | awk '{print $1}' | sort -u)

[ -n "$MOUNTPOINTS" ] || { echo "UNVERIFIED: the mount table yielded no sweepable mounts" >&2; exit 1; }

UNVERIFIED=""
OUTSIDE_LIST=$(mktemp)
INSIDE_COUNT=0
OUTSIDE_COUNT=0

for MP in $MOUNTPOINTS; do
    [ -d "$MP" ] || { UNVERIFIED="$UNVERIFIED $MP(not-a-directory)"; continue; }

    # ---- positive control: can this sweep see anything at all on this mount? --------
    SENTINEL="$MP/.podsweep_sentinel.$$"
    if ! touch "$SENTINEL" 2>/dev/null; then
        UNVERIFIED="$UNVERIFIED $MP(cannot-plant-sentinel)"
        continue
    fi
    SEEN=$(find "$MP" -xdev -maxdepth 1 -name ".podsweep_sentinel.$$" -newermt "@$SINCE" 2>/dev/null | wc -l)
    rm -f "$SENTINEL"
    if [ "$SEEN" -ne 1 ]; then
        UNVERIFIED="$UNVERIFIED $MP(sentinel-not-found)"
        continue
    fi

    # ---- the actual sweep, confined to THIS mount ------------------------------------
    HITS=$(find "$MP" -xdev -type f -newermt "@$SINCE" ! -name ".podsweep_sentinel.*" 2>/dev/null)
    N=$(printf '%s' "$HITS" | grep -c . )

    case "$MP" in
        *) IN=$(printf '%s\n' "$HITS" | grep -c "^$OUT/" || true) ;;
    esac
    OUTN=$(( N - IN ))
    INSIDE_COUNT=$(( INSIDE_COUNT + IN ))
    OUTSIDE_COUNT=$(( OUTSIDE_COUNT + OUTN ))
    printf '%-28s verified  %6d touched  %6d in $OUT  %6d OUTSIDE\n' "$MP" "$N" "$IN" "$OUTN"
    [ "$OUTN" -gt 0 ] && printf '%s\n' "$HITS" | grep -v "^$OUT/" | grep . >> "$OUTSIDE_LIST"
done

echo
if [ -n "$UNVERIFIED" ]; then
    echo "⛔ UNVERIFIED MOUNTS:$UNVERIFIED"
    echo "   The sweep could not prove it can see these. That is 'I could not look', not"
    echo "   'nothing is there' (register C-169). DO NOT destroy this host on this result."
    rm -f "$OUTSIDE_LIST"
    exit 1
fi

if [ "$OUTSIDE_COUNT" -gt 0 ]; then
    echo "⛔ $OUTSIDE_COUNT file(s) touched by this run live OUTSIDE \$OUT. Collect them first."
    echo "   Directories, with counts and sizes:"
    sed 's|/[^/]*$||' "$OUTSIDE_LIST" | sort | uniq -c | sort -rn | head -40 \
        | while read -r c d; do printf '   %6d  %-12s %s\n' "$c" "$(du -sh "$d" 2>/dev/null | cut -f1)" "$d"; done
    echo
    echo "   Full list: $OUTSIDE_LIST"
    exit 2
fi

echo "✅ swept and verified on every mount. $INSIDE_COUNT touched file(s), all inside \$OUT."
echo "   This is the only result on which destroying the host is justified."
rm -f "$OUTSIDE_LIST"
exit 0
