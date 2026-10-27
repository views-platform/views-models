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

# ⛔ `shift 2` with one argument left FAILS WITHOUT SHIFTING, and the loop condition never
#    changes — so a trailing `--since` hung forever instead of printing the usage error.
#    Measured 2026-10-27: `pod_sweep_outputs.sh --since` never returned. This is a tool run
#    on a pod under time pressure before destroying a host; a typo must cost a line of
#    output, not the terminal.
need_value() { [ "$2" -ge 2 ] || { echo "$1 requires a value" >&2; exit 3; }; }

while [ $# -gt 0 ]; do
    case "$1" in
        --since)  need_value --since  $#; SINCE_REF="$2"; shift 2 ;;
        --mounts) need_value --mounts $#; MOUNTS_FILE="$2"; shift 2 ;;
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
# Kernel pseudo filesystems hold no run output and would drown the result. Everything
# else is swept, including read-only mounts — a read-only mount still has to be VERIFIED,
# because "I cannot write a sentinel here" is not "nothing was written here".
#
# ⛔ `overlay` is NOT in this list, and that is deliberate. On a container `/` is usually
#    overlay: it is the real root filesystem, not a pseudo one, and `/root` is where this
#    platform's own convention puts credentials and hand-written files (/workspace silently
#    ignores chmod). Excluding it reproduced the incident this script exists to prevent,
#    with a mount-type filter in place of `-xdev`: measured 2026-10-27 with `/` overlay and
#    `/workspace` a network mount, a file in /root was invisible and the script printed
#    "safe to destroy".
#
# ⚠️ `tmpfs`, `ramfs` and `squashfs` ARE still excluded, and that is a stated judgement
#    rather than a measurement: /tmp and /dev/shm do die with the host, so output left
#    there would be lost — but no instance of this pipeline writing run output to tmpfs
#    has been observed, and sweeping /dev/shm makes the verdict depend on unrelated shared
#    memory. Only the overlay defect was demonstrated, so only overlay was changed. The
#    trigger to revisit: any run found to leave an artifact on a tmpfs mount.
#
# ⚠️ And the exclusion is PRINTED, not silent. A verdict that says "every mount" while a
#    filter removed a class is the defect, whatever the filter is (ADR-025 §2.4).
PSEUDO='^(proc|sysfs|devpts|devtmpfs|tmpfs|cgroup|cgroup2|securityfs|debugfs|tracefs|pstore|bpf|configfs|fusectl|mqueue|hugetlbfs|autofs|binfmt_misc|efivarfs|ramfs|nsfs|squashfs)$'
PSEUDO_ALT=$(echo "$PSEUDO" | sed 's/^\^(//; s/)\$$//')

MOUNTPOINTS=$(awk '{print $2, $3}' "$MOUNTS_FILE" \
    | grep -vE " ($PSEUDO_ALT)\$" \
    | awk '{print $1}' | sort -u)

EXCLUDED=$(awk '{print $2, $3}' "$MOUNTS_FILE" \
    | grep -E " ($PSEUDO_ALT)\$" \
    | sort -u)

[ -n "$MOUNTPOINTS" ] || { echo "UNVERIFIED: the mount table yielded no sweepable mounts" >&2; exit 1; }

if [ -n "$EXCLUDED" ]; then
    echo "excluded as kernel pseudo filesystems (not swept, and not claimed as swept):"
    printf '%s\n' "$EXCLUDED" | sed 's/^/   /'
    echo
fi

UNVERIFIED=""
OUTSIDE_LIST=$(mktemp)
INSIDE_COUNT=0
OUTSIDE_COUNT=0

FILE_MOUNTS=""

for MP in $MOUNTPOINTS; do
    # ⛔ A bind-mounted FILE is not a tree to sweep, and every container has several —
    #    /etc/hosts, /etc/hostname, /etc/resolv.conf, fstype ext4 or overlay, none of them
    #    pseudo. The previous version classified each as UNVERIFIED, so on a real pod this
    #    script could NEVER return 0: it printed "DO NOT DESTROY" every time, which trains
    #    an operator to ignore it. That is the cry-wolf inverse of C-169 and just as
    #    dangerous, because a refusal nobody reads is not a refusal.
    #
    #    A single file needs no sentinel and no recursion — stat it. ⚠️ A touched file mount
    #    is reported on its own line and deliberately does NOT drive exit 2: /etc/resolv.conf
    #    is rewritten by the container runtime and is not run output to collect. It is named
    #    rather than hidden, which is the standard this script holds itself to.
    if [ ! -d "$MP" ]; then
        if [ ! -e "$MP" ]; then
            UNVERIFIED="$UNVERIFIED $MP(mountpoint-missing)"
            continue
        fi
        FM=$(stat -c %Y "$MP" 2>/dev/null) || {
            UNVERIFIED="$UNVERIFIED $MP(cannot-stat-file-mount)"
            continue
        }
        if [ "$FM" -gt "$SINCE" ]; then
            FILE_MOUNTS="$FILE_MOUNTS $MP"
        fi
        continue
    fi

    # ---- positive control: can this sweep see anything at all on this mount? --------
    # ⛔ The sentinel is planted INSIDE A SUBDIRECTORY and found with the same recursive
    #    find as the real sweep. An earlier version planted it at depth 1 and looked with
    #    `-maxdepth 1`, so the control proved only that the top of the mount was visible
    #    while the real sweep recursed — and an unreadable SUBdirectory holding uncollected
    #    output was skipped in silence. Measured 2026-10-27: the script printed "safe to
    #    destroy" over a chmod-000 subdirectory. A control must exercise the instrument,
    #    not a shallower proxy (ADR-025 §2.4).
    SENTINEL_DIR="$MP/.podsweep_sentinel.$$"
    if ! mkdir -p "$SENTINEL_DIR/deep" 2>/dev/null || ! touch "$SENTINEL_DIR/deep/probe" 2>/dev/null; then
        UNVERIFIED="$UNVERIFIED $MP(cannot-plant-sentinel)"
        rm -rf "$SENTINEL_DIR" 2>/dev/null
        continue
    fi
    SEEN=$(find "$MP" -xdev -type f -newermt "@$SINCE" -path "*/.podsweep_sentinel.$$/deep/probe" 2>/dev/null | wc -l)
    rm -rf "$SENTINEL_DIR"
    if [ "$SEEN" -ne 1 ]; then
        UNVERIFIED="$UNVERIFIED $MP(sentinel-not-found)"
        continue
    fi

    # ---- the actual sweep, confined to THIS mount ------------------------------------
    # ⛔ find's stderr AND its exit code are both read. Discarding them is C-113's lesson
    #    exactly: a permission-denied on one subtree otherwise yields a short list that is
    #    indistinguishable from a complete one. "I could not read part of this" is
    #    UNVERIFIED, never a pass.
    ERRFILE=$(mktemp)
    HITS=$(find "$MP" -xdev -type f -newermt "@$SINCE" ! -path "*/.podsweep_sentinel.*" 2>"$ERRFILE")
    FIND_RC=$?
    if [ "$FIND_RC" -ne 0 ] || [ -s "$ERRFILE" ]; then
        UNVERIFIED="$UNVERIFIED $MP(find-could-not-read-all-of-it)"
        echo "   $MP: find exited $FIND_RC and reported:" >&2
        head -3 "$ERRFILE" | sed 's/^/      /' >&2
        rm -f "$ERRFILE"
        continue
    fi
    rm -f "$ERRFILE"
    N=$(printf '%s' "$HITS" | grep -c . )

    IN=$(printf '%s\n' "$HITS" | grep -c "^$OUT/" || true)
    OUTN=$(( N - IN ))
    INSIDE_COUNT=$(( INSIDE_COUNT + IN ))
    OUTSIDE_COUNT=$(( OUTSIDE_COUNT + OUTN ))
    printf '%-28s verified  %6d touched  %6d in $OUT  %6d OUTSIDE\n' "$MP" "$N" "$IN" "$OUTN"
    [ "$OUTN" -gt 0 ] && printf '%s\n' "$HITS" | grep -v "^$OUT/" | grep . >> "$OUTSIDE_LIST"
done

echo
if [ -n "$FILE_MOUNTS" ]; then
    echo "file mounts touched since --since (single bind-mounted files, NOT run output):"
    for F in $FILE_MOUNTS; do echo "   $F"; done
    echo "   Reported, not collected, and not counted toward the verdict. If one of these"
    echo "   IS a deliverable for this run, that is a new case and this line is the signal."
    echo
fi

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

echo "✅ swept and verified on every mount listed above. $INSIDE_COUNT touched file(s), all inside \$OUT."
echo "   This is the only result on which destroying the host is justified."
[ -n "$EXCLUDED" ] && echo "   ⚠️  It does NOT cover the kernel pseudo filesystems listed at the top."
rm -f "$OUTSIDE_LIST"
exit 0
