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
# It is read-only with respect to the run: it writes nothing but its own sentinels and two
# temp files, and a trap removes all of them on any exit, including Ctrl-C. It does not touch
# pod_run_model.sh, which is read-only.
#
# ⛔ WHAT THIS CANNOT SEE, stated because the headline above over-claims otherwise. It answers
#    "which files have an mtime at or after --since", which is NOT the same as "which files did
#    this run touch":
#      - a file copied in with `cp -p`, `rsync -a` or `tar -xp` keeps its SOURCE mtime, so a
#        deliverable staged that way is invisible to this sweep;
#      - a file written and then back-dated is invisible;
#      - `find -newermt` is strictly-greater, so the whole-second boundary case (common when
#        --since is an @epoch read from a log) would be missed. The sweep therefore runs from
#        SINCE-1 and accepts one second of slack rather than dropping a boundary file.
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
#           `$OUT/.lessons` is usually the right reference, but NOT for the reason an earlier
#           version of this comment gave. It claimed "the lesson loop rewrites it, so its mtime
#           brackets training". It does not: `pod_run_model.sh:321` writes it exactly once, in
#           the pre-flight, and `:134` removes it at startup. So its mtime marks the START of
#           training, which is what --since wants. ⛔ Had the old claim been true, --since would
#           have landed at the END of training and every sweep would have found nothing — the
#           failure the A1 floor below now catches. Safe by accident, correct on purpose now.
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
        *)        [ -z "$MODEL" ] || { echo "two model names given ('$MODEL' and '$1'); refusing"\
                      " rather than silently using the last" >&2; exit 3; }
                  MODEL="$1"; shift ;;
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

# ⛔ One trap, every exit path. The old version removed its sentinel inline, which meant a
#    Ctrl-C -- the interaction most likely on a pod under time pressure -- left a sentinel
#    directory on the mount, and the non-PID-scoped sweep exclusion below then made that
#    leftover a permanent, unreported blind spot. Lines above claim "removes each one it
#    writes"; this is what makes that true.
#
# ⚠️ The temp files carry a PID-scoped, recognisable name so the sweep can exclude ITS OWN
#    working files. With TMPDIR on a swept mount the old version reported its own mktemp files
#    as uncollected run output -- the cry-wolf direction, but still a mis-report.
TMPPREFIX=".podsweep_tmp.$$"
OUTSIDE_LIST=$(mktemp -t "$TMPPREFIX.outside.XXXXXX")
INSIDE_LIST=$(mktemp -t "$TMPPREFIX.inside.XXXXXX")
SENTINEL_DIR=""
ERRFILE=""
KEEP_OUTSIDE_LIST=0

cleanup() {
    [ -n "$SENTINEL_DIR" ] && rm -rf "$SENTINEL_DIR"
    [ -n "$ERRFILE" ] && rm -f "$ERRFILE"
    rm -f "$INSIDE_LIST"
    [ "$KEEP_OUTSIDE_LIST" = "1" ] || rm -f "$OUTSIDE_LIST"
    return 0
}
# ⛔ INT and TERM must EXIT, not just clean up. `trap cleanup EXIT INT TERM` runs the handler
#    and then lets the script CONTINUE -- verified 2026-10-27 -- so a Ctrl-C mid-sweep would
#    delete the working lists and then compute the verdict from empty ones. The A1 floor would
#    catch the resulting 0-inside case, but a signalled run must stop and say so, not produce a
#    verdict at all. 130/143 are the conventional signal exits and are distinguishable from
#    this script's own 0/1/2/3 vocabulary.
trap cleanup EXIT
trap 'cleanup; echo "interrupted -- no verdict was reached, and none should be read from this run" >&2; exit 130' INT
trap 'cleanup; echo "terminated -- no verdict was reached, and none should be read from this run" >&2; exit 143' TERM

# A4: one second of slack so a file whose mtime EQUALS --since is not dropped.
SWEEP_SINCE=$(( SINCE - 1 ))

UNVERIFIED=""
VERIFIED_MPS=""

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
        SENTINEL_DIR=""
        continue
    fi
    SEEN=$(find "$MP" -xdev -type f -newermt "@$SWEEP_SINCE" -path "*/.podsweep_sentinel.$$/deep/probe" 2>/dev/null | wc -l)
    rm -rf "$SENTINEL_DIR"
    SENTINEL_DIR=""
    if [ "$SEEN" -ne 1 ]; then
        UNVERIFIED="$UNVERIFIED $MP(sentinel-not-found)"
        continue
    fi
    VERIFIED_MPS="$VERIFIED_MPS $MP"

    # ---- the actual sweep, confined to THIS mount ------------------------------------
    # ⛔ find's stderr AND its exit code are both read. Discarding them is C-113's lesson
    #    exactly: a permission-denied on one subtree otherwise yields a short list that is
    #    indistinguishable from a complete one. "I could not read part of this" is
    #    UNVERIFIED, never a pass.
    ERRFILE=$(mktemp -t "$TMPPREFIX.err.XXXXXX")
    HITS=$(find "$MP" -xdev -type f -newermt "@$SWEEP_SINCE" \
                ! -path "*/.podsweep_sentinel.$$/*" ! -name "$TMPPREFIX.*" 2>"$ERRFILE")
    FIND_RC=$?
    if [ "$FIND_RC" -ne 0 ] || [ -s "$ERRFILE" ]; then
        UNVERIFIED="$UNVERIFIED $MP(find-could-not-read-all-of-it)"
        echo "   $MP: find exited $FIND_RC and reported:" >&2
        head -3 "$ERRFILE" | sed 's/^/      /' >&2
        rm -f "$ERRFILE"; ERRFILE=""
        continue
    fi
    rm -f "$ERRFILE"; ERRFILE=""

    # ⛔ `case`, not `grep -c "^$OUT/"`. $OUT was interpolated into a REGEX, so a model name
    #    or PODRUN_ROOT containing `.`, `*`, `[`, `+` or `?` made a file OUTSIDE the bundle
    #    match as INSIDE -- measured with MODEL='a.c': deliver/aXc/ counted inside, exit 0,
    #    "safe to destroy". A glob with a quoted variable cannot do that.
    #
    # ⚠️ Hits go to FILES, deduped once at the end, because `-xdev` prunes at DEVICE
    #    boundaries while this loop iterates DECLARED mountpoints: a bind-mounted directory is
    #    swept by both its parent and itself, and the running totals counted the same file
    #    twice. Deriving the counts from the deduped lists removes the arithmetic entirely.
    IN=0; OUTN=0
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        case "$f" in
            "$OUT"/*) IN=$(( IN + 1 ));   printf '%s\n' "$f" >> "$INSIDE_LIST" ;;
            *)        OUTN=$(( OUTN + 1 )); printf '%s\n' "$f" >> "$OUTSIDE_LIST" ;;
        esac
    done <<HITS_EOF
$HITS
HITS_EOF
    printf '%-28s verified  %6d touched  %6d in $OUT  %6d OUTSIDE\n' \
        "$MP" "$(( IN + OUTN ))" "$IN" "$OUTN"
done

sort -u -o "$INSIDE_LIST"  "$INSIDE_LIST"
sort -u -o "$OUTSIDE_LIST" "$OUTSIDE_LIST"
INSIDE_COUNT=$(grep -c . "$INSIDE_LIST"  || true)
OUTSIDE_COUNT=$(grep -c . "$OUTSIDE_LIST" || true)

# ---- A3: was $OUT itself inside a mount this sweep verified? -------------------------
# ⛔ Measured 2026-10-27: with $OUT's own mount declared tmpfs -- which the filter excludes --
#    the sweep covered one unrelated empty mount and printed "safe to destroy", exit 0. The
#    deliverable's own filesystem was never looked at. Nothing checked that, so it is checked
#    here, and the message names the mount rather than leaving a reader to work it out
#    (ADR-020: a refusal says where the stairs end).
OUT_COVERED=0
for MP in $VERIFIED_MPS; do
    case "$OUT" in "$MP"|"$MP"/*) OUT_COVERED=1; break ;; esac
done
if [ "$OUT_COVERED" -ne 1 ]; then
    OUT_MP=$(awk -v o="$OUT" '{ if (index(o, $2) == 1 && length($2) > length(best)) best=$2 } END { print best }' "$MOUNTS_FILE")
    UNVERIFIED="$UNVERIFIED \$OUT(not-on-any-verified-mount${OUT_MP:+; it is under $OUT_MP})"
fi

# ---- A1: the independent floor -- did the sweep see ANY of the deliverable? ----------
# ⛔ This is the control §2.4 of ADR-025 asks for, and the sentinel is not it. The sentinel is
#    created NOW and only requires SINCE < NOW, so it passes for ANY accepted --since: it
#    proves the mount and the find predicates work, never that the window brackets the run.
#    Measured 2026-10-27: --since set one hour ago over a two-hour-old tree swept 0 files and
#    printed "0 touched file(s), all inside $OUT. This is the only result on which destroying
#    the host is justified." -- with a gate posterior sitting outside the bundle.
#
#    The floor is independent because another party already guarantees it:
#    pod_run_model.sh:546 refuses to report OK unless $OUT holds a non-empty
#    predictions_forecasting_* . So $OUT is known non-empty by someone other than this script,
#    and a sweep that sees nothing under $OUT has the wrong window, not a clean host.
if [ "$INSIDE_COUNT" -eq 0 ] && [ "$OUT_COVERED" -eq 1 ]; then
    UNVERIFIED="$UNVERIFIED \$OUT(swept-but-empty-in-this-window)"
fi

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
    case "$UNVERIFIED" in
      *swept-but-empty-in-this-window*)
        echo
        echo "   \$OUT was swept and held NOTHING newer than --since. The run's own deliverable"
        echo "   is always there, so this is a --since that does not bracket the run, not a"
        echo "   clean host. Re-run with --since \$OUT/.lessons, whose mtime is the training"
        echo "   START (pod_run_model.sh:321), or with the @epoch from the run log." ;;
    esac
    case "$UNVERIFIED" in
      *not-on-any-verified-mount*)
        echo
        echo "   \$OUT is not under any mount this sweep verified, so the deliverable's own"
        echo "   filesystem was never examined. Check whether its mount is being excluded as"
        echo "   a pseudo filesystem at the top of this output." ;;
    esac
    exit 1
fi

if [ "$OUTSIDE_COUNT" -gt 0 ]; then
    echo "⛔ $OUTSIDE_COUNT file(s) touched by this run live OUTSIDE \$OUT. Collect them first."
    echo "   Directories, with counts and sizes:"
    sed 's|/[^/]*$||' "$OUTSIDE_LIST" | sort | uniq -c | sort -rn | head -40 \
        | while read -r c d; do printf '   %6d  %-12s %s\n' "$c" "$(du -sh "$d" 2>/dev/null | cut -f1)" "$d"; done
    echo
    echo "   Full list: $OUTSIDE_LIST"
    KEEP_OUTSIDE_LIST=1
    exit 2
fi

echo "✅ swept and verified on every mount listed above. $INSIDE_COUNT touched file(s), all inside \$OUT."
echo "   This is the only result on which destroying the host is justified."
[ -n "$EXCLUDED" ] && echo "   ⚠️  It does NOT cover the kernel pseudo filesystems listed at the top."
exit 0
