#!/usr/bin/env bash
# pod_run_exp02_arm.sh [options] <arm>
#
# ┌──────────────────────────────────────────────────────────────────────────────────────┐
# │  ONE EXP-02 ARM, ON ONE POD, END TO END.  Four pods run this, one arm each.           │
# │  It drives `pod_run_model.sh` and NEVER modifies it (standing rule: read-only).       │
# └──────────────────────────────────────────────────────────────────────────────────────┘
#
# WHY THIS EXISTS, and it is not convenience. `pod_run_model.sh` is correct and complete for
# what it claims, and EXP-02 needs two things it does not and should not do:
#
#   1. THE GATE MUST COME HOME. The runner archives `-path '*/lr_*'` and leaves `by_*` — the
#      gate posterior — to die with the pod, while reporting `STATUS: OK` (views-models#558,
#      register C-161). Every P-23 endpoint is gate-side. Until now the rule "slice the gate
#      before teardown" lived only in prose (docs/runpod_run_guide.md) and in an xfail stub
#      (tests/test_falsification_pod_teardown_completeness.py). ⛔ On 2026-10-26 the prose was
#      not followed, and on 2026-10-13 the gate survived only because someone sliced it by
#      hand at 2am. **Here it is structural: this script does not write OK unless the slice
#      succeeded.** A readout that depends on an operator remembering something at teardown is
#      a readout that will one day be missing.
#
#   2. EVERY ARM MUST BE ON ONE DATA PULL. P-23 is a within-experiment sign test, and sigma
#      pools seeds 42-47, so the set is TWELVE arms and one pull spans all of them. If two
#      arms train on different datafactory pulls it silently becomes a
#      cross-population comparison — which is exactly what P-30's refetch already did to P-27,
#      and it did not announce itself. The runner's train leg (`main.py -r calibration -t`,
#      pod_run_model.sh:379) carries NO `-sa`, so it refetches on every pod regardless of what
#      is shipped to it. ⛔ So do NOT scp the parquet and do NOT add `-sa`: neither changes
#      what the train leg does. What works is to let each pod fetch and then CHECK what it
#      got, against a digest pinned here.
#
# WHAT IT DOES, in order:
#
#   preflight   -> the arm is one of the declared four; repo, runner and slicer present;
#                  /root/.netrc present; the box can archive (see CORES_MAX below)
#   train       -> `pod_run_model.sh <arm>`, unmodified, in its own process group
#   digest gate -> as soon as the fetch writes its provenance sidecar — MINUTES in, not
#                  hours — the pull is verified and a mismatched arm is KILLED
#   gate slice  -> `slice_gate_posterior.py <arm>`, h=1 and h=36, mandatory
#   manifest    -> $OUT/EXP02_MANIFEST and $OUT/EXP02_STATUS
#
# ⚠️ IT WRITES ITS OWN MARKERS AND NEVER THE RUNNER'S. `$OUT/STATUS` belongs to
# `pod_run_model.sh` and `pod_run_fao_delivery.sh` reads it; this script writes
# `$OUT/EXP02_STATUS`, `$OUT/EXP02_MANIFEST` and `$OUT/QUARANTINED`. Two writers on one file
# is how a stale verdict outlives the run that wrote it.
#
# POD SPEC, measured, from the runs that failed: RAM >= 32 GB (prefer 48-64), VRAM >= 16 GB,
# disk >= 60 GB. Only the core count is enforced here, because it is the one that fails
# SILENTLY — see CORES_MAX.

set -uo pipefail

USAGE='usage: pod_run_exp02_arm.sh [--repo DIR] [--root DIR] [--runner PATH] [--slicer PATH]
                                  [--poll-seconds N] [--fetch-timeout-seconds N]
                                  [--max-cores N] [--expect-digest SHA256]
                                  [--netrc PATH] <arm>'

# ── the four arms, declared ───────────────────────────────────────────────────────────
# Named rather than globbed. A glob over `models/*fullzero*` would silently pick up the
# `*tinyzero*` rehearsal arms, which share the pull and the config shape and are NOT part of
# the P-23 sign test. The set is the experiment's definition, so it is written down.
# ⛔ TWELVE, not four. EXP-03 (seeds 44-47) extended the scored set, and sigma pools 42-47 —
# 5 degrees of freedom is n=6 — so the within-experiment comparison this list protects now
# spans all twelve. An arm absent from here is refused at preflight, which is what kept the
# `*tinyzero*` rehearsal arms out: they share the pull and the config shape and are
# deliberately not members. That exclusion is unchanged; only the membership grew.
ARMS=(
  calm_orbiter      calm_lander
  keen_orbiter    keen_lander
  mellow_orbiter     mellow_lander
  nimble_orbiter     nimble_lander
  quiet_orbiter      quiet_lander
  rapid_orbiter    rapid_lander
)

# ── the pull every arm must land on ───────────────────────────────────────────────────
# ⛔ NOT a guess and not a relayed string. Read 2026-11-02 from the provenance sidecars of all
# six prepared arm directories on the operator's laptop
# (`models/<arm>/data/raw/calibration_datafactory_df.parquet.provenance.json`), which agreed
# on all 64 characters: the post-P-30 global pull, pgm, months 121-504. The views-hydranet
# seat supplied the first 16 and they match.
#
# If the pull is deliberately moved, change this line and say so in the EXP-02 prereg. The
# `--expect-digest` flag exists so a corrected value does not need a code edit at 2am, and
# the refusal below prints it.
EXPECTED_DIGEST="b32a221b70e5255d3d85e683b483d9398ae08ed3b5e9d1be71c6e8f7aa13422c"
EXPECTED_LEVEL="pgm"
EXPECTED_MONTH_FIRST=121
EXPECTED_MONTH_LAST=504

# ── the one hardware bound that fails silently ────────────────────────────────────────
# `pod_run_model.sh:638` archives the posterior with `zstd -3 -T0` — one thread per core. On a
# 128-core pod that died in 2.3 s and left a TRUNCATED 1 MB archive from 182 MB of input
# (#558 addendum), and the runner's own N_DRAWS/N_ARCHIVED guard did not catch it. The runner
# is read-only, so the bound is enforced on the box instead of on the flag: refuse to start on
# a machine whose core count is known to break the archive step.
#
# `slice_gate_posterior.py` uses `-T4` and is unaffected; this is about the runner's archive.
CORES_MAX=32

REPO_DEFAULT=""
ROOT_DEFAULT="${PODRUN_ROOT:-/workspace}"
POLL_SECONDS=10
FETCH_TIMEOUT_SECONDS=3600
RUNNER=""
SLICER=""
MAX_CORES=$CORES_MAX
EXPECT_DIGEST="$EXPECTED_DIGEST"
# The credential this run cannot start without. A CLI argument rather than a constant so
# the test suite can point it at a fixture, and so what a run used is visible in the
# command that ran it — the same reason slice_gate_posterior.py takes --repo and --out.
NETRC="/root/.netrc"

while [ $# -gt 0 ]; do
    case "$1" in
        --repo)                   [ $# -ge 2 ] || { echo "!!! --repo needs a directory" >&2; exit 1; }
                                  REPO_DEFAULT="$2"; shift 2 ;;
        --root)                   [ $# -ge 2 ] || { echo "!!! --root needs a directory" >&2; exit 1; }
                                  ROOT_DEFAULT="$2"; shift 2 ;;
        --runner)                 [ $# -ge 2 ] || { echo "!!! --runner needs a path" >&2; exit 1; }
                                  RUNNER="$2"; shift 2 ;;
        --slicer)                 [ $# -ge 2 ] || { echo "!!! --slicer needs a path" >&2; exit 1; }
                                  SLICER="$2"; shift 2 ;;
        --poll-seconds)           POLL_SECONDS="$2"; shift 2 ;;
        --fetch-timeout-seconds)  FETCH_TIMEOUT_SECONDS="$2"; shift 2 ;;
        --max-cores)              MAX_CORES="$2"; shift 2 ;;
        --expect-digest)          EXPECT_DIGEST="$2"; shift 2 ;;
        --netrc)                  NETRC="$2"; shift 2 ;;
        --*) echo "!!! unknown option: $1" >&2; echo "!!! $USAGE" >&2; exit 1 ;;
        *)   break ;;
    esac
done

ARM="${1:?$USAGE}"

SELF_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
ROOT="$ROOT_DEFAULT"
REPO="${REPO_DEFAULT:-$ROOT/views-models}"
RUNNER="${RUNNER:-$SELF_DIR/pod_run_model.sh}"
SLICER="${SLICER:-$SELF_DIR/slice_gate_posterior.py}"
OUT="$ROOT/deliver/$ARM"
PROVENANCE="$REPO/models/$ARM/data/raw/calibration_datafactory_df.parquet.provenance.json"

mkdir -p "$OUT"
EXP_LOG="$OUT/exp02_arm.log"
exec > >(tee -a "$EXP_LOG") 2>&1

say()  { echo "=== [$(date -u +%H:%M:%S)] $*"; }
fail() {
    printf 'FAILED:%s\n' "$1" > "$OUT/EXP02_STATUS"
    shift
    printf '%s\n' "$*" > "$OUT/EXP02_FAILURE"
    echo "!!! $*" >&2
    exit 1
}

rm -f "$OUT/EXP02_STATUS" "$OUT/EXP02_FAILURE" "$OUT/QUARANTINED" "$OUT/EXP02_MANIFEST"
echo "### EXP-02 arm $ARM — started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── preflight ─────────────────────────────────────────────────────────────────────────
say "preflight"

declared=0
for known in "${ARMS[@]}"; do [ "$known" = "$ARM" ] && declared=1; done
[ "$declared" -eq 1 ] || fail preflight \
    "$ARM is not one of the ${#ARMS[@]} declared arms (${ARMS[*]}). The set is the
  experiment's definition and is not inferred from a glob — the *tinyzero* rehearsal arms
  share the pull and the config shape and are deliberately not in it."

[ -d "$REPO" ]   || fail preflight "no views-models checkout at $REPO"
[ -x "$RUNNER" ] || fail preflight "the runner is missing or not executable: $RUNNER"
[ -f "$SLICER" ] || fail preflight "the gate slicer is missing: $SLICER
  ⛔ This is fatal BEFORE training rather than after it. The slice is what makes the arm
  readable, and discovering it is absent after 12.5 h of GPU is discovering it too late."
[ -d "$REPO/models/$ARM" ] || fail preflight \
    "$REPO/models/$ARM does not exist. The four arm directories must be on the branch this
  pod cloned (views-models#685). A pod that clones a tree without its arm trains nothing."

[ -s "$NETRC" ] || fail preflight \
    "$NETRC missing or empty — the datafactory fetch would fail after setup. ⛔ On /root,
  never on /workspace: /workspace silently ignores chmod (register C-154)."

cores=$(nproc 2>/dev/null || echo 0)
if [ "$cores" -gt "$MAX_CORES" ] 2>/dev/null; then
    fail preflight \
    "this box reports $cores cores and the limit is $MAX_CORES.
  pod_run_model.sh:638 archives the posterior with 'zstd -3 -T0' — one thread per core. On a
  128-core pod that died in 2.3 s and left a TRUNCATED 1 MB archive from 182 MB of input
  (#558 addendum), and the runner's own N_DRAWS/N_ARCHIVED guard reported success anyway.
  The runner is read-only by standing instruction, so the bound is enforced on the BOX.
  Rent a box with <= $MAX_CORES cores, or raise --max-cores deliberately and know why."
fi
say "preflight OK — $cores cores, arm declared, runner/slicer/netrc present"

# ── train, in its own process group so the digest gate can stop it ────────────────────
say "train: $RUNNER $ARM"
setsid "$RUNNER" "$ARM" &
RUNNER_PID=$!
# `setsid` puts the runner in its own process group whose id equals its pid, so the gate
# below can signal the whole tree (python, the fetch, the archive) rather than only the
# shell. Without it, killing the shell orphans a GPU process that keeps the box busy.
RUNNER_PGID=$RUNNER_PID

stop_runner() {
    kill -TERM "-$RUNNER_PGID" 2>/dev/null
    sleep 5
    kill -KILL "-$RUNNER_PGID" 2>/dev/null
    # ⚠️ pod_run_model.sh removes $OUT/.lock from an EXIT trap, and bash does NOT run an EXIT
    # trap when the shell dies on an untrapped signal. So the lock it took is ours to clear,
    # or the next invocation for this arm refuses with "another run is in progress".
    rmdir "$OUT/.lock" 2>/dev/null
}

# ── the pull check, as a function because it is needed in two places ─────────────────
# Called from the watcher below, and again after `wait` if the sidecar appeared in the race
# between the liveness check and the runner's exit. A gate skipped because of a timing
# accident is a gate that did not run, so there is one implementation and two callers.
check_pull() {
    python3 - "$PROVENANCE" "$EXPECT_DIGEST" "$EXPECTED_LEVEL" \
                "$EXPECTED_MONTH_FIRST" "$EXPECTED_MONTH_LAST" <<'PY'
import json, sys

path, want_digest, want_level, want_first, want_last = sys.argv[1:6]
try:
    with open(path) as handle:
        prov = json.load(handle)
except Exception as exc:                         # unreadable => refuse, never pass (ADR-025)
    print(f"UNREADABLE\t{type(exc).__name__}: {exc}")
    raise SystemExit(0)

got = {
    "queryset_digest": prov.get("queryset_digest"),
    "level": prov.get("level"),
    "month_first": prov.get("month_first"),
    "month_last": prov.get("month_last"),
}
want = {
    "queryset_digest": want_digest,
    "level": want_level,
    "month_first": int(want_first),
    "month_last": int(want_last),
}
# Every field is compared, not only the digest. The digest alone says "some pull"; it does not
# say WHICH window, and a digest pinned by hand can be pinned to the wrong thing. The window
# fields are independently checkable facts about the same file, so they are checked.
bad = [k for k in want if got[k] != want[k]]
if bad:
    print("MISMATCH\t" + "; ".join(f"{k}: got {got[k]!r} want {want[k]!r}" for k in bad))
else:
    print(f"OK\t{got['queryset_digest']}")
PY
}

# Set by enforce_pull, and ONLY by enforce_pull, when a pull has actually been compared
# against the pin. The manifest prints it and the certification below requires it: a field
# that reports how the gate went must be written by the gate, or it goes stale the first time
# the path through the script changes (feedback: status fields lie).
GATE_RESULT="not-run"

# $1: how the sidecar came to be read, for the log. Refuses, stops the run, and quarantines
# the arm on anything but OK.
enforce_pull() {
    local how="$1" gate_report verdict detail
    gate_report=$(check_pull)
    verdict="${gate_report%%$'\t'*}"
    detail="${gate_report#*$'\t'}"

    if [ "$verdict" != "OK" ]; then
        stop_runner
        {
            echo "arm:       $ARM"
            echo "verdict:   $verdict"
            echo "detail:    $detail"
            echo "read:      $how"
            echo "expected:  $EXPECT_DIGEST"
            echo "           ($EXPECTED_LEVEL, months $EXPECTED_MONTH_FIRST-$EXPECTED_MONTH_LAST)"
            echo "sidecar:   $PROVENANCE"
            echo
            echo "⛔ This arm is NOT eligible for the P-23 sign test. Four arms on two"
            echo "   different pulls is a cross-population comparison, not a within-experiment"
            echo "   one, and nothing downstream would announce it — which is what P-30's"
            echo "   refetch already did to P-27."
        } > "$OUT/QUARANTINED"
        fail digest_gate \
    "the pull this pod fetched is not the pre-registered one.
  $verdict: $detail
  Read $how. See $OUT/QUARANTINED."
    fi
    GATE_RESULT="verified $how"
    say "digest gate OK ($how) — $detail"
}

# ── the digest gate: as early as the data allows, not after the GPU is spent ──────────
# The fetch writes its provenance sidecar when the fetch completes — minutes into a ~12.5 h
# run. Waiting for the run to finish would still protect the EXPERIMENT (a quarantined arm is
# not eligible for the sign test) but would protect none of the money. So it is watched for.
say "digest gate: waiting for $PROVENANCE"
waited=0
gate_verdict="not-reached"
while :; do
    if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
        # The runner exited before writing a sidecar. That is the runner's own failure and it
        # has written its own STATUS/FAILURE; this is not a digest problem and must not be
        # reported as one.
        gate_verdict="runner-exited-first"
        break
    fi
    if [ -f "$PROVENANCE" ]; then
        gate_verdict="present"
        break
    fi
    if [ "$waited" -ge "$FETCH_TIMEOUT_SECONDS" ]; then
        gate_verdict="timeout"
        break
    fi
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
done

if [ "$gate_verdict" = "timeout" ]; then
    stop_runner
    fail digest_gate \
    "no provenance sidecar after ${FETCH_TIMEOUT_SECONDS}s at
    $PROVENANCE
  ⛔ Refusing rather than training on an unverified pull. 'I could not look' is not 'the pull
  is fine' (ADR-025) — and an arm trained on an unknown pull cannot be compared with the
  other three, which is the only thing it is for. The run has been stopped."
fi

if [ "$gate_verdict" = "present" ]; then
    enforce_pull "after ${waited}s of the run, before the GPU was spent"
else
    say "digest gate: $gate_verdict (no sidecar yet)"
fi

# ── wait for the run, then report what the RUNNER said about itself ──────────────────
wait "$RUNNER_PID"
runner_rc=$?
runner_status=$(cat "$OUT/STATUS" 2>/dev/null || echo "<no STATUS written>")
say "runner exited rc=$runner_rc, STATUS=$runner_status"

if [ "$gate_verdict" != "present" ] && [ -f "$PROVENANCE" ]; then
    # The sidecar appeared in the race between the liveness check and the runner's exit, so
    # the gate has not run yet on a file that does exist. It is the authoritative record of
    # what was trained on, so it is checked now. ⚠️ Not re-running this script to get here:
    # a self-exec would redo the preflight, wipe the markers and re-enter the slice, which is
    # three behaviours to reason about where one function call is enough.
    enforce_pull "after the runner exited"
fi

[ "$runner_rc" -eq 0 ] || fail train \
    "the runner exited $runner_rc. Its own reason is in $OUT/FAILURE and $OUT/run.log;
  this script does not restate it, because a second copy of a reason is a second thing to
  go stale."
[ "$runner_status" = "OK" ] || fail train \
    "the runner wrote STATUS=$runner_status. ⚠️ Note it can write OK while the gate is
  missing — that is C-161 and the slice below is the answer to it — but a non-OK STATUS is
  the runner saying its own leg failed."

# ── the gate slice, and it is not optional ───────────────────────────────────────────
# THIS IS THE STRUCTURAL CHANGE. The guide has said "you must slice the gate before you
# destroy the pod" since 2026-10-16 and on 2026-10-26 it was not done. Prose cannot gate a
# teardown; an exit code can.
say "gate slice: $SLICER $ARM"
if ! python3 "$SLICER" "$ARM" --repo "$REPO" --out "$OUT"; then
    fail gate_slice \
    "the gate slice FAILED, so this arm has no readout even though training succeeded.
  ⛔ DO NOT DESTROY THIS POD. Every P-23 endpoint — median |gate - p|, median implied w,
  gate AP, at h=1 and h=36 — is computed from the gate posterior, which the runner's own
  archive does not contain (C-161). The posterior is still on this pod and nowhere else;
  views-hydranet C-317 is that a lost posterior needs a new GPU run.
  Read the slicer's refusal above: it names what it examined."
fi

[ -f "$OUT/gate_endpoints.sha256" ] || fail gate_slice \
    "the slicer exited 0 but $OUT/gate_endpoints.sha256 is absent. A success that left no
  digest behind is not a success this script will certify — the digest is how the operator
  verifies the transfer after rsync, and without it the check at the other end cannot run."

# ── the pull must actually have been verified, not merely not-refused ────────────────
# ⛔ Found by `test_a_clean_run_reports_ok_and_leaves_a_verifiable_gate` 2026-11-02. If the
# runner exits before writing a sidecar AND no sidecar ever appears, every branch above is
# skipped rather than failed — and the run was certified OK with NOTHING having compared the
# pull against the pin. That is "I could not look" rendered as "the pull is fine", which is
# the substitution ADR-025 exists to forbid and which the register holds eight instances of.
#
# The watcher's own timeout covers the case where the runner is still alive; this covers the
# case where it is not. The two together are why the check is on GATE_RESULT — the variable the
# comparison itself sets — rather than on which branch was taken.
[ "$GATE_RESULT" != "not-run" ] || fail digest_gate \
    "nothing ever compared this arm's data pull against the pin.
  The runner exited before writing $PROVENANCE and no sidecar appeared afterwards, so the
  digest gate had no subject. ⛔ Refusing to certify the arm: an arm whose pull is unknown
  cannot be compared with the other three, which is the only thing it is for. Training may
  well have succeeded — see $OUT/STATUS — but this script will not call it eligible."

# ── the manifest ─────────────────────────────────────────────────────────────────────
{
    echo "arm:             $ARM"
    echo "finished:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "runner STATUS:   $runner_status"
    echo "queryset digest: $EXPECT_DIGEST"
    echo "digest gate:     $GATE_RESULT"
    echo "pull window:     $EXPECTED_LEVEL, months $EXPECTED_MONTH_FIRST-$EXPECTED_MONTH_LAST"
    echo "gate endpoints:  h1 and h36, from $OUT/gate_endpoints.tar.zst"
    echo "gate digest:     $(cut -d' ' -f1 < "$OUT/gate_endpoints.sha256")"
    echo "cores:           $cores (limit $MAX_CORES, because the runner archives with zstd -T0)"
    echo
    echo "Verify after rsync:  cd <local dir> && sha256sum -c gate_endpoints.sha256"
} > "$OUT/EXP02_MANIFEST"

echo OK > "$OUT/EXP02_STATUS"
say "EXP02_STATUS: OK — trained, pull verified, gate sliced at both endpoints"
cat "$OUT/EXP02_MANIFEST"
