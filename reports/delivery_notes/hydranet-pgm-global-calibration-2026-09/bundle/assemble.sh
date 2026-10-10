#!/usr/bin/env bash
# Assemble the files a recipient needs on top of the data itself.
#
#   bundle/assemble.sh <bundle-dir>
#
# The bundle is 180-odd MB of Parquet and cannot live in git. What CAN live in
# git is the thing that puts the paperwork on top of it, so that a hand-fix
# does not die with the folder it was made in (project_pod_artefact_pattern).
#
# It writes two things:
#
#   DELIVERY_NOTE.pdf   the built note
#   README.txt          a packing list, generated from the note's own title
#                       page so the two cannot drift apart
#
# and removes the hand-written README.md the folder was first uploaded with,
# which this supersedes.
#
# Every number in README.txt is MEASURED from the bundle, never asserted. If
# the bundle does not hold what the note says it holds, this refuses to write.
#
# A SECOND COPY, DELIBERATELY. The darts delivery's assembler is the sibling
# of this one, and the recorded trigger said to promote it to ../_template/
# on the second data bundle --- which is this one. Not done, because the two
# bundles are not the same shape: <model>/parquet/ against
# <model>/<frame>/, different column names, nine per-model manifests against
# none, and a verification notebook here that the darts bundle does not have.
# Sharing now means a config block per delivery inside one script, on the path
# of every later delivery. Two copies that are understood beat one that is
# guessed. New trigger: a THIRD bundle, or a second bundle that shares the
# <model>/parquet/ layout.

set -uo pipefail

NOTE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="${1:-}"
PYTHON="${BUNDLE_PYTHON:-/home/simon/anaconda3/envs/views_pipeline/bin/python}"

die() { printf 'assemble: %s\n' "$*" >&2; exit 1; }

[[ -n "$BUNDLE" ]]    || die "usage: bundle/assemble.sh <bundle-dir>"
[[ -d "$BUNDLE" ]]    || die "no such directory: $BUNDLE"
BUNDLE="$(cd "$BUNDLE" && pwd)"
[[ -f "$NOTE_DIR/main.pdf" ]] || die "main.pdf not built — run ./build.sh first"
[[ -x "$PYTHON" ]]    || die "no python at $PYTHON (set BUNDLE_PYTHON)"

# --- the note's title page is the single source for the identity fields ----
macro() {
    local v
    v="$(sed -n "s/^\\\\newcommand{\\\\$1}{\(.*\)}$/\1/p" "$NOTE_DIR/main.tex")"
    [[ -n "$v" ]] || die "\\$1 not found in main.tex"
    printf '%s' "$v"
}
ID="$(macro deliveryid)"             || exit 1
CHECKSUM="$(macro deliverychecksum)" || exit 1
DATE="$(macro deliverydate)"         || exit 1
AUTHOR="$(macro deliveryauthor)"     || exit 1
TITLE="$(macro deliverytitle)"       || exit 1

# --- measure the bundle; refuse on anything inconsistent ------------------
read -r FILECOUNT ROWCOUNT SAMPLEFILE MODELCOUNT ORIGINCOUNT < <(
    "$PYTHON" - "$BUNDLE" <<'PY'
import pathlib, sys
import pyarrow.parquet as pq

EXPECT = sorted(["month_id", "priogrid_id",
                 "pred_lr_sb_best", "pred_lr_ns_best", "pred_lr_os_best"])
# The frame dimension is the shape of THIS delivery, so it is checked and not
# discovered: a model missing a frame, or a frame missing an origin, is the
# failure this script exists to catch before the folder goes out again.
FRAMES = ("I_mean", "II_q95", "III_conditional_mean")
ORIGINS = 13
root = pathlib.Path(sys.argv[1])
models = sorted(p for p in root.iterdir() if p.is_dir())
if not models:
    sys.exit("no model directories in the bundle")

rows, count, sample = set(), 0, None
for m in models:
    found = sorted(p.name for p in m.iterdir() if p.is_dir())
    if found != sorted(FRAMES):
        sys.exit(f"{m.name}: frame directories are {found}, expected {sorted(FRAMES)}")
    for frame in FRAMES:
        files = sorted((m / frame).glob("*.parquet"))
        if len(files) != ORIGINS:
            sys.exit(f"{m.name}/{frame}: {len(files)} parquet files, expected {ORIGINS}")
        for f in files:
            md = pq.read_metadata(f)
            rows.add(md.num_rows)
            if sorted(md.schema.names) != EXPECT:
                sys.exit(f"{f}: unexpected columns {md.schema.names}")
        count += len(files)
    if m.name == "purple_alien":
        sample = sorted((m / "I_mean").glob("*.parquet"))[0].name

if len(rows) != 1:
    sys.exit(f"row counts differ across files: {sorted(rows)}")
print(f"{count} {rows.pop()} {sample or 'predictions_calibration_..._00.parquet'} {len(models)} {ORIGINS}")
PY
) || die "the bundle does not hold what the note describes — nothing written"

[[ "$FILECOUNT" -gt 0 ]] || die "measured zero parquet files"

MODELLIST="$(cd "$BUNDLE" && ls -d */ | tr -d '/' | sort \
             | awk '{printf "%s%s", (NR%3==1?"  ":"  "), $0; if(NR%3==0) printf "\n"} END{if(NR%3) printf "\n"}')"
# LC_ALL=C: the note writes 2,333,448 and 182 MiB. A Danish locale would write
# 2 333 448 here, and the README would disagree with the note.
BUNDLESIZE="$(LC_ALL=C du -sh --apparent-size "$BUNDLE" | cut -f1 | sed 's/G/ GB/; s/M/ MB/')"
# Grouped with sed rather than printf %'d, which follows the locale and would
# put spaces in a Danish one.
ROWPRETTY="$(printf '%s' "$ROWCOUNT" | sed -e :a -e 's/\([0-9]\)\([0-9]\{3\}\)\($\|,\)/\1,\2\3/;ta')"

# The note's page count and the number of the section that defines the
# columns both change when the note is edited. Read them, do not restate
# them: a README that misdirects the reader is worse than one that is terse.
PAGECOUNT="$(pdfinfo "$NOTE_DIR/main.pdf" | awk '/^Pages:/{print $2}')"
[[ -n "$PAGECOUNT" ]] || die "could not read the page count from main.pdf"
FILESECTION="$(awk '/^\\section/{n++} /\\label\{sec:files\}/{print n; exit}' "$NOTE_DIR/main.tex")"
[[ -n "$FILESECTION" ]] || die "could not locate \\label{sec:files} in main.tex"
FRAMESECTION="$(awk '/^\\section/{n++} /\\label\{sec:frames\}/{print n; exit}' "$NOTE_DIR/main.tex")"
[[ -n "$FRAMESECTION" ]] || die "could not locate \\label{sec:frames} in main.tex"

# The notebook is this delivery's verification artefact, in place of the
# checksum manifest nobody runs. If it is not in the folder, the README must
# not tell the recipient to open it.
[[ -f "$BUNDLE/audit_selector_frames.ipynb" ]] \
    || die "audit_selector_frames.ipynb is not in the bundle — the README points at it"

# --- write ----------------------------------------------------------------
cp -f "$NOTE_DIR/main.pdf" "$BUNDLE/DELIVERY_NOTE.pdf"

awk -v title="$TITLE" -v id="$ID" -v sum="$CHECKSUM" -v date="$DATE" \
    -v author="$AUTHOR" -v models="$MODELLIST" -v fc="$FILECOUNT" \
    -v size="$BUNDLESIZE" -v rows="$ROWPRETTY" -v sample="$SAMPLEFILE" \
    -v pages="$PAGECOUNT" -v filesec="$FILESECTION" -v framesec="$FRAMESECTION" \
    -v modelcount="$MODELCOUNT" -v origincount="$ORIGINCOUNT" '
{
    gsub(/@DELIVERYTITLE@/, title);     gsub(/@DELIVERYID@/, id)
    gsub(/@DELIVERYCHECKSUM@/, sum);    gsub(/@DELIVERYDATE@/, date)
    gsub(/@DELIVERYAUTHOR@/, author);   gsub(/@MODELLIST@/, models)
    gsub(/@FILECOUNT@/, fc);            gsub(/@BUNDLESIZE@/, size)
    gsub(/@ROWCOUNT@/, rows);           gsub(/@SAMPLEFILE@/, sample)
    gsub(/@PAGECOUNT@/, pages);         gsub(/@FILESECTION@/, filesec)
    gsub(/@FRAMESECTION@/, framesec)
    gsub(/@MODELCOUNT@/, modelcount); gsub(/@ORIGINCOUNT@/, origincount)
    print
}' "$NOTE_DIR/bundle/README.txt.in" > "$BUNDLE/README.txt"

if grep -q '@[A-Z]*@' "$BUNDLE/README.txt"; then
    grep -n '@[A-Z]*@' "$BUNDLE/README.txt" >&2
    die "unsubstituted placeholders left in README.txt"
fi

# The hand-written README.md this folder went out with is superseded by the
# note and by the generated README.txt. Two readmes disagreeing is worse than
# one that is generated, so it goes. A copy is kept in views-models beside
# this script, as bundle/README_as_uploaded_2026-09-29.md.
rm -f "$BUNDLE/README.md"

# No checksum manifest. Nobody in this delivery's audience will run one, and
# an unexplained SHA256SUMS.txt at the top of the folder is a question the
# recipient has to ask. The audit notebook is the verification artefact here.
rm -f "$BUNDLE/SHA256SUMS.txt"

printf 'assembled %s\n' "$BUNDLE"
printf '  %s models x 3 frames, %s parquet files, %s rows each, %s\n' \
       "$MODELCOUNT" "$FILECOUNT" "$ROWPRETTY" "$BUNDLESIZE"
printf '  README.txt, DELIVERY_NOTE.pdf, audit_selector_frames.ipynb\n'
