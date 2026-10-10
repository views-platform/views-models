#!/usr/bin/env bash
# Assemble the files a recipient needs on top of the data itself.
#
#   bundle/assemble.sh <bundle-dir>
#
# The bundle is 2 GB of Parquet and cannot live in git. What CAN live in git
# is the thing that puts the paperwork on top of it, so that a hand-fix does
# not die with the folder it was made in (project_pod_artefact_pattern).
#
# It writes three things and renames one:
#
#   DELIVERY_NOTE.pdf   the built note
#   README.txt          a packing list, generated from the note's own title
#                       page so the two cannot drift apart
#   MANIFEST            -> MANIFEST.txt, so a browser will preview it
#
# Every number in README.txt is MEASURED from the bundle, never asserted. If
# the bundle does not hold what the note says it holds, this refuses to write.
#
# WET: this is the first delivery to need it. Trigger for promoting it to
# ../_template/: the second delivery that ships a data bundle.

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
read -r FILECOUNT ROWCOUNT SAMPLEFILE MODELCOUNT < <(
    "$PYTHON" - "$BUNDLE" <<'PY'
import pathlib, sys
import pyarrow.parquet as pq

EXPECT = sorted(["month_id", "priogrid_id",
                 "pred_lr_ged_sb", "pred_lr_ged_ns", "pred_lr_ged_os"])
root = pathlib.Path(sys.argv[1])
models = sorted(p for p in root.iterdir() if p.is_dir())
if not models:
    sys.exit("no model directories in the bundle")

rows, count, sample = set(), 0, None
for m in models:
    files = sorted((m / "parquet").glob("*.parquet"))
    if not files:
        sys.exit(f"{m.name}: no parquet files")
    for f in files:
        md = pq.read_metadata(f)
        rows.add(md.num_rows)
        if sorted(md.schema.names) != EXPECT:
            sys.exit(f"{f}: unexpected columns {md.schema.names}")
    count += len(files)
    if m.name == "dark_river":
        sample = files[0].name

if len(rows) != 1:
    sys.exit(f"row counts differ across files: {sorted(rows)}")
print(f"{count} {rows.pop()} {sample or 'predictions_calibration_..._00.parquet'} {len(models)}")
PY
) || die "the bundle does not hold what the note describes — nothing written"

[[ "$FILECOUNT" -gt 0 ]] || die "measured zero parquet files"

MODELLIST="$(cd "$BUNDLE" && ls -d */ | tr -d '/' | sort \
             | awk '{printf "%s%s", (NR%3==1?"  ":"  "), $0; if(NR%3==0) printf "\n"} END{if(NR%3) printf "\n"}')"
# LC_ALL=C: the note writes 2,333,448 and 2 GB. A Danish locale would write
# 2 333 448 and 1,9 GB here, and the README would disagree with the note.
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

# --- write ----------------------------------------------------------------
cp -f "$NOTE_DIR/main.pdf" "$BUNDLE/DELIVERY_NOTE.pdf"

for f in "$BUNDLE"/*/MANIFEST; do
    [[ -e "$f" ]] && mv -n "$f" "$f.txt"
done

awk -v title="$TITLE" -v id="$ID" -v sum="$CHECKSUM" -v date="$DATE" \
    -v author="$AUTHOR" -v models="$MODELLIST" -v fc="$FILECOUNT" \
    -v size="$BUNDLESIZE" -v rows="$ROWPRETTY" -v sample="$SAMPLEFILE" \
    -v pages="$PAGECOUNT" -v filesec="$FILESECTION" '
{
    gsub(/@DELIVERYTITLE@/, title);     gsub(/@DELIVERYID@/, id)
    gsub(/@DELIVERYCHECKSUM@/, sum);    gsub(/@DELIVERYDATE@/, date)
    gsub(/@DELIVERYAUTHOR@/, author);   gsub(/@MODELLIST@/, models)
    gsub(/@FILECOUNT@/, fc);            gsub(/@BUNDLESIZE@/, size)
    gsub(/@ROWCOUNT@/, rows);           gsub(/@SAMPLEFILE@/, sample)
    gsub(/@PAGECOUNT@/, pages);         gsub(/@FILESECTION@/, filesec)
    print
}' "$NOTE_DIR/bundle/README.txt.in" > "$BUNDLE/README.txt"

if grep -q '@[A-Z]*@' "$BUNDLE/README.txt"; then
    grep -n '@[A-Z]*@' "$BUNDLE/README.txt" >&2
    die "unsubstituted placeholders left in README.txt"
fi

# No checksum manifest. Nobody in this delivery's audience will run one,
# and an unexplained SHA256SUMS.txt at the top of the folder is a question
# the recipient has to ask. Remove it if an earlier run left one.
rm -f "$BUNDLE/SHA256SUMS.txt"

printf 'assembled %s\n' "$BUNDLE"
printf '  %s models, %s parquet files, %s rows each, %s\n' \
       "$MODELCOUNT" "$FILECOUNT" "$ROWPRETTY" "$BUNDLESIZE"
printf '  README.txt, DELIVERY_NOTE.pdf, %s MANIFEST.txt\n' "$MODELCOUNT"
