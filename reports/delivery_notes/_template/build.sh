#!/usr/bin/env bash
# Build a delivery note. Three pdflatex passes: cross-references settle
# on pass two, xltabular column widths on pass three.
#
# Not inherited from the FAO notes --- those are built by hand. Added
# here because a note in a repository should build the same way for
# whoever picks it up next.
set -uo pipefail
cd "$(dirname "$0")"

for i in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null || {
        echo "FAILED on pass $i — see main.log"; grep -m5 '^!' main.log; exit 1; }
done

# The checksum is derived from the title-page specification fields, not
# chosen. If it disagrees, a field changed and the note no longer matches
# the delivery it claims to describe.
if [ -f ../checksum.py ]; then
    python3 ../checksum.py . --check || {
        echo "  regenerate with: python3 ../checksum.py ."; exit 1; }
fi

errors=$(grep -c '^!'              main.log || true)
overfull=$(grep -c 'Overfull \\hbox' main.log || true)
undef=$(grep -ci 'undefined'       main.log || true)

printf 'built main.pdf  (%s pages)\n' "$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}')"
printf '  errors            %s\n  overfull hboxes   %s\n  undefined refs    %s\n' \
       "$errors" "$overfull" "$undef"
[ "$errors" = 0 ] && [ "$undef" = 0 ] || { echo "  ^ not clean"; exit 1; }
