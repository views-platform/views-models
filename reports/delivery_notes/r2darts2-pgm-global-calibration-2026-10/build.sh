#!/usr/bin/env bash
# Build a delivery note. Three pdflatex passes: cross-references settle
# on pass two, xltabular column widths on pass three.
#
# Not inherited from the FAO notes --- those are built by hand. Added
# here because a note in a repository should build the same way for
# whoever picks it up next.
set -uo pipefail
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null || {
    echo "FAILED on pass 1 — see main.log"; grep -m5 '^!' main.log; exit 1; }

# bibtex only if the document actually cites something
if grep -q '^\\bibliography{' main.tex && grep -q '\\citation{' main.aux 2>/dev/null; then
    bibtex main >/dev/null 2>&1 || { echo "bibtex FAILED — see main.blg"; exit 1; }
fi

for i in 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null || {
        echo "FAILED on pass $i — see main.log"; grep -m5 '^!' main.log; exit 1; }
done

errors=$(grep -c '^!'              main.log || true)
overfull=$(grep -c 'Overfull \\hbox' main.log || true)
undef=$(grep -ci 'undefined'       main.log || true)
cites=$(grep -c 'Citation.*undefined' main.log || true)

printf 'built main.pdf  (%s pages)\n' "$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}')"
printf '  errors            %s\n  overfull hboxes   %s\n  undefined refs    %s\n  undefined cites   %s\n' \
       "$errors" "$overfull" "$undef" "$cites"
[ "$errors" = 0 ] && [ "$undef" = 0 ] || { echo "  ^ not clean"; exit 1; }
