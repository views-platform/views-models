# Delivery notes

A **delivery note** accompanies a set of data handed to someone outside the team that produced it.
It records what was delivered, how to read it, and what is *not* in it.

It is deliberately not the other two things we write:

| | purpose |
|---|---|
| **pre-release note** | asks a partner to decide something. Built around Decision Points — Question, Action, Recommendation |
| **release note** | documents a specification already locked after that decision |
| **delivery note** | documents a *hand-over of data*. Asks for nothing, locks nothing |

A delivery note has **no fixed section structure.** Deliveries differ, and forcing one shape on all
of them produces padding. What is fixed is the preamble, the title page and the metadata block;
everything below that is chosen per note.

## Using the template

```bash
cp -r _template <delivery-id>
cd <delivery-id>
$EDITOR main.tex          # fill the six \delivery* macros, build the section spine
$EDITOR sections/*.tex    # write the prose
./build.sh
```

`main.tex` owns every `\section` heading and `\input` line; the files in `sections/` carry body
prose only and contain no `\section` commands. Reordering, renaming or dropping a section is
therefore a single-file edit. Delete the stubs you do not use.

The six metadata macros are all required:

| macro | |
|---|---|
| `\deliveryid` | the identifier, e.g. `darts-pgm-calibration-2026-10`. Appears in the page footer |
| `\deliverytitle` | short descriptive title |
| `\deliveryto` | recipient |
| `\deliveryfrom` | author |
| `\deliverydate` | **a fixed date, typed by hand** — see below |
| `\deliverysha` | `git rev-parse --short HEAD` at the time of the delivery |

## Where this came from, and how it differs

The preamble, title-page construction and page furniture are taken from the FSFC–PRIO–VIEWS
Project 02 notes, specifically pre-release note 07 — the most recent and the one carrying the
`tabularx`/`xltabular` table machinery. The editorial conventions below come from the two house
style guides:

- `~/brain/2_projects/fao02/_dev_materials/prerelease_notes/STYLE.md`
- `~/brain/2_projects/fao02/_dev_materials/release_notes/STYLE.md`

Five deliberate deviations from that series, each for a reason:

1. **`\usepackage[T1]{fontenc}` added.** The FAO notes omit it, and default OT1 cannot set `å`,
   `ø`, `š` or `ð`. A table in pre-release note 07 omits nine place names for exactly this reason.
   Our first recipient is Håvard Hegre, so this was not optional.
2. **A fixed `\deliverydate` instead of `\today`.** The FAO notes date themselves at compile time,
   so rebuilding one six weeks later restamps it with the wrong date. A delivery note is a record
   of what was sent on a particular day; it must not drift.
3. **A delivery identifier.** The FAO scheme carries identity only as a hand-maintained sequence
   number in the title. In September a model delivery went out with no identifier at all and we
   have since been unable to establish what was in it. The identifier travels in the page footer so
   that a loose printed page can still be traced.
4. **A git sha on the title page.** Immutable provenance, which a date alone does not give.
5. **A configurable `\author`.** The FAO series hard-codes three names across every note. A
   delivery note may have different authorship and should say so rather than inherit it.

Also added, with no FAO counterpart: `build.sh` and a `.gitignore`. The FAO notes are built by hand
and their build artefacts are committed; we should not copy that.

## Conventions

Inherited from the house style guides and worth following:

- **British spelling throughout** — *modelling*, *behaviour*, *recognise*, *analyse*.
- **En-dashes `--` for compound separators**, not spaced em-dashes.
- **Title Case in headings, sentence case in bodies.**
- **Every table gets a `\caption` and a `\label{tab:...}`.** `booktabs` rules only — `\toprule`,
  `\midrule`, `\bottomrule` — and no vertical rules. `\renewcommand{\arraystretch}{1.2}`, or 1.3
  for dense tables.
- **Leave the bibliography lines commented** unless the note actually cites something; otherwise it
  renders an empty References section. `natbib`, `apalike`, `\citep`. Add a `bibtex` pass to
  `build.sh` if you enable it.

## Traps

- **The `L{}` column type's weights must sum to the column count.** `L{0.75}L{0.40}L{1.85}` across
  three columns sums to 3.00. Get it wrong and the table silently stops filling `\textwidth` —
  there is no error. Narrowing one column means widening another by the same amount.
- **For a table that breaks across pages use `xltabular`, not `tabularx`.** Same column syntax, plus
  `\endfirsthead` / `\endhead` / `\endfoot` / `\endlastfoot`; the `\caption` and `\label` go *inside*
  the environment, with no surrounding `table` float.
- **The title strings appear twice** — once in `\title`/`\author`/`\date`, once in the hand-built
  `titlepage`. Both read from the `\delivery*` macros here, so they cannot drift; do not hard-code
  either copy.
- **`\usepackage[section]{placeins}`** stops floats drifting past a `\section`. Good default, but a
  large table in a short section can push the next heading onto a new page.

## Building

`./build.sh` runs three `pdflatex` passes and then checks the log for errors, overfull boxes and
undefined references, failing if any error or undefined reference remains. Three passes because
cross-references settle on the second and `xltabular` column widths on the third.

**Commit the PDF of any note that has gone out.** `.gitignore` excludes the `.aux`/`.log`/`.out`
family but not the PDF, deliberately: the PDF is the record of what the recipient actually received,
and reading `sections/*.tex` later is not the same as reading what was transmitted.
