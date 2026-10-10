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

## The identifier

Every delivery note carries an identifier, composed from the facts on its own title page rather
than hashed into an opaque number. A hashed id needs a lookup table to mean anything, and the
lookup table is the thing that goes missing.

```
<library>-<loa>-<scope>-<partition>-<YYYY-MM>[-<n>]

r2darts2-pgm-global-calibration-2026-10
```

| field | |
|---|---|
| `library` | the model library that produced the forecasts — `r2darts2`, `hydranet`, `stepshifter` |
| `loa` | the VIEWS level of analysis: `pgm` (PRIO-GRID month) or `cm` (country month). This carries **both** the spatial unit and the temporal resolution |
| `scope` | the spatial extent — `global`, `africa`, … **Always present.** A delivery is not assumed global |
| `partition` | `calibration`, `validation` or `forecasting` |
| `YYYY-MM` | month of delivery |
| `n` | sequence number, **only** when a second delivery with the same fields goes out in the same month |

The identifier is also the directory name, so the repository tree sorts by library, then level,
then scope, then partition, then date.

## The checksum

The identifier says what a delivery **is**. The checksum says it has not **drifted**.

`checksum.py` derives an eight-digit number from every specification field on the title page —
identifier, library, level, scope, resolution, partition, targets, backend, compute environment,
source commit and date. Change any one of them and the number changes. It is computed, never
chosen:

```bash
python3 ../checksum.py .          # print it
python3 ../checksum.py . --check  # fail if main.tex disagrees
```

`build.sh` runs the check, so a note cannot quietly come to disagree with its own specification —
the failure this whole scheme exists to prevent.

Two design points. LaTeX markup is stripped before hashing, because `\texttt{pgm}` and `pgm`
describe the same delivery and must not produce different numbers. And the field *order* is part
of the definition: appending a field changes every checksum computed afterwards, which is correct,
because it is then a different specification.

Eight decimal digits gives ~10⁸ values. For a project producing tens of deliveries a year the
collision risk is negligible, and the number stays short enough to read aloud.

**Quote the pair.** `r2darts2-pgm-global-calibration-2026-10 / 26493508` names a delivery and
proves the note still describes it. The slug alone is readable but can be reused after an edit;
the checksum alone is unambiguous but says nothing.

## What the provenance block does and does not pin

Three versions plus a commit — model library, framework, data backend, and the `views-models` sha —
fix the **code** that produced a delivery. They do not fix the **input**. Our data backend serves
from a live store, and a delivery note should say so rather than let a reader infer that a re-run
would reproduce the numbers. Record what you can; state the gap in the note body.

## Title-page fields

All of these are required. They are set once, as macros at the top of `main.tex`, and the title
page and page furniture read from them — so nothing can drift between two copies of the same fact.

| macro | |
|---|---|
| `\deliveryid` | the identifier above. Also printed in every page footer |
| `\deliverytitle` | short descriptive title |
| `\deliveryauthor` | who wrote the note |
| `\deliverydate` | **a fixed date, typed by hand** — see below |

The remaining fields sit in two blocks, and the line between them is a test rather than a
judgement call:

> **Scope** is what the data *is*. Every field could be written down before anything was run.
> **Provenance** is what *produced* it. No field exists until it has run.

That is why the model library, the framework and the data backend are all under Provenance even
though the library also appears in the identifier: they are tools, and a tool is not a property of
the data. An earlier version split them — library and backend under Scope, framework under
Provenance — which had no rule behind it and did not read.

**Scope** — what the data is:

| macro | |
|---|---|
| `\deliveryloa` | unit of analysis |
| `\deliveryscope` | spatial extent, with the cell or country count |
| `\deliveryresolution` | temporal resolution |
| `\deliverypartition` | partition, with the training and forecast spans in calendar dates |
| `\deliverytargets` | the outcome variables |

**Provenance** — what produced it:

| macro | |
|---|---|
| `\deliverylibrary` | model library and version, e.g. `views-r2darts2 0.2.4` |
| `\deliverybackend` | where the input data came from, **with its version**, e.g. `views-datafactory 1.13.0` |
| `\deliveryframework` | the pipeline framework and version, e.g. `views-pipeline-core 3.3.4`. It decides the partitions, the rolling origins and how predictions are saved, so it belongs beside the model library |
| `\deliverysha` | `git rev-parse --short HEAD` in views-models at delivery time. This pins the exact model configs, runner and converter |
| `\deliverycompute` | where it was computed — own hardware, a named server, or rented cloud GPUs. Name the thing; do not add what it was *not*, which is implied. Unrecoverable later if not recorded, and it has already mattered: a rented machine's network filesystem once cost a hundred minutes of a run |
| `\deliverychecksum` | **computed, not chosen** — see above |

There is deliberately **no "To" field**. A delivery note describes a delivery, not a letter; the
same note may go to several people, and addressing it to one of them dates it badly.

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
5. **A configurable author, and no recipient field.** The FAO series hard-codes three names
   across every note. A delivery note may have different authorship and should say so rather than
   inherit it — and it names no recipient at all, because it describes a delivery rather than
   addressing a letter.
6. **A structured title-page spec block** — library, level of analysis, spatial scope, temporal
   resolution, partition, targets, source commit, compute environment. The FAO notes carry none of
   this; their subject is a specification under negotiation, ours is a dataset someone has to be
   able to identify later.

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
