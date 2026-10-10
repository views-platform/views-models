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

### Section 1 is always Context

Everything below it varies; the opening section does not. It takes its shape from the FAO release
notes and makes five moves in order:

1. what this document is, and which delivery it accompanies;
2. its purpose, as a short list;
3. what it covers, cross-referenced to the sections;
4. **what it explicitly does *not* cover**; and
5. a closing statement that the delivery is a fixed artefact, identified by its identifier and
   checksum, not revised in place.

The fourth move is the one that earns its keep. A reader will otherwise assume that anything
plausibly related is in scope — that the folder contains the observed data, or that someone has
already evaluated the models. Say what is out of scope before they find out by looking.

### Naming sections

Each title should say what is in the section, and no two should overlap. An earlier draft of the
darts note ran *What These Models Are* / *The Models* / *The Model Types* as three consecutive
headings, which told a reader nothing about where to look for what.

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

**Scope** — what the data is. Resolution and extent are stated separately on each axis, and in
parallel: a reader should be able to see the shape of the dataset without reading prose.

| macro | |
|---|---|
| `\deliverypartition` | `Calibration`, `Validation` or `Forecasting` |
| `\deliveryspatialres` | the spatial unit — `PRIO-GRID cell`, `Country` |
| `\deliveryspatialextent` | which units, with a count — `Global land — 64,818 cells` |
| `\deliverytemporalres` | the temporal unit — `Month` |
| `\deliverytemporalextent` | which periods: the forecast span, and the training span |
| `\deliverystructure` | the forecast design — rolling origins and horizon |
| `\deliverytargets` | the outcome variables |

The level-of-analysis shorthand (`pgm`, `cm`) is not a field. It is the product of the two
resolutions and already the second component of the identifier; repeating it alongside them is what
made an earlier version of this block incoherent — it listed "unit of analysis" against "temporal
resolution", which are not the same kind of thing.

**Provenance** — what produced it:

| macro | |
|---|---|
| `\deliverylibrary` | model library and version, e.g. `views-r2darts2 0.2.4` |
| `\deliverybackend` | where the input data came from, **with its version**, e.g. `views-datafactory 1.13.0` |
| `\deliveryframework` | the pipeline framework and version, e.g. `views-pipeline-core 3.3.4`. It decides the partitions, the rolling origins and how predictions are saved, so it belongs beside the model library |
| `\deliverysha` | the views-models commit **the data was produced at** — not the commit of the note. The `MANIFEST` beside each model's output records the sha its config was read at; use the latest commit that touched any delivered model's configuration. It **must be pushed**, or the config links will not resolve |
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

## Linking to configurations

`\cfg{model_name}` and `\qs{model_name}` render the word *link*, pointing at a model's
`configs/config_hyperparameters.py` and `configs/config_queryset.py` on GitHub, pinned to
`\deliverysha`. They let a table of models double as an index into the exact configuration each
one was run from — the question a reader has after seeing that two models differ in one setting.

Give them their own columns rather than hyperlinking the model names. A linked name is not
discoverable as a link by a reader who is not looking for one, and the people who click these are
not the people the note is addressed to.

Two things to know. Underscores in model names are mathematics to LaTeX, so wrap any table using
`\cfg` in a group that makes them ordinary:

```latex
\begingroup
\catcode`\_=12
\begin{table}[h!]
...
\endgroup
```

And **check the links resolve before sending.** A pinned link to an unpushed commit returns 404.
Extract them from the built PDF and test them:

```bash
pdftk main.pdf output /tmp/u.pdf uncompress
for u in $(grep -ao 'https://github.com/[^)]*' /tmp/u.pdf | sort -u); do
    printf '%s  %s\n' "$(curl -s -o /dev/null -w '%{http_code}' -L "$u")" "$u"
done
```

## Conventions

Inherited from the house style guides and worth following:

- **British spelling throughout** — *modelling*, *behaviour*, *recognise*, *analyse*.
- **En-dashes `--` for compound separators**, not spaced em-dashes.
- **Title Case in headings, sentence case in bodies.**
- **Every table gets a `\caption` and a `\label{tab:...}`.** `booktabs` rules only — `\toprule`,
  `\midrule`, `\bottomrule` — and no vertical rules. `\renewcommand{\arraystretch}{1.2}`, or 1.3
  for dense tables.
- **Citations go in `references.bib` and render as a References section at the end** — never as
  inline blocks in the body, which interrupt the argument and scatter the same reference across
  sections. `natbib` with `apalike`: `\citep{key}` for a parenthetical, `\citet{key}` when the
  work is the subject of the sentence. `build.sh` runs `bibtex` automatically when the
  `\bibliography` line is uncommented and the document actually cites something, and fails on an
  undefined citation.
- **Leave the bibliography lines commented if the note cites nothing**, and delete
  `references.bib`; an empty References section is worse than none.

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

## Shipping the data with the note

A note that goes out alongside a data bundle needs two things on top of the
data, and the recipient needs both before the first file makes sense:

| file | why |
|---|---|
| `DELIVERY_NOTE.pdf` | the note itself, travelling with what it describes |
| `README.txt` | a packing list — what is in the folder, and how to open one file |

`bundle/assemble.sh <bundle-dir>` writes both and renames each model's
`MANIFEST` to `MANIFEST.txt` so a browser will preview it.

**No checksum manifest.** An earlier version of this shipped `SHA256SUMS.txt`
and cross-platform verification instructions. Nobody in this audience will
run `sha256sum -c`, and an unexplained file at the top of the folder is a
question the recipient has to ask rather than an assurance. The prior
HydraNet delivery solved the same problem better and is the pattern to copy
if a delivery needs one: a notebook that audits the folder it sits in, with
its outputs already saved so it can be read without running anything, ending
in a cell that raises rather than prints
(`/home/simon/Documents/views_selector_frames_20260929/`).

**Nothing in `README.txt` is written by hand.** The identity fields come from
the note's own `\delivery*` macros, and every figure — file count, row count,
total size, the note's page count, the number of the section that defines the
columns — is measured from the bundle or read from the built PDF. This is
deliberate: a packing list that restates the note's numbers is a second copy
of them, and the two copies drift. The script refuses to write anything if the
bundle does not hold what the note describes.

The generated README names the note as authoritative where the two disagree.
That is a backstop, not a licence — if they disagree, the generator is broken.

`tests/test_delivery_bundle_assembly.py` holds this, including the cases that
matter: a bundle with the wrong columns, and one whose files disagree on row
count, must both be refused rather than described.

## What the recipient cannot work without

A delivery at PRIO-GRID carries `priogrid_id` and no coordinates. Without the
conversion to latitude and longitude the recipient cannot place a single row
on a map, so a pgm delivery note states it, with a worked example they can
check their own arithmetic against. Guarded in
`tests/test_delivery_note_is_readable.py`.

**WET:** `bundle/` lives in the delivery it was written for, not in
`_template/`. Trigger for promoting it: the second delivery that ships a data
bundle.

## Writing for this recipient

Four rules came out of the mail that accompanied the 2026-10 delivery, after
four drafts were rejected. Three of the rejections were corrections to how the
text was written rather than to what it said, so they apply to the next note as
much as to the next mail.

**No lateness line, anywhere.** A delivery that also asks whether anyone has
looked at the previous one cannot open by conceding that this one is late. The
concession hands the recipient the frame, and the question then reads as
deflection. Open on the files.

**Anything inside our remit reads as a decision, not a request.** Column
naming, which collapse to use, what the partition means — these are ours.
Putting them to the recipient as questions asks him to adjudicate engineering,
and he will. Section 9 (Next Steps) is where this goes wrong most easily: write
it as statements plus, where needed, a documentation requirement. The sent mail
has no question marks at all.

**Where the recipient owns the subject, state the inference and stop.** He
knows GED better than anyone on the distribution list. Three sentences
explaining how UCDP codes aggregated events are three sentences of
junior-explaining. "I read it as the Tigray war arriving as one aggregated
event" is the whole move. A delivery note has more licence than a mail —
Sections 1 and 8 exist to give context — but not unlimited.

**Concision is not compression.** Asked to tighten, the draft went 880 → 552
words by deleting connective tissue and produced verdict-stacked telegraphese
with no actors and repeated topic labels. The sent version is 624. The win is
deleting one redundant *move*, never the words that join the remaining ones.

Two things the 2026-10 mail does **not** say, which the note does, and which
therefore live only here:

- That the nine are deterministic and no draws were retained. In the mail by
  Simon's decision ("I'll tell him if he asks"). It is in Section 8, so it is
  on the record. Consequence: the first request for any HydraNet summary other
  than the mean, q95 or E[y|y>0] will arrive without the asker expecting a GPU
  re-run. Answer that straight.
- Tigray went out hedged as "I read it as", pending confirmation from the
  datafactory session against the actual GED records. If that confirmation
  contradicts the reading, the mail needs no retraction — but this note's
  Section 8 must match whatever comes back.

**The Drive URL belongs in the record.** September's mail kept its folder link
in front-matter and that has been useful since. The front-matter of
`~/brain/0_inbox/comms/mail/_sent/2026-10-10_havard_darts_calibration_nine_SENT.md`
records that the URL replacing `<<<DRIVE LINK>>>` was not captured. Put it there
and in the delivery's own record when it is known.
