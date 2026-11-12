# ADR-020: Errors must descend — and we must say where the stairs end

**Status:** **Accepted** (2026-08-04)
**Date:** 2026-08-04
**Deciders:** Simon (maintainer)
**Origin:** extracted from ADR-017 §13 when that document was split for containment. The scope is
**this repository**, not delivery — which is why it is its own ADR rather than a section inside one.

---

> **A note on names.** Every model, ensemble, consumer, region and target named in this document is an
> **example**. They are real names where possible, because concrete examples are easier to read than
> placeholders — but which source feeds which consumer changes, consumers are added and retired, and
> buckets get renamed. Nothing here is a declaration about a particular name. The rules are about the
> **shape**; the names are illustration.

---

## Summary

**The problem, in one breath.** A person who edits a config in this repository and gets it wrong is
told *what* failed, and left to work out *where* to go. That works if you already know the system. It
is the reason turning the FAO delivery on in July 2026 was hard — not because anything failed
silently, but because *how* to do it was undiscoverable to anyone who had not built it.

**The decision, in one breath.** When a config is wrong, the error sends the reader **exactly one
level down**, naming the next file to open. Where the reader cannot go any further, the error says so
and names a person — it never ends in a task they cannot perform.

---

## 1. Who this is for

The person editing a config in this repository is, realistically, a research assistant with a
social-science background who is good at Python *compared to social scientists*. They will only ever
edit **this** repository. They cannot publish a package, edit the platform coordinate registry, or
open a pull request against an API repo.

Designing error messages for anyone else is designing for someone this project does not have.

This is not a claim about any individual's ability. It is a claim about **what the work actually is**:
the repository is 131 model directories maintained by a research group with no dedicated ops engineer,
and the person holding the task on any given month is whoever has time.

## 2. The decision

**Every error a human reads from a config in this repository must name the next file to open.**

Not the failing value. Not the exception type. The **file**, and where in it.

Concretely — taking the delivery declaration (ADR-019) as the worked case, with example names —
the staircase is:

| what is wrong | where the error sends you |
|---|---|
| `pgm("skinny_love")` but it declares `cm` | `ensembles/skinny_love/configs/config_meta.py` |
| `reconciled = True` but the partner is someone else | the same file — `reconcile_with` is on the next line |
| `reconcile_with` names an ensemble that does not exist | `ensembles/` — and the closest name to what was typed |
| an active ensemble contains a retired member (ADR-017 R1) | `config_modelset.py`, then `models/<member>/configs/` |

Four steps, each exactly one level down, no loops, no sideways jumps.

**Error messages are load-bearing architecture here, not politeness.** They are also the least-tested
thing in most codebases, which is why the rule below exists.

## 3. Error messages are tested

For each failure class, a test asserts that the message **names the next file**. Without this the
staircase rots in its second month — someone refactors a check, the message becomes
`KeyError: 'level'`, and nothing fails.

This repository already does this in three places, so the pattern is established rather than proposed:

- `tests/test_reconciliation_skip_is_truthful.py` — asserts the guard ordering that makes a skip
  truthful, with a failure message naming the fix.
- `tests/test_run_sh_portability.py` — *"If a newly scaffolded model appears here, fix the template in
  views-pipeline-core, not the copy."*
- `bootstrap.sh` — *"Activate an environment with Python 3.11+ and re-run. Every run.sh builds its own;
  `conda activate` any of them, or use the base 3.11."*

## 4. Where the stairs end

Some checks cannot be answered inside this repository. Pretending otherwise would be the same
dishonesty as a config whose docstring says it does nothing while deciding what reaches the UN.

Four boundaries, as of 2026-08-04. Three are locked doors; one is a hole in the floor.

- **Coverage** — the cell counts defining a region live in views-postprocessing, beside the GAUL
  asset. They belong there.
- **Target names** — checked against the manifests of a real run in the shelf. Not a file; a network
  resource. *(And it cannot move earlier until a source's config truthfully describes what it emits:
  `rusty_bucket` declares `lr_*_best` and produces `lr_ged_*` — register C-123. An edit-time check
  today would reject a **correct** file, and the first thing the repository would teach a newcomer is
  that its errors are wrong.)*
- **A genuinely new consumer** — needs a bucket in the platform coordinate registry (views-appwrite)
  and a producer package (views-postprocessing).
- **A declared-live delivery that nothing ever runs** — no error exists, because nothing failed. This
  is the hole in the floor, and it is the failure that actually happened: 145 days of silence while a
  complete forecast sat unshipped (#320, C-121). No message can fix it; only an assertion about
  freshness and a report of derived status can (ADR-019).

## 5. What an error must do at a locked door

Name the person, supply the request, and confirm the rest of the work is fine.

```
un_ocha is not a registered consumer.

  Registering one needs a bucket address from the platform coordinate registry,
  which is in another repository you are not expected to edit.

  Ask Simon, or open an issue: "Register consumer un_ocha (bucket + API)".
  Everything else in this file is fine — this is the only thing blocking it.
```

That last line is the difference between a handoff and a dead end. A locked door that also tells you
the rest of your work is correct is a good place to stop. One that does not is where people give up
and ask someone else to do it for them — which is how delivery became undiscoverable in the first
place (ADR-017 §2).

## 6. Rationale (against the maintainer's principles)

- **Screaming architecture** — a repository screams what it does through its failures as much as its
  folder names. An error that names the next file *teaches the layout* at the moment someone needs it,
  without them reading a 585-line README first.
- **Fail loud (ADR-003)** — this ADR does not change *whether* we fail loudly; it constrains *what the
  loud thing says*.
- **Easier to reason about, harder to accidentally break** — the descent is a property a reviewer can
  check by reading a message, and a test can check mechanically.

## 7. Consequences

**Positive:** a newcomer can repair their own mistakes without a guide; the layout teaches itself;
"where do I go next?" stops being tribal knowledge.

**Negative:** error text becomes something we maintain and test, which is real ongoing cost. Messages
naming files couple the message to the layout — moving a file means updating messages, and the tests
in §3 are what make that a failure rather than a slow rot.

**Known limit:** §4's four boundaries are not fixed by this ADR. Three are correctly outside this
repository; the fourth needs ADR-019's freshness rule. This ADR's contribution there is only that we
**say so** rather than leaving a reader to discover it.

## 8. Considered alternatives

- **Document the layout better instead.** Rejected: prose rots silently, and the person who needs it
  is the least equipped to notice it is stale. This repository's own README is 585 lines and entirely
  about building a model; it says nothing about delivery.
- **A single "troubleshooting" page.** Rejected for the same reason, plus it puts the answer somewhere
  the reader must already know to look. The error is where they already are.
- **Richer exception types instead of message text.** Rejected: the audience does not read exception
  hierarchies. They read the last line of a traceback.

## References

- **ADR-017** — the three axes; its R1/R2 rules produce two of §2's descents.
- **ADR-019** — the delivery declaration; the worked staircase in §2 is its file.
- **ADR-003** — authority and fail-loud.
- `docs/forecast_delivery_map.md` — what the boundaries in §4 actually are today.
- Register: **C-121** (the hole in the floor), **C-123** (why the target check cannot descend yet),
  **C-125** (the same, as a pedagogical cost).
