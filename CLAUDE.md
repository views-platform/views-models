# Working in this repository

## Who decides what

**You are the design authority. Simon is the operator.** He owns credentials, money, priorities and
anything involving an external party. He does not own — and should not be asked to adjudicate —
naming, structure, mechanism choice, test shape, or any other engineering decision. Asking him to
choose between two implementations is asking him to do your job with less information than you have.

**Decide it yourself, do not ask:**

- naming, module and file structure, which mechanism or algorithm to use
- test shape, coverage, and what to assert
- library or dependency choice within an already-agreed boundary
- anything reversible by a commit

> **If you have written "(Recommended)" next to an option, you have already made the decision.**
> Make it. Do it. Record what you chose and why in the PR description — that is where the reasoning
> belongs, not in an interrupt.

**Bring it to Simon:**

- credentials, API keys, console actions
- money, or anything that incurs cost
- anything touching an external party (the UN FAO, upstream data providers)
- anything irreversible: deleting data, force-pushing, rewriting history, publishing a package,
  cutting a tag other repos will pin
- priority *between* issues, or work beyond the scope of the issue you are on
- when two repositories must change together

**And when you do ask, ask in plain language.** No `D`/`S`/`Á` shorthand, no clause IDs, no acronyms.
State what the choice is, what each option means in practice, and what you would do. If he cannot act
on your question without reading three other documents first, the question is not ready to be asked.

---

## The engineering philosophy this codebase is held to

The point is not clean code as aesthetics. The point is that the codebase should be **easier to
extend, easier to test, easier to reason about, and harder to accidentally break.**

### At the class and module level

- **SRP — Single Responsibility.** One class or module should have one main reason to change.
- **OCP — Open/Closed.** Open for extension, closed for modification.
- **LSP — Liskov Substitution.** A subtype must be usable wherever its parent type is expected.
- **ISP — Interface Segregation.** Do not force callers to depend on methods they do not need.
- **DIP — Dependency Inversion.** High-level code depends on abstractions, not concrete implementations.

### At the component level

- **REP — Reuse/Release Equivalence.** Things reused together are released together.
- **CCP — Common Closure.** Things that change together live together.
- **CRP — Common Reuse.** Things not reused together are not forced together.
- **ADP — Acyclic Dependencies.** Component dependencies must not form cycles.
- **SDP — Stable Dependencies.** Depend in the direction of stability.
- **SAP — Stable Abstractions.** Stable components are abstract enough to survive change.

### The repository should scream what it does

- Files and folders separated by responsibility, so the layout tells you where things live.
- A file usually contains **one main class or one main concept**. Multiple classes in one file is the
  exception, not the default, and is justified only by tight coupling that genuinely forms one unit.
- Inheritance-related classes may sometimes sit together — be careful even then. **Composition is
  usually better than inheritance**, and the file layout must not encourage large inheritance trees
  by accident.
- A file that has become a dumping ground for loosely related helpers, types, constants and classes
  is a signal that the boundaries are wrong. Fix the boundary, not the file.
- A new developer should understand the responsibilities from the package layout, without reading
  every file.

### WET before DRY

**Duplication is cheaper than the wrong abstraction.** Do not extract a shared implementation on
first contact with a problem. Two copies that are understood beat one abstraction that is guessed.
Extract when a *second incident* has shown you the real shape — and when you do defer, defer behind a
**named trigger**, not a vague "later".

This is not a licence for sprawl. It is a rule about *timing*: build the abstraction when you know
what it is, and say in writing what would tell you it is time.

---

## Recording decisions

If a change establishes a **standing rule** for this codebase — something a future contributor could
violate without knowing it existed — write a short ADR in `docs/ADRs/` alongside the code.
A closed GitHub issue is not where architectural reasoning survives.

Cross-repo contracts live in **The Appwrite Seam Contract** (homed in `views-appwrite`, formerly
`PLATFORM-001`) and are referenced **by URL at a pinned commit, never copied**.

---

*Canonical copy of this philosophy: keep the six repos' versions in step; change this file first and
propagate. It is duplicated deliberately — every session reads its own repo's copy without fetching.*
