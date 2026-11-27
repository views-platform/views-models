# Working in this repository

## Who decides what

The global CLAUDE.md sets the split: you decide engineering; Simon gets credentials, money, outside parties and anything irreversible. In this codebase also:

- Engineering choices (naming, structure, mechanism, tests) are yours: do not offer Simon options on them; decide, and record what and why in the PR description. A choice that is his (the list above and below) still comes to him as options with your recommendation, and he picks; your recommendation is never his answer.
- Outside parties here are the UN FAO and upstream data providers. Irreversible includes publishing a package and cutting a tag other repos will pin.
- Bring Simon priority between issues, work beyond the current issue, and any change where two repositories must change together.

---

## Engineering philosophy

The goal: easier to extend, test and reason about, and harder to break by accident. Use SOLID and the component principles (REP, CCP, CRP, ADP, SDP, SAP) as heuristics, not laws.

- The layout should scream what the repo does: files and folders split by responsibility, one main class or concept per file.
- Prefer composition to inheritance.
- A file that collects loosely related helpers means a boundary is wrong. Fix the boundary, not the file.

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
