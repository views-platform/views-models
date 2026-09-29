# Environment snapshots

One `pip freeze` per conda environment, written by `monthly_run.sh` **after** each
folder runs, and committed with that month's work.

## Why these are in git

Your code is in git; the ~200 packages that ran alongside it were not. They live in
`envs/`, which is gitignored and exists only on whichever laptop ran the month — of
this repo's 11 environments, 3 existed on the maintainer's machine when this was
written, totalling 22GB.

That made a delivered UN FAO forecast **half-reproducible**: you could recover the
configuration that produced it but not the dependency versions. The configuration
half of the same gap is register **C-110**; this is the dependency half, **C-117**.

`logs/` would not do. It is gitignored, so a snapshot there would be exactly as
ephemeral as the environment it describes.

## Reading one

The header names the run, the environment, and **the commit**. Neither half is
sufficient alone — the commit tells you what your code said, the package list tells
you what it ran against.

```
# run_id:      20260803T000709Z
# environment: envs/views_ensemble
# commit:      7d22f608...
```

## What they are good at

Diffing month to month. `diff` two snapshots for the same environment and you have
the complete list of what changed underneath the models between two forecasts — the
first question worth asking when output moves and no config did.

They also make environment problems visible that nothing else does. The first
snapshot ever taken showed `views-pipeline-core` installed **editable**, with no
`views-frames` line — which is why all four production ensembles were failing at
import (see C-116).

## Committing them

`monthly_run.sh` writes them; you commit them. They are deliberately not written
automatically into a commit, because a production run should not also be a git
author.
