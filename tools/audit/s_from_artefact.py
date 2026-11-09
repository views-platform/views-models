"""Derive the sample count S from a run's saved predictions, never from a note.

views-baseline's condition on views-models#718: the S in every reported number must come from the
predictions themselves, so a mismatch between what was run and what is reported cannot slip through.
The committed configs say `n_samples: 256` while the epic runs at 64 (256 needs ~192 GB peak disk
against a 100 GB pod quota), so a hand-written label is exactly the thing that would go stale.

Reads every `y_pred.npy` under a predictions tree and reports the sample axis. REFUSES rather than
guessing: if the tree cannot be read, or arrays disagree on S, it exits non-zero and says so — a
report that cannot establish its own S must not produce one (ADR-025).

Usage:
    python -m tools.report_s_from_artefact <predictions_dir_or_tar.zst> [...]
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


def _s_values_from_dir(root: Path) -> Counter[int]:
    found: Counter[int] = Counter()
    for p in sorted(root.rglob("y_pred.npy")):
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 2:
            raise SystemExit(f"REFUSING: {p} has ndim {arr.ndim}, expected 2 (rows, samples)")
        found[int(arr.shape[1])] += 1
    return found


def _s_values_from_archive(tf: Path) -> Counter[int]:
    with tempfile.TemporaryDirectory() as td:
        out = subprocess.run(
            ["tar", "-I", "zstd -d", "-xf", str(tf), "-C", td],
            capture_output=True, text=True, check=False,
        )
        if out.returncode != 0:
            raise SystemExit(f"REFUSING: cannot read {tf.name}: tar exited {out.returncode}")
        return _s_values_from_dir(Path(td))


def report(target: str) -> int:
    p = Path(target)
    if not p.exists():
        print(f"REFUSING: {target} does not exist")
        return 2
    counts = _s_values_from_archive(p) if p.is_file() else _s_values_from_dir(p)

    if not counts:
        print(f"REFUSING: no y_pred.npy found under {target} — cannot establish S")
        return 2
    if len(counts) > 1:
        print(f"REFUSING: {target} holds arrays with DIFFERENT sample counts: {dict(counts)}")
        print("  A single S cannot describe this run; report per-array or re-run.")
        return 2

    s, n = next(iter(counts.items()))
    print(f"{p.name}: S = {s}  (derived from {n} y_pred.npy arrays, all agreeing)")
    print()
    print("  Boilerplate for the write-up, required once per result by views-baseline on #718:")
    print(f'    "config says 256; run at {s} on the pod because 256 needs ~192 GB peak against a '
          '100 GB quota (#718)."')
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    return max(report(t) for t in sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
