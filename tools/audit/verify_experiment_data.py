"""Answer one question for every model in the Matryoshka comparison: do we hold its data?

WHY THIS EXISTS. On 2026-11-17 this seat told the operator that eleven of the experiment's models
had never run. They had. The 2026-11-13 batch of twelve landed in `pod_deliverables/`, not under
`reports/`, so looking in one directory and reading the absence of a name there as "never ran" was
wrong by eight days and twelve models. A ledger that is re-derived from the disk on demand cannot
make that mistake; a sentence in a session cannot avoid it.

READ-ONLY. It lists, hashes and reads archive indexes. It extracts one array per sampled model into
a temporary directory and deletes it. It writes nothing into the repository and nothing into the
deliverable directories.

WHAT IT CHECKS, per model that claims to hold data:
  (a) the archive file exists where the roster says it does;
  (b) its sha256 matches the batch's own checksum list, where the batch kept one;
  (c) its index holds EXPECTED_ORIGINS distinct origins and EXPECTED_TARGETS targets under each;
  (d) with --deep, one forecast array is loaded and its shape asserted to be
      (cells x months, EXPECTED_DRAWS) — the only check that reads the numbers rather than the names.

⛔ (b) IS NOT OPTIONAL WHERE A LIST EXISTS AND IS NOT A SKIP WHERE IT DOES NOT. A batch with no
checksum list is reported `no-list`, which is a WEAKER result than `ok`, never an equal one. The
2026-11-04 batch is in that state permanently and the ledger must keep saying so (ADR-025: a check
that cannot see its subject must refuse, not skip).

⛔ A MODEL WITH NO DATA IS NOT A FAILURE. Four of the sixteen are known-missing by design — they are
the moving-window controls, and three of them have no folder yet. They print `missing` and do not
change the exit status. `--require-all` is the flag that makes them fatal, for the day the set is
supposed to be complete.

The roster below is declared here, in one literal, on purpose. Deriving it from `models/` would make
the ledger silent about `narrow_sailcloth`, `middle_sailcloth` and `broad_sailcloth` precisely
because they do not exist — and their absence is the single most important fact in the ledger.

Keep in step with `docs/matryoshka_experiment_catalog.md`; `tests/test_experiment_ledger_matches_the_catalog.py`
fails if the two disagree about which models are in the set.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLATFORM = REPO.parent

#: The 2026-11-13 batch: twelve models, one archive each, with the pod's own checksum list.
BATCH_NOV13 = PLATFORM / "pod_deliverables/2026-11-13_baseline_twelve"
#: The 2026-11-04 batch: ten models, forecasts and scores in separate archives, NO checksum list.
BATCH_NOV04 = REPO / "reports/2026-11-04_baseline_phase1/archives_s64"

EXPECTED_ORIGINS = 13
EXPECTED_TARGETS = 3
EXPECTED_DRAWS = 64
#: 64,818 land cells x 36 forecast months. Asserted, not derived, so a change to either shows up.
EXPECTED_ROWS = 64_818 * 36


@dataclass(frozen=True)
class Member:
    name: str
    algorithm: str
    role: str
    #: The batch directory holding this model's archive, or None when no run exists.
    batch: Path | None


#: Section numbers match docs/matryoshka_experiment_catalog.md.
ROSTER: tuple[Member, ...] = (
    # 1 — the subject of the study: all history, recency-weighted, pooled over a nest of areas.
    Member("nested_burlap", "MatryoshkaConflictology", "nested-doll", BATCH_NOV13),
    Member("lone_burlap", "MatryoshkaConflictology", "nested-doll", BATCH_NOV13),
    Member("local_burlap", "MatryoshkaConflictology", "nested-doll", BATCH_NOV13),
    Member("civic_burlap", "MatryoshkaConflictology", "nested-doll", BATCH_NOV13),
    Member("global_burlap", "MatryoshkaConflictology", "nested-doll", BATCH_NOV13),
    Member("mapless_burlap", "MatryoshkaConflictology", "nested-doll", BATCH_NOV13),
    Member("stateless_burlap", "MatryoshkaConflictology", "nested-doll", BATCH_NOV13),
    Member("distant_burlap", "MatryoshkaConflictology", "nested-doll", BATCH_NOV13),
    # 2 — fixed window, ending at the last training month, never moving.
    Member("plain_burlap", "ConflictologyModel", "fixed-window", BATCH_NOV13),
    Member("narrow_canvas", "PooledConflictology", "fixed-window", BATCH_NOV13),
    Member("middle_canvas", "PooledConflictology", "fixed-window", BATCH_NOV13),
    Member("broad_canvas", "PooledConflictology", "fixed-window", BATCH_NOV13),
    # 3 — moving window: the comparators that make sections 1 and 2 comparable past origin 0.
    Member("rolling_burlap", "RollingConflictology", "moving-window", None),
    Member("narrow_sailcloth", "RollingPooledConflictology", "moving-window", None),
    Member("middle_sailcloth", "RollingPooledConflictology", "moving-window", None),
    Member("broad_sailcloth", "RollingPooledConflictology", "moving-window", None),
)

#: The twin each moving-window model must equal at origin 0 — the acceptance check for section 3.
ORIGIN_ZERO_TWIN = {
    "rolling_burlap": "plain_burlap",
    "narrow_sailcloth": "narrow_canvas",
    "middle_sailcloth": "middle_canvas",
    "broad_sailcloth": "broad_canvas",
}


def _archive(member: Member) -> Path | None:
    return None if member.batch is None else member.batch / f"{member.name}_s64.tar.zst"


def _index(archive: Path) -> list[str]:
    """The archive's member names. Reads the index only; never unpacks."""
    out = subprocess.run(
        ["tar", "--use-compress-program=unzstd", "-tf", str(archive)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.splitlines()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _declared_hashes(batch: Path) -> dict[str, str] | None:
    """The batch's own checksum list, or None when it kept none."""
    listing = batch / "SHA256.pod"
    if not listing.is_file():
        return None
    pairs = {}
    for line in listing.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        pairs[name.strip()] = digest.strip()
    return pairs


def _shape_of_first_origin(archive: Path) -> tuple[int, ...]:
    """Load one forecast array and return its shape. The only check that reads numbers."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "y_pred.npy"
        with dest.open("wb") as fh:
            subprocess.run(
                ["tar", "--use-compress-program=unzstd", "-xf", str(archive),
                 "--wildcards", "*/origin_0/lr_sb_best/y_pred.npy", "-O"],
                stdout=fh, check=True,
            )
        return tuple(np.load(dest, allow_pickle=False).shape)


def verify(member: Member, deep: bool) -> tuple[str, str]:
    """Return (verdict, detail). Verdict is one of: ok, no-list, missing, FAIL."""
    archive = _archive(member)
    if archive is None:
        folder = REPO / "models" / member.name
        where = "folder exists, never run" if folder.is_dir() else "no folder, never run"
        return "missing", where

    if not member.batch.is_dir():
        # ADR-025: the batch directory is the subject. Unreachable is a refusal, not a pass.
        return "FAIL", f"batch directory absent: {member.batch}"
    if not archive.is_file():
        return "FAIL", f"archive absent: {archive}"

    declared = _declared_hashes(member.batch)
    if declared is None:
        checksum = "no-list"
    elif archive.name not in declared:
        return "FAIL", f"not named in {member.batch.name}/SHA256.pod"
    elif _sha256(archive) != declared[archive.name]:
        return "FAIL", "sha256 does not match the batch's own list"
    else:
        checksum = "sha256 ok"

    index = _index(archive)
    origins = {p.split("origin_")[1].split("/")[0] for p in index if "origin_" in p}
    if len(origins) != EXPECTED_ORIGINS:
        return "FAIL", f"{len(origins)} origins, expected {EXPECTED_ORIGINS}"
    targets = {
        p.split("/")[-2] for p in index if p.endswith("y_pred.npy")
    }
    if len(targets) != EXPECTED_TARGETS:
        return "FAIL", f"{len(targets)} targets, expected {EXPECTED_TARGETS}"

    detail = f"{len(origins)} origins, {len(targets)} targets, {checksum}"
    if deep:
        shape = _shape_of_first_origin(archive)
        if shape != (EXPECTED_ROWS, EXPECTED_DRAWS):
            return "FAIL", f"origin 0 array is {shape}, expected {(EXPECTED_ROWS, EXPECTED_DRAWS)}"
        detail += f", {EXPECTED_DRAWS} draws"

    return ("ok" if checksum == "sha256 ok" else "no-list"), detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deep", action="store_true",
                    help="also load one forecast array per model and assert its shape (slow, ~1 GB read)")
    ap.add_argument("--require-all", action="store_true",
                    help="treat a model with no data as a failure (for when the set should be complete)")
    args = ap.parse_args()

    rows, failures, missing = [], [], []
    for member in ROSTER:
        verdict, detail = verify(member, args.deep)
        rows.append((member, verdict, detail))
        if verdict == "FAIL":
            failures.append(member.name)
        elif verdict == "missing":
            missing.append(member.name)

    width = max(len(m.name) for m in ROSTER)
    last_role = None
    for member, verdict, detail in rows:
        if member.role != last_role:
            print(f"\n-- {member.role}")
            last_role = member.role
        mark = {"ok": "ok     ", "no-list": "ok*    ", "missing": "NO DATA", "FAIL": "FAIL   "}[verdict]
        print(f"  {member.name:<{width}}  {mark}  {detail}")

    held = sum(1 for _, v, _ in rows if v in ("ok", "no-list"))
    print(f"\n{held} of {len(ROSTER)} models have verified data; "
          f"{len(missing)} have none; {len(failures)} failed.")
    if any(v == "no-list" for _, v, _ in rows):
        print("ok* = contents verified, but the batch kept no checksum list, so the bytes "
              "cannot be proved identical to what left the pod.")
    if missing:
        print("\nNo data, and each needs its origin-0 twin to accept it once run:")
        for name in missing:
            print(f"  {name:<{width}}  must equal {ORIGIN_ZERO_TWIN[name]} at origin 0")
    if not args.deep:
        print("\nRun with --deep to also assert the draw count from the arrays themselves.")

    if failures:
        print(f"\nFAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    if missing and args.require_all:
        print(f"\nINCOMPLETE: {', '.join(missing)} have no data", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
