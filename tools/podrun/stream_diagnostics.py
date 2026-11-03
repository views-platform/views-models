"""Pull a running pod's diagnostic plots to the laptop, live, and prove the copy is complete.

⛔ **This existed once and was not committed, which is why it is being written again.** On
2026-10-26 a plot feed was hand-built on a pod, streamed correctly, and died with the machine.
Nothing of it survives: `git log -S 'diagnostics/' -- tools/ scripts/` is empty on every
branch, and the only three files in this repository that mention it — the risk register,
ADR-025, and `tests/test_falsification_pod_teardown_completeness.py` — are records of it
going wrong rather than the thing itself. The residue is the empty
`models/{indigo,bronze,silver}_kestrel/reports/plots/diagnostics/<ts>/` trees on disk now.
That is the fifth instance of the pattern "hand-fixes on a rented pod die with it".

So this runs on the LAPTOP and pulls. A puller can be restarted, read, and fixed while the
run continues; a pod-side pusher cannot be any of those once the pod is gone.

## The three failures it is built against, each recorded before it was written

**Enumerate run folders, never hardcode one.** The 2026-10-26 feed was armed at 03:56 against
`diagnostics/261026_0143/`, the only folder existing then. `evaluate` created `261026_0410/`
at 04:10 and wrote 17 files into it, including all 13 per-origin `03_model_reasoning`
biopsies — the forecast-reasoning plots. The sync never looked. A HydraNet run produces one
folder per stage, so the second folder is not an edge case: it is where the evaluation plots
always live.

**Equal counts are not equal content.** That run reported local and remote as matching
because *both* numbers were counted inside the hardcoded folder. Parity here compares the set
of (relative path, size) pairs, so a folder the puller never visited cannot contribute a
matching number to both sides.

**A check that cannot see its subject must refuse (ADR-025).** `diagnostics/` absent means
either "viz is off", "the stage has not started", or "the path is wrong", and those are not
the same answer. While streaming, absence is reported and tolerated; at `--final` it is a
refusal, because that is the moment the distinction becomes expensive.

    python tools/podrun/stream_diagnostics.py --host root@216.81.151.63 --port 12787 \
        --model calm_orbiter --dest models/calm_orbiter/reports/plots

⚠️ `reports/plots/` is gitignored (`.gitignore:240` is `*.png`), so what this pulls is durable
on disk and never in git. It is retrieved evidence, not a tracked artefact.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

#: Where HydraNet writes plots, relative to the model directory. `pod_run_model.sh:371` does
#: `cd "$REPO/models/$MODEL"` before invoking main.py, so this resolves under the model dir on
#: the pod. Confirmed twice by observation: the empty trees on this laptop have exactly this
#: shape, and C-68 records `/workspace/views-models/models/heavy_freighter/reports/plots`
#: existing on a live pod.
REMOTE_PLOTS = "reports/plots/diagnostics"

DEFAULT_REPO = "/workspace/views-models"


class Refuse(SystemExit):
    """Stop with a reason, rather than report a verdict the evidence does not support."""


@dataclass(frozen=True)
class Pod:
    host: str
    port: int
    key: str

    def ssh(self, command: str, timeout: int = 120) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=20", "-p", str(self.port), "-i", self.key,
             self.host, command],
            capture_output=True, text=True, timeout=timeout,
        )

    def rsync_from(self, remote: str, dest: Path, timeout: int = 900) -> subprocess.CompletedProcess:
        dest.mkdir(parents=True, exist_ok=True)
        return subprocess.run(
            ["rsync", "-a", "-e",
             f"ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p {self.port} -i {self.key}",
             f"{self.host}:{remote}/", f"{dest}/"],
            capture_output=True, text=True, timeout=timeout,
        )


def remote_folders(pod: Pod, plots_root: str) -> list[str]:
    """Every `diagnostics/<ts>/` on the pod. The enumeration the 2026-10-26 feed skipped."""
    r = pod.ssh(f"test -d {plots_root} && find {plots_root} -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' | sort")
    if r.returncode != 0:
        return []
    return [line for line in r.stdout.split() if line]


def remote_manifest(pod: Pod, plots_root: str) -> dict[str, int]:
    """(path relative to diagnostics/, size) for every remote file — content, not a count."""
    r = pod.ssh(f"cd {plots_root} 2>/dev/null && find . -type f -printf '%s %P\\n' | sort -k2")
    if r.returncode != 0:
        return {}
    out: dict[str, int] = {}
    for line in r.stdout.splitlines():
        size, _, path = line.strip().partition(" ")
        if path:
            out[path] = int(size)
    return out


def local_manifest(dest: Path) -> dict[str, int]:
    """The same shape, locally, so the two can be compared as sets rather than as totals."""
    if not dest.is_dir():
        return {}
    return {
        str(p.relative_to(dest)): p.stat().st_size
        for p in dest.rglob("*")
        if p.is_file()
    }


def compare(remote: dict[str, int], local: dict[str, int]) -> tuple[list[str], list[str]]:
    """What the pod holds that the laptop does not, and what disagrees on size."""
    missing = sorted(set(remote) - set(local))
    differing = sorted(p for p in set(remote) & set(local) if remote[p] != local[p])
    return missing, differing


def sweep(pod: Pod, plots_root: str, dest: Path, final: bool) -> tuple[int, int]:
    """One pass: enumerate, pull every folder, then verify by content. Returns (folders, files)."""
    folders = remote_folders(pod, plots_root)
    if not folders:
        if final:
            raise Refuse(
                f"⛔ no {plots_root}/<run>/ directory on the pod at the final sweep.\n"
                f"   That is three different situations wearing one face: diagnostics are off,\n"
                f"   the stage has not run, or this path is wrong. Reporting 'nothing to copy'\n"
                f"   would be a pass on an unread subject (ADR-025). Check\n"
                f"   `diagnostic_visualizations` in the arm's config_meta.py — it OVERWRITES\n"
                f"   config_hyperparameters and defaults to False (C-68)."
            )
        print(f"  {plots_root} not present yet — nothing to pull")
        return 0, 0

    r = pod.rsync_from(plots_root, dest)
    if r.returncode != 0:
        raise Refuse(f"⛔ rsync failed ({r.returncode}): {r.stderr.strip()[:400]}")

    remote = remote_manifest(pod, plots_root)
    local = local_manifest(dest)
    missing, differing = compare(remote, local)
    if missing or differing:
        raise Refuse(
            f"⛔ the copy is not complete after rsync reported success.\n"
            f"   folders on pod: {folders}\n"
            f"   missing locally ({len(missing)}): {missing[:8]}\n"
            f"   size mismatch ({len(differing)}): {differing[:8]}\n"
            f"   ⚠️ Equal file COUNTS would not have caught this — the 2026-10-26 feed\n"
            f"   reported matching counts while missing a whole folder, because both\n"
            f"   numbers were taken inside the folder it did look at."
        )
    if final and not remote:
        raise Refuse(
            f"⛔ {plots_root} exists but holds no files at the final sweep. An empty tree is\n"
            f"   exactly what this laptop already has from three past runs; copying it again\n"
            f"   and reporting success is how that happened."
        )
    print(f"  {len(folders)} folder(s), {len(remote)} file(s) verified by name and size: {folders}")
    return len(folders), len(remote)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", required=True, help="e.g. root@216.81.151.63")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--key", default=str(Path.home() / ".ssh" / "id_ed25519"))
    ap.add_argument("--model", required=True, help="the arm name, e.g. calm_orbiter")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="the pod's checkout")
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--interval", type=int, default=120, help="seconds between sweeps")
    ap.add_argument("--once", action="store_true", help="one sweep, then exit")
    ap.add_argument("--final", action="store_true",
                    help="refuse if there is nothing to copy — use for the teardown sweep")
    args = ap.parse_args(argv)

    pod = Pod(args.host, args.port, args.key)
    plots_root = f"{args.repo}/models/{args.model}/{REMOTE_PLOTS}"
    print(f"feed: {args.host}:{args.port}  {plots_root}  ->  {args.dest}")

    if args.once or args.final:
        sweep(pod, plots_root, args.dest, final=args.final)
        return 0

    while True:
        print(f"[{time.strftime('%H:%M:%S')}] sweep")
        try:
            sweep(pod, plots_root, args.dest, final=False)
        except Refuse as e:
            print(str(e), file=sys.stderr)
            return 1
        except subprocess.TimeoutExpired:
            print("  ssh timed out; the pod may be busy — retrying next sweep")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
