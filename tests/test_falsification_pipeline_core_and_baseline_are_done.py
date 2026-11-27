"""Falsification audit, 2026-11-02: "pipeline-core and baseline is done".

⛔⛔ RESOLVED THE SAME DAY — READ THIS BEFORE THE VERDICT BELOW.
`views-datafactory 1.17.0` was published 2026-11-02 and widened `views-frames<2` to `<4`
(views-datafactory#527, closed). The blocker this audit found is GONE, and the floors it
warned against are now the correct ones:

| probe, re-run 2026-11-02 after the release | result |
|---|---|
| the lane's set with its OLD floors, `--no-cache --refresh` | **views-frames 3.0.0**, pipeline-core 3.4.0, datafactory 1.17.0, baseline 1.0.3 |
| the lane's set with the RAISED floors (>=3.4.0, >=1.17.0) | same, exit 0 |

⚠️ Two consequences worth keeping, because they are not obvious:

1. **A fresh venv needed no code change at all** to get the fixed reconciler — the old floors
   already resolve to frames 3.0.0 now. The raised floors and the new `FRAMES_FLOOR` asserts
   earn their place on the REUSED venv, which is never re-resolved, and that is the case this
   lane actually hits.
2. **PyPI's JSON API served 1.16.0 for a while after a green publish**, and a clean install
   needed `--no-cache --refresh` to see 1.17.0. A failed publish and a cached index look
   identical from here, so a resolve that still picks 1.16.0 is not evidence of either.

The verdict below is kept verbatim as the record of why the floors could not be raised before
that release. It is history, not current state (ADR-000).

VERDICT AS WRITTEN AT THE TIME: **FALSIFIED.** The claim is literally true about those two packages and false about
what it was offered to support — that the views-frames 3.0.0 reconciliation fix is now
reachable by `chrome_box`.

Measured with the lane's own tool (`uv`), the lane's own Python (3.11) and the lane's own
requirement set (`pod_run_crafd_chain.sh:351-354`, which deliberately omits the `[pandas]`
extra), in a venv proven to hold no views-* package first:

| probe | result |
|---|---|
| `views-pipeline-core>=3.4.0` + `views-baseline` | resolves → **views-frames 3.0.0**, baseline 1.0.3 |
| `views-pipeline-core>=3.3.5` + `views-datafactory` | resolves → **views-frames 1.11.0** (matches the pod's committed FREEZE exactly) |
| `views-pipeline-core>=3.4.0` + `views-datafactory` | ⛔ **No solution found. Requirements unsatisfiable.** |

⛔ **The blocker nobody named is `views-datafactory`.** Every published release — 1.9.0 through
the latest 1.16.0 — pins `views-frames>=1.10.2,<2`, as a base dependency with no extra marker.
`views-pipeline-core` 3.4.0 pins `views-frames>=3.0.0,<4.0.0`. Those are mutually unsatisfiable,
and every `chrome_box` member declares `views-datafactory>=1.13.0,<2.0.0` directly.

So flooring the reconciling ensembles at `>=3.4.0` (views-models#662 as written) does not leave
the reconciliation silently unfixed — it makes the environment **unbuildable**. That is the one
piece of good news here: the failure is loud. `pod_run_crafd_chain.sh:354` ends in
`|| die "pip install failed"`, so the chain stops at install rather than producing a wrong
forecast.

⚠️ How the audit nearly produced the wrong cause. The first two probe runs used
`views-datafactory[pandas]`, which the lane omits on purpose, and both returned
`ResolutionImpossible` — for the **pandas** conflict, not the frames one. Reporting that would
have named the wrong blocker with a true-looking error message. The instrument was only
trusted after the 3.3.5 probe reproduced the committed FREEZE byte for byte.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RECONCILING = ("chrome_box", "skinny_love", "white_mustang")


def _pypi(pkg: str, ver: str | None = None) -> dict:
    url = f"https://pypi.org/pypi/{pkg}/{ver}/json" if ver else f"https://pypi.org/pypi/{pkg}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as fh:
            return json.load(fh)
    except (urllib.error.URLError, TimeoutError) as exc:
        # ⛔ ADR-025: a check that cannot reach its subject must say so, not pass.
        pytest.skip(f"PyPI unreachable, so this assertion was NOT evaluated: {exc}")


def _frames_pins(pkg: str, ver: str | None = None) -> list[str]:
    info = _pypi(pkg, ver)["info"]
    return [r for r in (info.get("requires_dist") or []) if "views-frames" in r.replace("_", "-")]


def test_views_datafactory_admits_views_frames_3():
    """CLOSED 2026-11-02 by views-datafactory 1.17.0 — now a regression guard, not a finding.

    This was `xfail` while the cap stood. It is asserted plainly now, because a re-narrowing
    upstream would make every floor this repo just raised unsatisfiable again, and the pod
    scripts would stop at `pip install failed` after a box had already been rented.
    """
    pins = _frames_pins("views-datafactory")
    assert pins, "views-datafactory stopped declaring views-frames; re-derive this audit"
    assert not any("<2" in p for p in pins), (
        f"views-datafactory still caps views-frames below 3: {pins}. "
        "Until this widens, flooring any reconciling ensemble at views-pipeline-core>=3.4.0 "
        "makes its environment unsatisfiable."
    )


def test_pipeline_core_and_baseline_really_are_done():
    """The half of the claim that holds — recorded so a regression upstream is visible here."""
    pc = _frames_pins("views-pipeline-core", "3.4.0")
    assert pc == ["views-frames<4.0.0,>=3.0.0"], f"pipeline-core 3.4.0 frames pin changed: {pc}"
    base = _frames_pins("views-baseline", "1.0.3")
    assert base and not any("<2" in p for p in base), (
        f"views-baseline 1.0.3 was the widening release; its pin is now {base}"
    )


def test_chrome_box_floors_the_stack_that_carries_the_reconciliation_fix():
    """⛔ The guard that matters, and it needs no network.

    This REPLACES `test_no_reconciling_ensemble_floors_pipeline_core_above_what_datafactory_
    allows`, which asserted the opposite and was correct until 2026-11-02: before datafactory
    1.17.0, flooring at >=3.4.0 made the environment unbuildable. The release inverted the
    risk. What is now worth guarding is that the floors do not quietly fall BACK, because the
    failure that follows is silent — 2.9-7.3% of fatality mass dropped with `STATUS: OK` and
    `scaling_failures: 0` (views-frames#283).

    ⛔ It reads BOTH pins, not just pipeline-core. pipeline-core >=3.4.0 is a proxy that
    happens to drag frames 3.x in today; views-frames >=3.0.0 is the thing actually required.
    Asserting only the proxy is this repository's recurring defect (C-116, #594, #635).

    ⚠️ SCOPE, stated so the silence is not read as a pass: `skinny_love` and `white_mustang`
    also declare a reconciliation and are NOT covered here, because they were not part of the
    chrome_box work and nobody has measured them. That is a known gap, not a clean result.
    """
    req = REPO / "ensembles" / "chrome_box" / "requirements.txt"
    assert req.is_file(), f"{req} is missing, so this guard examined nothing"

    wanted = {"views-pipeline-core": (3, 4, 0), "views-frames": (3, 0, 0)}
    found: dict[str, tuple[int, int, int]] = {}
    for line in req.read_text().splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        for pkg in wanted:
            # Anchor on the line STARTING with the package, so a mention of one inside the
            # other's comment-free text cannot satisfy it.
            if line.startswith(pkg):
                m = re.search(r">=\s*(\d+)\.(\d+)\.(\d+)", line)
                if m:
                    found[pkg] = tuple(int(g) for g in m.groups())

    missing = sorted(set(wanted) - set(found))
    assert not missing, (
        f"chrome_box/requirements.txt declares no floor for {missing}. A check that cannot "
        f"find its subject must fail, not pass (ADR-025)."
    )
    too_low = {p: found[p] for p in wanted if found[p] < wanted[p]}
    assert not too_low, (
        f"chrome_box floors these below the stack that carries the reconciliation fix: "
        f"{too_low} (need {wanted}). Below views-frames 3.0.0 the reconciler spreads a "
        f"country total only over already-non-zero cells, so a country whose grid is all-zero "
        f"in a draw loses its whole mass with no error — measured 4.023%/7.321%/2.943% on the "
        f"2026-10-30 delivery, which reported STATUS: OK (views-frames#283, #659/#663)."
    )
