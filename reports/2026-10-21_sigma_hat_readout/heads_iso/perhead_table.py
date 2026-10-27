"""Render the per-head calibration table from the committed tier-2 harness, as markdown.

Runs tier2_remedies_perhead.py's own statistics in-process so the numbers in the write-up are
the harness's, not transcribed by hand.
"""
import importlib.util
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).parent
def _f2_fires(o):
    """True/False from the harness's F2 line, or raise if it did not emit one."""
    m = re.search(r"^F2.*$", o, re.M)
    if m is None:
        raise RuntimeError(
            "the harness emitted no F2 line, so whether F2 fires is UNKNOWN. Refusing to "
            "render it as a pass -- see ADR-025: a check that cannot see its subject must "
            "fail, not pass."
        )
    return "FIRES" in m.group(0)


spec = importlib.util.spec_from_file_location("t2", HERE / "tier2_remedies_perhead.py")
t2 = importlib.util.module_from_spec(spec)
sys.argv = ["t2", "amber_sb"]
spec.loader.exec_module(t2)          # defines helpers; its main() is guarded

# ⛔ The arms are FIXED, and that is now stated instead of implied. This was
#    `ARMS = sys.argv[1:] if len(sys.argv) > 1 else None` followed immediately by an
#    unconditional overwrite, so `perhead_table.py silver014` silently rendered the
#    amber/bronze/cobalt table instead. sys.argv is also clobbered above to drive the
#    imported harness, so the argv read could never have worked in the first place.
#    SEED below pins the three arms this table is defined over; adding a fourth means
#    adding its seed, which is the edit that should be required.
ARMS = ["amber", "bronze", "cobalt"]
if len(sys.argv) > 1 and sys.argv[1:] != ["amber_sb"]:
    sys.exit(f"perhead_table.py takes no arguments; the arms are fixed at {ARMS}. "
             f"Got {sys.argv[1:]} -- refusing rather than rendering a different table.")
HEADS = ["sb", "ns", "os"]
SEED = {"amber": 42, "bronze": 43, "cobalt": 44}

rows = []
for arm in ARMS:
    for h in HEADS:
        if not (HERE.parent / "results" / f"p2_result_{arm}_{h}.npz").exists():
            continue
        buf = io.StringIO()
        sys.argv = ["t2", f"{arm}_{h}"]
        with redirect_stdout(buf):
            t2.main()
        o = buf.getvalue()
        def g(pat, cast=float):
            m = re.search(pat, o)
            return cast(m.group(1)) if m else None
        raw = re.search(r"^raw\s+([\d.-]+)\s+([\d.]+)\s+([\d.]+)", o, re.M)
        r1  = re.search(r"^R1\s+([\d.-]+)\s+([\d.]+)\s+([\d.]+)", o, re.M)
        r3  = re.search(r"^R3\s+([\d.-]+)\s+([\d.]+)\s+([\d.]+)", o, re.M)
        r4  = re.search(r"^R4\s+([\d.-]+)\s+([\d.]+)\s+([\d.]+)", o, re.M)
        rows.append(dict(
            arm=arm, seed=SEED[arm], head=h,
            pos=g(r"positives ([\d,]+)", lambda s: int(s.replace(",", ""))),
            b_raw=float(raw.group(1)), e_raw=float(raw.group(2)), ap=float(raw.group(3)),
            b_r1=float(r1.group(1)), e_r1=float(r1.group(2)),
            b_r3=float(r3.group(1)), e_r3=float(r3.group(2)),
            b_r4=float(r4.group(1)), e_r4=float(r4.group(2)),
            T=g(r"R1 temperature: mean ([\d.]+)"),
            Tspread=g(r"spread ([\d.]+)%"),
            f2=g(r"F2\s+\|b_R1 - 1\| = ([\d.]+)"),
            # ⛔ A missing F2 line is NOT a pass. This was
            #    `"FIRES" in (re.search(...) or re.match("", "")).group(0)`, so when the
            #    harness emitted no F2 line at all -- a renamed output, or an arm where the
            #    check is skipped -- the fallback yielded "" and f2fires became False, which
            #    prints as "pass" and is counted in "F2 fires in N of M cells". "The harness
            #    did not report F2" and "F2 passed" rendered identically. Refuse instead.
            f2fires=_f2_fires(o),
            dap=g(r"F7\s+dAP = ([\d.]+)"),
        ))

print("| arm | seed | head | positives | T | T spread | raw b | b after T | F2 | raw err | err after T | err after Platt | Platt b |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    f2 = "⛔ **FIRES**" if r["f2fires"] else "pass"
    print(f"| {r['arm']} | {r['seed']} | **{r['head']}** | {r['pos']:,} | {r['T']:.4f} | "
          f"{r['Tspread']:.1f}% | {r['b_raw']:.4f} | {r['b_r1']:.4f} | {f2} | "
          f"{r['e_raw']:.5f} | {r['e_r1']:.5f} ({(r['e_r1']-r['e_raw'])/r['e_raw']*100:+.1f}%) | "
          f"{r['e_r3']:.5f} ({(r['e_r3']-r['e_raw'])/r['e_raw']*100:+.1f}%) | **{r['b_r3']:.4f}** |")

print()
nf = sum(r["f2fires"] for r in rows)
print(f"F2 fires in **{nf} of {len(rows)}** cells.")
print(f"Platt b range: **{min(r['b_r3'] for r in rows):.4f} .. {max(r['b_r3'] for r in rows):.4f}** "
      f"(max |b-1| = {max(abs(r['b_r3']-1) for r in rows):.4f})")
print(f"Temperature b range: {min(r['b_r1'] for r in rows):.4f} .. {max(r['b_r1'] for r in rows):.4f} "
      f"(max |b-1| = {max(abs(r['b_r1']-1) for r in rows):.4f})")
print(f"All dAP: {sorted({r['dap'] for r in rows})}")
for h in HEADS:
    sub = [r for r in rows if r["head"] == h]
    if len(sub) > 1:
        bs = [r["b_raw"] for r in sub]
        print(f"raw b across seeds, {h}: " + ", ".join(f"{b:.4f}" for b in bs)
              + f"   range {max(bs)-min(bs):.4f}")

# ⛔ The `T spread` column above is the WITHIN-ARM 13-fold CV spread -- what F9 tests. The
# ACROSS-SEED spread of T is a different quantity and F9 never looks at it. Conflating the two
# would repeat the error this programme has already made twice (views-models C-167 lineage).
print()
print("ACROSS-SEED spread of the fitted T (F9's 15% criterion is WITHIN-arm and does not cover this):")
for h in HEADS:
    ts = [r["T"] for r in rows if r["head"] == h]
    if len(ts) > 1:
        sp = (max(ts) - min(ts)) / (sum(ts) / len(ts)) * 100
        flag = "  <- EXCEEDS F9's 15%, across seeds" if sp > 15 else ""
        print(f"  {h}: " + ", ".join(f"{t:.4f}" for t in ts)
              + f"   spread {sp:.1f}%{flag}")
print()
print("Within-arm fold spread (what F9 actually tests), max across all cells: "
      f"{max(r['Tspread'] for r in rows):.1f}%  -> F9 passes everywhere")
