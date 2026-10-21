"""Collect the gate for ALL THREE classification heads in ONE forward sweep.

⛔ WHY THIS EXISTS. Every number in the #565 programme — σ̂, every a and b, and the −71%
temperature result — comes from `p2_gate_calibration.py:119`:

    gate[d, fed - BASE_ORIGIN] = torch.sigmoid(out.cls[0, SB])[lri, lci].numpy()

`SB` is the fixed index 0 (`vehicle.py:53`). The model declares three classification targets
(`config_hyperparameters.py:210`: by_sb_best, by_ns_best, by_os_best). So `ns` and `os` have never
been measured, and a temperature fitted on `sb` has no claim on them.

This is a COPY of that harness's sweep with the head index lifted to a loop. views-hydranet is
read only from this seat: nothing here writes to it.

⛔ PARITY GATE: the `sb` cube this produces must equal the committed `p2_result_<arm>.npz` cube
EXACTLY. If it does not, this harness is not the same instrument and no ns/os number from it is
comparable to the published sb numbers. Checked before anything is written.

usage: sweep_all_heads.py <path to calibration_model_*.pt> <arm-name> <reference p2_result npz>
"""
import sys
from pathlib import Path

import numpy as np
import torch

import hashlib  # noqa: E402

import vehicle

_vsha = hashlib.sha256(open(vehicle.__file__, 'rb').read()).hexdigest()

# ⛔ TWO SHAPES OF vehicle.py EXIST and this must work against both, loudly.
#
# Before views-hydranet's Amendment 4 (2026-10-21 14:18), F6 pinned the family inside
# EXPECTED_SIDECAR["output_distribution"] = "nb" and REFUSED a truncated_nb vehicle, so run_p2.py
# and this script shimmed that one key. After Amendment 4 the key is GONE from EXPECTED_SIDECAR
# and the family is gated by ADMITTED_FAMILIES = ("nb", "truncated_nb"), which admits these arms
# outright — and the old shim then raises KeyError.
#
# The measurements in ../PERHEAD_CALIBRATION.md were taken with the PRE-amendment file
# (sha256 3d1085402d8a948b9e25d9799aa6980ae09e8d085e6124091214685627cf5a02). What makes them valid
# is not which file was used but the parity gate below: the sb cube reproduced the committed
# p2_result_<arm>.npz exactly. Still, the two files are different instruments, so which one ran is
# recorded rather than left to be inferred.
HAS_ADMITTED = hasattr(vehicle, "ADMITTED_FAMILIES")
PINS_FAMILY = "output_distribution" in vehicle.EXPECTED_SIDECAR
if PINS_FAMILY and not HAS_ADMITTED:
    assert vehicle.EXPECTED_SIDECAR["output_distribution"] == "nb", "gate already changed; stop"
    vehicle.EXPECTED_SIDECAR["output_distribution"] = "truncated_nb"
    VEHICLE_SHAPE = "pre-Amendment-4 (shimmed output_distribution nb -> truncated_nb)"
elif HAS_ADMITTED and not PINS_FAMILY:
    assert "truncated_nb" in vehicle.ADMITTED_FAMILIES, (
        "post-Amendment-4 vehicle.py does not admit truncated_nb; these arms cannot load"
    )
    VEHICLE_SHAPE = f"post-Amendment-4 (ADMITTED_FAMILIES={vehicle.ADMITTED_FAMILIES}, no shim)"
else:
    raise SystemExit(
        "VOID: vehicle.py is in neither known shape — "
        f"ADMITTED_FAMILIES={HAS_ADMITTED}, EXPECTED_SIDECAR pins family={PINS_FAMILY}. "
        "Refusing to guess which F6 is in force."
    )

from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate  # noqa: E402
from vehicle import TARGETS, build_volume, grid_rowcol, load_vehicle, pull  # noqa: E402

# announce which F6 is in force BEFORE nine minutes of compute, not after
print(f"vehicle.py  sha256 {_vsha}\n            shape  {VEHICLE_SHAPE}", flush=True)

ART, ARM, REF = sys.argv[1], sys.argv[2], sys.argv[3]
HEADS = {"sb": 0, "ns": 1, "os": 2}          # vehicle.py:52 input order
CACHE = ("/home/simon/Documents/scripts/views_platform/views-hydranet/reports/"
         "2026-10-12_underfire_diagnosis/pull_land_all3_121_504.parquet")
OUT = Path(__file__).parent.parent / "results"

POSTERIOR_D = 4
NP_SEED, TORCH_SEED = 42, 42
BASE_ORIGIN, N_ORIGINS = 456, 13
LEADIN_START = 421
LAST_FED = BASE_ORIGIN + N_ORIGINS - 1

ReproducibilityGate.lock_entropy(np_seed=NP_SEED, torch_seed=TORCH_SEED)
model, cfg, sha = load_vehicle(ART)
model.set_locked_dropout(True)
for m in model.modules():
    if "BatchNorm" in type(m).__name__ and m.training:
        raise SystemExit(f"VOID: {type(m).__name__} in train() mode — BN would be rewritten.")

df = pull((121, 504), CACHE)
land_cells = np.sort(df.priogrid_id.unique())
lr, lc = grid_rowcol(land_cells)
lri, lci = lr - 1, lc - 1
months = list(range(LEADIN_START, LAST_FED + 1))
vol, _, _ = build_volume(df, months)

print(f"sweeping {months[0]}-{months[-1]} x {POSTERIOR_D} passes over "
      f"{len(land_cells)} land cells, collecting ALL {len(HEADS)} heads …", flush=True)

gate = {h: np.zeros((POSTERIOR_D, N_ORIGINS, len(land_cells)), dtype=np.float32) for h in HEADS}
with torch.no_grad():
    for d in range(POSTERIOR_D):
        model.reset_locked_dropout()
        h_state = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
        for t, fed in enumerate(months):
            out = model(vol[:, t], h_state)
            h_state = out.h_next
            if fed < BASE_ORIGIN:
                continue
            for name, idx in HEADS.items():
                gate[name][d, fed - BASE_ORIGIN] = \
                    torch.sigmoid(out.cls[0, idx])[lri, lci].numpy()
        print(f"  pass {d + 1}/{POSTERIOR_D} done", flush=True)

# truth per head, exactly as p2_gate_calibration.py:124-128 does for sb
by_month = {m: g for m, g in df.groupby("month_id")}
truth = {}
for name, idx in HEADS.items():
    t_ = np.zeros((N_ORIGINS, len(land_cells)), dtype=bool)
    for n in range(N_ORIGINS):
        g = by_month[BASE_ORIGIN + 1 + n].set_index("priogrid_id").reindex(land_cells)
        t_[n] = (g[TARGETS[idx]].to_numpy() > 0)
    truth[name] = t_

# ⛔ PARITY GATE — before writing anything
ref = np.load(REF, allow_pickle=True)
ok_gate = np.array_equal(ref["gate"], gate["sb"])
ok_truth = np.array_equal(ref["truth"], truth["sb"])
print("\n" + "=" * 78)
print(f"PARITY against {Path(REF).name}")
print(f"  sb gate  identical: {ok_gate}")
print(f"  sb truth identical: {ok_truth}")
if not (ok_gate and ok_truth):
    gd = np.abs(ref["gate"].astype(np.float64) - gate["sb"].astype(np.float64)).max()
    print(f"  max|gate diff| = {gd:.3e}")
    raise SystemExit("⛔ PARITY FAILED — this is not the same instrument. Nothing written.")
print("  => same instrument. ns/os numbers are comparable to the published sb numbers.")
print(f"  vehicle.py shape: {VEHICLE_SHAPE}")
print(f"  vehicle.py sha256: {_vsha}")
print("=" * 78)

for name in HEADS:
    p = OUT / f"p2_result_{ARM}_{name}.npz"
    np.savez_compressed(p, gate=gate[name], truth=truth[name],
                        land_cells=land_cells, sha=np.array(sha))
    pr = truth[name].mean()
    print(f"  wrote {p.name}   positives {int(truth[name].sum()):>7,}   base rate {pr:.6f}")
