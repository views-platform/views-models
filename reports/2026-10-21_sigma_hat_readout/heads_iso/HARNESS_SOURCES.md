# Per-head harness — where each file comes from

Committed here: `sweep_all_heads.py` and `perhead_table.py` (written by the views-models seat).

## ⛔ Correction, 2026-10-21

The first version of this file recorded `vehicle.py` as sha256
`9735bfed83580b8c4df3c687e90ff943c4907e46312b0ffba0dd412c87d958c0`. **That is the sha of a file the
measurements never used.** It was produced by running `sha256sum` against the *source path* at the
time this document was written, not against the copy the run imported — and in between, the
views-hydranet seat amended the source:

| when | what |
|---|---|
| 13:50:40 | this seat copied `vehicle.py` → sha `3d1085402d…` |
| 14:18:44 | views-hydranet landed **Amendment 4**, rewriting it → sha `9735bfed83…` |
| ~14:35 | this file was written and recorded the **source's current** sha, i.e. the wrong one |

**The measurements are unaffected and the reason is the parity gate, not luck:** `sweep_all_heads.py`
refuses to write unless its `sb` cube equals the committed `p2_result_<arm>.npz` cube under
`np.array_equal`, gate and truth. It matched for all three arms, which is what establishes that the
instrument was the one that produced the published `sb` numbers. Provenance recorded from the
wrong place is still a defect — same class as views-models C-167 and
`feedback_provenance_before_reuse`: a correct value read from a path that answers a different
question. `sweep_all_heads.py` now hashes `vehicle.__file__` at run time and prints it, so the
record comes from the artifact rather than from a path.

## ⚠️ The two shapes of `vehicle.py`, and why this matters to anyone re-running it

- **Pre-Amendment 4** (sha `3d1085402d…`, **what the measurements used**): F6 pinned the family
  inside `EXPECTED_SIDECAR["output_distribution"] = "nb"` and refused a `truncated_nb` vehicle, so
  the harness shimmed that one key — as `run_p2.py` does.
- **Post-Amendment 4** (sha `9735bfed83…`, current): the key is **gone** from `EXPECTED_SIDECAR`
  and the family is gated by `ADMITTED_FAMILIES = ("nb", "truncated_nb")`, which admits these arms
  outright. The old shim then raises `KeyError: 'output_distribution'`.

⛔ **Consequence beyond this directory:** `views-hydranet`'s own `run_p2.py` carries that shim, so
`../sweep_all.sh` — which rebuilds its harness from the live source — **fails against the current
file**. Verified: `KeyError: 'output_distribution'`. `build_iso` now pins the sources' shas and
refuses rather than running a changed instrument silently.

`sweep_all_heads.py` handles both shapes, asserts exactly one applies, and refuses if neither does.

## Sources

| file here | source | sha256 OF THE COPY ACTUALLY USED |
|---|---|---|
| `vehicle.py` | `views-hydranet/reports/2026-10-12_underfire_diagnosis/vehicle.py` | 3d1085402d8a948b9e25d9799aa6980ae09e8d085e6124091214685627cf5a02 |
| `tier2_remedies_perhead.py` | `views-hydranet/reports/2026-10-18_gate_underdispersion_prereg/results/tier2_remedies.py` — **only `EV` repointed** | c815451e5150e1d7b2d52d9bae46a546219c0a2be11cb13072756629424bfab8 |

Source shas as they stand now (for drift detection, NOT what was used):

| source | sha256 now |
|---|---|
| `.../2026-10-12_underfire_diagnosis/vehicle.py` | 9735bfed83580b8c4df3c687e90ff943c4907e46312b0ffba0dd412c87d958c0 |
| `.../2026-10-18_gate_underdispersion_prereg/results/tier2_remedies.py` | 4322a2fd7109d1db144a2f49fedad64fa0ee502f7c1c02b63a7a7903d7bff889 |

Data input (56 MB, views-hydranet's cache, not committed):
`.../2026-10-12_underfire_diagnosis/pull_land_all3_121_504.parquet`
sha256 56bd72f785bcb3dbf462fdf97ba7ebd94f8472d38913463ea67313c399c86ffb

⛔ views-hydranet is read-only from this seat. Nothing here writes to it.
