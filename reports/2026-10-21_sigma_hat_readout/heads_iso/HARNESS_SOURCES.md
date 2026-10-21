# Per-head harness — where each file comes from

Committed here: `sweep_all_heads.py` and `perhead_table.py` (written by the views-models seat).

Copied in, NOT committed (see ../.gitignore), with their sha256 at time of use:

| file here | source | sha256 at use |
|---|---|---|
| `vehicle.py` | `views-hydranet/reports/2026-10-12_underfire_diagnosis/vehicle.py` | 9735bfed83580b8c4df3c687e90ff943c4907e46312b0ffba0dd412c87d958c0 |
| `tier2_remedies_perhead.py` | `views-hydranet/reports/2026-10-18_gate_underdispersion_prereg/results/tier2_remedies.py` — **only `EV` repointed**, nothing else changed | 4322a2fd7109d1db144a2f49fedad64fa0ee502f7c1c02b63a7a7903d7bff889 (of the SOURCE) |

Data input, also not committed (56 MB, and it is views-hydranet's cache):
`views-hydranet/reports/2026-10-12_underfire_diagnosis/pull_land_all3_121_504.parquet`
sha256 56bd72f785bcb3dbf462fdf97ba7ebd94f8472d38913463ea67313c399c86ffb

To re-create:
```bash
cp ../../../../views-hydranet/reports/2026-10-12_underfire_diagnosis/vehicle.py .
sed 's|2026-10-16_pos_weight_arm_outcome/evidence|../results|' \
  ../../../../views-hydranet/reports/2026-10-18_gate_underdispersion_prereg/results/tier2_remedies.py \
  > tier2_remedies_perhead.py   # verify the EV block by eye; this is the only intended change
```

⛔ views-hydranet is read-only from this seat. Nothing here writes to it.
