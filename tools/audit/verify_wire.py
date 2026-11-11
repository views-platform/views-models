"""Verify a built wire against its own ADR-013 manifest, before anyone uploads it.

Every shard record carries a sha256. This recomputes all of them and reports any file that is
missing, extra, or whose bytes have changed. REFUSES (exit 2) rather than reporting a partial
pass: a wire that cannot be fully verified must not be uploaded to a partner-facing bucket.

Usage: python verify_wire.py <wire_contract/<run_id>/>
"""
import hashlib, json, sys
from pathlib import Path

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

d = Path(sys.argv[1])
mans = list(d.glob('*__manifest.json'))
if len(mans) != 1:
    print(f"REFUSING: expected exactly one __manifest.json in {d}, found {len(mans)}"); sys.exit(2)
m = json.loads(mans[0].read_text())
run = m['run_id']
print(f"run_id        {run}")
print(f"contract      {m['contract_version']}")
print(f"targets       {', '.join(m['targets'])}")
print(f"months        {len(m['expected_months'])} ({min(m['expected_months'])}..{max(m['expected_months'])})")
print(f"cells         {m['expected_cell_count']:,}")

declared = {s['name']: s['sha256'] for s in m['shards']}
sc = m.get('sidecar')
if sc: declared[sc['name']] = sc['sha256']

on_disk = {p.name for p in d.glob('*.parquet')}
missing = sorted(set(declared) - on_disk)
extra   = sorted(on_disk - set(declared))

bad, ok = [], 0
for name, want in sorted(declared.items()):
    p = d / name
    if not p.exists(): continue
    got = sha256(p)
    if got == want: ok += 1
    else: bad.append((name, want[:12], got[:12]))

print(f"\ndeclared      {len(declared)} files ({len(m['shards'])} shards + {'1 sidecar' if sc else 'no sidecar'})")
print(f"verified OK   {ok}")
print(f"MISSING       {len(missing)}" + (f"  {missing[:3]}" if missing else ""))
print(f"EXTRA on disk {len(extra)}" + (f"  {extra[:3]}" if extra else ""))
print(f"CHECKSUM BAD  {len(bad)}" + (f"  {bad[:3]}" if bad else ""))

if missing or bad:
    print("\n⛔ WIRE IS NOT INTACT — do not upload."); sys.exit(2)
if extra:
    print("\n⚠️  Extra parquet files present that the manifest does not declare. Not fatal for the")
    print("    upload (the manifest drives it), but the directory is not what the manifest says.")
print(f"\n✅ WIRE INTACT — {ok}/{len(declared)} files match their declared sha256.")
