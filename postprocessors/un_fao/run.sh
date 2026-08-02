#!/usr/bin/env bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  if ! grep -q 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"' ~/.zshrc; then
    echo 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"' >> ~/.zshrc
  fi
  if ! grep -q 'export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"' ~/.zshrc; then
    echo 'export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"' >> ~/.zshrc
  fi
  if ! grep -q 'export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"' ~/.zshrc; then
    echo 'export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
  fi
  source ~/.zshrc
fi

script_path=$(dirname "$(realpath $0)")
project_path="$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )"
env_path="$project_path/envs/views-postprocessing"

# ── #308: the registry must resolve, and its absence is FATAL — checked FIRST ─────────
# The default below is a relative sibling hop: it assumes views-appwrite is checked out
# next to views-models at exactly this path. On a different layout, in CI, or in a
# container, it is simply absent.
#
# This check is deliberately here — before conda, before pip, before anything — because
# the cost of continuing is not a missing warning, it is a failure that lands minutes
# later at the datastore boundary, describing a symptom rather than the cause, in front
# of whoever is least equipped to trace it back to a checkout layout.
#
# Existence only at this point: parsing needs tomllib, which needs the 3.11 interpreter
# conda has not activated yet. The parse is checked below, and is equally fatal.
APPWRITE_REGISTRY="${APPWRITE_REGISTRY:-$project_path/../views-appwrite/docs/ADRs/platform/coordinate_registry.toml}"
if [ ! -f "$APPWRITE_REGISTRY" ]; then
  echo "FATAL: the Appwrite coordinate registry does not exist." >&2
  echo "  looked for: $APPWRITE_REGISTRY" >&2
  echo "  override with: APPWRITE_REGISTRY=/path/to/coordinate_registry.toml" >&2
  echo "" >&2
  echo "  The default is a relative hop to a sibling checkout of views-appwrite. If this" >&2
  echo "  machine lays the repositories out differently, set APPWRITE_REGISTRY explicitly." >&2
  echo "  This is fatal by design (#308): the registry is the ONLY source of Appwrite" >&2
  echo "  coordinates. Continuing would fail later, somewhere less informative." >&2
  exit 1
fi

# Sourcing sets SHELL variables; only what is `export`ed reaches `python main.py` (a child process).
# Do NOT replace this with `set -a` — .env carries unquoted values containing spaces (the *_NAME
# coordinates), which `set -a` would export truncated at the first space. Export by name instead. (#293)
if [ -f "$project_path/.env" ]; then
  source "$project_path/.env"
  export GITHUB_TOKEN
  export APPWRITE_DATASTORE_API_KEY   # the one secret; the operator slot. Coordinates come from the registry below.
fi

eval "$(conda shell.bash hook)"

if [ -d "$env_path" ]; then
  echo "Conda environment already exists at $env_path. Checking dependencies..."
  conda activate "$env_path"
  echo "$env_path is activated"

  missing_packages=$(pip install --dry-run -r $script_path/requirements.txt 2>&1 | grep -v "Requirement already satisfied" | wc -l)
  if [ "$missing_packages" -gt 0 ]; then
    echo "Installing missing or outdated packages..."
    pip install -r $script_path/requirements.txt
  else
    echo "All packages are up-to-date."
  fi
  echo "Installing views-postprocessing from GitHub..."
  pip install git+https://${GITHUB_TOKEN}@github.com/views-platform/views-postprocessing.git@main
else
  echo "Creating new Conda environment at $env_path..."
  conda create --prefix "$env_path" python=3.11 -y
  conda activate "$env_path"
  pip install -r $script_path/requirements.txt
  echo "Installing views-postprocessing from GitHub..."
  pip install git+https://${GITHUB_TOKEN}@github.com/views-platform/views-postprocessing.git@main
fi

# ── #294: does the build we just installed do what config_meta DECLARES? ──────────────
# The installs above pin `@main` — a MOVING branch pointer. Whether it can produce the
# artifact this postprocessor declares is a fact about someone else's repository on the
# day you run, and nothing here checked it.
#
# That is not hypothetical. On 2026-07-31 `@main` was 208 commits behind, carried zero
# wire modules, and still ran to completion: it would have ignored `wire_contract: True`
# and delivered the LEGACY parquet instead of the ADR-013 contract dialect — green run,
# wrong artifact, and faoapi then serving the previous month. views-postprocessing#178
# merged on 2026-08-01 and `@main` now carries the wire, so the pin happens to be correct
# today. Correct by timing is not the same as verified, and the next drift is silent
# again.
#
# So: read what config_meta declares, then assert the installed package can honour it.
# Declaration read through importlib, never grep — a commented-out key looks identical to
# a live one to a regex, which is register entry C-57.
_wire_declared="$(python - "$script_path/configs/config_meta.py" <<'PY' 2>/dev/null || echo unknown
import importlib.util, sys
spec = importlib.util.spec_from_file_location("un_fao_config_meta", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("yes" if module.get_meta_config().get("wire_contract") else "no")
PY
)"

if [ "$_wire_declared" = "yes" ]; then
  if ! python -c "import views_postprocessing.contract.wire" >/dev/null 2>&1; then
    echo "ERROR: config_meta declares wire_contract: True, but the installed" >&2
    echo "       views-postprocessing cannot import views_postprocessing.contract.wire." >&2
    echo "       This build CANNOT produce the ADR-013 contract dialect. Left to run, it" >&2
    echo "       would exit 0 having delivered the legacy artifact, and FAO would keep" >&2
    echo "       being served the previous month (#294)." >&2
    echo "       Installed from: github.com/views-platform/views-postprocessing @main" >&2
    echo "       Fix: point the pin at a build that carries the wire, or set" >&2
    echo "       wire_contract: False if the legacy artifact is genuinely what you want." >&2
    exit 1
  fi
  echo "Capability check: wire_contract declared and views_postprocessing.contract.wire importable."
elif [ "$_wire_declared" = "unknown" ]; then
  echo "Capability check SKIPPED: could not read wire_contract from config_meta." >&2
  echo "  Not fatal — this check only ever ADDS a failure mode, it must not invent one." >&2
fi

# ── þing-01 / PLATFORM-001 (#287): coordinates come from the OWNED registry, not a copied .env ──
# The non-secret Appwrite coordinates (endpoint, project/bucket/collection/db ids & names) are
# READ from the single canonical coordinate registry — never copied into this repo. The one SECRET
# (APPWRITE_DATASTORE_API_KEY) stays an operator slot — exported by name above; see #293 for why a
# bare `source` was never a carrier. This is a DECLARATION layered on run.sh, not a rewrite. The hard
# cutover (registry-only coordinates, secret from a real store, .env retired) is sequenced with #280
# and pairs with views-postprocessing's kill of its runtime load_dotenv (their P1). Uses the
# just-activated env's python because tomllib needs 3.11+. Override the path with APPWRITE_REGISTRY.
#
# CORRECTION (#293): this comment previously claimed "the .env sourced above still works as the
# operator slot + fallback". It did not — `source` without `export` never reached the python child,
# so between views-postprocessing's load_dotenv removal (2026-07-28) and this change there was no
# carrier for the secret at all. The claim was wrong when written.
# The path was resolved and its existence made fatal at the top of this script (#308).
#
# REMOVED HERE (#308/#309): `_platform001_coordinate_state()`, added 2026-07-31, which on a
# missing registry announced "Coordinates ARE present in the environment (exported outside
# this script) — continuing." **That claim was false.** It tested SHELL variables, which
# `source .env` above does set — but nothing exports them, so the python child never saw
# them. Verified: after `source .env; export GITHUB_TOKEN APPWRITE_DATASTORE_API_KEY`, the
# shell has APPWRITE_ENDPOINT and `python -c 'os.environ.get("APPWRITE_ENDPOINT")'` returns
# None. It was written to avoid asserting an unverified environment fact, and asserted one
# by checking the wrong scope. It also settles the design question it was hedging: there was
# never a second working source for the child, so registry-absent IS coordinates-absent, and
# fatal costs nothing.

_reg_err="$(mktemp "${TMPDIR:-/tmp}/registry_to_env.XXXXXX")"
trap 'rm -f "$_reg_err"' EXIT
if ! _coords="$(python "$project_path/tools/credentials/registry_to_env.py" "$APPWRITE_REGISTRY" 2>"$_reg_err")"; then
  echo "FATAL: the coordinate registry exists but could not be read." >&2
  echo "  registry: $APPWRITE_REGISTRY" >&2
  echo "  python:   $(python -V 2>&1)  (registry_to_env.py needs 3.11+ for tomllib)" >&2
  sed 's/^/  /' "$_reg_err" >&2
  echo "  Fatal by design (#308): the registry is the ONLY source of coordinates." >&2
  rm -f "$_reg_err"; exit 1
fi
rm -f "$_reg_err"; trap - EXIT

# ── #309: one writer. `.env` must not also declare a coordinate the registry owns ────
# Two writers to the same names is a data race whose winner is decided by line order:
# reverse these blocks and the semantics invert silently, with no test failing. Per the
# seam contract that is an error, not a precedence question — so it is reported, not
# resolved.
#
# Deleting these keys from `.env` is safe: they were NEVER exported (see the note above),
# so nothing downstream has ever received them from that file. Keep the SECRET there.
_owned="$(echo "$_coords" | cut -d= -f1)"
_conflicts=""
if [ -f "$project_path/.env" ]; then
  for _name in $_owned; do
    if grep -qE "^[[:space:]]*(export[[:space:]]+)?${_name}=" "$project_path/.env"; then
      _conflicts="$_conflicts $_name"
    fi
  done
fi
if [ -n "$_conflicts" ]; then
  echo "FATAL: .env declares coordinates that the registry owns (#309)." >&2
  echo "  file:     $project_path/.env" >&2
  echo "  registry: $APPWRITE_REGISTRY" >&2
  echo "  both declare:$_conflicts" >&2
  echo "" >&2
  echo "  Two writers to one name is a data race decided by line order, so this is an" >&2
  echo "  error rather than a precedence question. Delete these lines from .env — they" >&2
  echo "  were never exported, so nothing has ever received them from there. Keep" >&2
  echo "  APPWRITE_DATASTORE_API_KEY: the secret is the one value .env still carries." >&2
  exit 1
fi

while IFS= read -r _cl; do
  [ -z "$_cl" ] && continue
  export "${_cl%%=*}=${_cl#*=}"
done <<< "$_coords"
echo "Appwrite coordinates read from the registry (the only source); secret stays the operator slot."

echo "Running $script_path/main.py "
python $script_path/main.py "$@"
