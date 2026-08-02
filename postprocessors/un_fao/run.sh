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

# ── the environment contract lives in ONE place (#309) ───────────────────────────────
# tools/credentials/platform_env.sh is the only writer of Appwrite coordinates and the secret. This
# script keeps what is genuinely its own — macOS notes, conda lifecycle, pip install — and
# borrows nothing else. bootstrap.sh sources the same file, which is what makes the shared
# file worth having: two consumers, not one.
# shellcheck source=../../tools/credentials/platform_env.sh
. "$project_path/tools/credentials/platform_env.sh"

# Registry existence FIRST — before conda, before pip. Continuing without it does not cost
# a warning, it moves the failure to the datastore boundary minutes later, describing a
# symptom rather than a cause, in front of whoever is least equipped to trace it back to a
# checkout layout (#308). Parsing needs 3.11, so that half is checked after conda below.
platform_env_require_registry || exit 1

# GITHUB_TOKEN only, and only because the pip install below needs it before anything else
# runs. The Appwrite secret is NOT exported here: `platform_env_export_secret` owns it, and
# doing it in both places would reinstate the second writer #309 exists to remove.
#
# Sourcing sets SHELL variables; only what is `export`ed reaches a child process. Do NOT
# replace this with `set -a` — .env carries unquoted values containing spaces (the *_NAME
# coordinates), which `set -a` would export truncated at the first space (#293).
if [ -f "$project_path/.env" ]; then
  source "$project_path/.env"
  export GITHUB_TOKEN
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

# ── the environment, from the one writer (#287 -> #309) ──────────────────────────────
# The original þing-01 block lived here and inlined the registry read. Its content now
# lives in tools/credentials/platform_env.sh, which bootstrap.sh sources too; what remains
# here is the call sequence. The #293 correction it carried is preserved in that file --
# a removal must name what it was carrying.
# Coordinates and the secret, from tools/credentials/platform_env.sh — the one writer (#309).
# Every failure below is fatal and names itself; none of it is recoverable here.
#
# REMOVED in #314 and kept out deliberately: `_platform001_coordinate_state()`, which on a
# missing registry announced "Coordinates ARE present in the environment (exported outside
# this script)". That was FALSE — it tested SHELL variables, which `source .env` sets and
# nothing exports, so the python child never saw them. It was written to avoid asserting an
# unverified environment fact and asserted one by checking the wrong scope. If a
# coordinate-state reporter ever returns, it must read the EXPORTED environment.
platform_env_assert_no_env_conflicts || exit 1
platform_env_export_secret          || exit 1
platform_env_export_coordinates     || exit 1
platform_env_validate               || exit 1
echo "Appwrite environment loaded: coordinates from the registry, secret from the operator slot."

echo "Running $script_path/main.py "
python $script_path/main.py "$@"
