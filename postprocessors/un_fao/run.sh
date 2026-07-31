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
APPWRITE_REGISTRY="${APPWRITE_REGISTRY:-$project_path/../views-appwrite/docs/ADRs/platform/coordinate_registry.toml}"

# Report the ACTUAL coordinate state rather than asserting one. The registry is not the only way
# coordinates can arrive — an operator's shell, a wrapper or an EnvironmentFile may already have
# exported them, which is exactly how the secret arrives. Claiming "NOT set" without looking would
# repeat the mistake this change corrects (#293).
_platform001_coordinate_state() {
  if [ -n "$APPWRITE_ENDPOINT" ] && [ -n "$APPWRITE_DATASTORE_PROJECT_ID" ] \
     && [ -n "$APPWRITE_UNFAO_BUCKET_ID" ]; then
    echo "  Coordinates ARE present in the environment (exported outside this script) — continuing." >&2
  else
    echo "  Coordinates are NOT in the environment either — the run will fail at the datastore boundary." >&2
    echo "  Note: the .env sourced earlier does not supply them; its values are not exported (#293)." >&2
  fi
}

if [ -f "$APPWRITE_REGISTRY" ]; then
  # Failures are reported, not swallowed (#293): a silent fallback here degrades to a path that
  # may supply nothing, because the sourced .env coordinates are not exported either.
  _reg_err="$(mktemp "${TMPDIR:-/tmp}/registry_to_env.XXXXXX")"
  trap 'rm -f "$_reg_err"' EXIT
  if _coords="$(python "$project_path/tools/credentials/registry_to_env.py" "$APPWRITE_REGISTRY" 2>"$_reg_err")"; then
    while IFS= read -r _cl; do
      [ -z "$_cl" ] && continue
      export "${_cl%%=*}=${_cl#*=}"
    done <<< "$_coords"
    echo "PLATFORM-001: Appwrite coordinates read from the registry (secret stays the operator slot)."
  else
    echo "PLATFORM-001 WARNING: registry read FAILED — coordinates not set BY THE REGISTRY." >&2
    echo "  registry: $APPWRITE_REGISTRY" >&2
    echo "  python:   $(python -V 2>&1)  (registry_to_env.py needs 3.11+ for tomllib)" >&2
    sed 's/^/  /' "$_reg_err" >&2
    _platform001_coordinate_state
  fi
  rm -f "$_reg_err"; trap - EXIT
else
  echo "PLATFORM-001 WARNING: registry NOT FOUND at $APPWRITE_REGISTRY — coordinates not set BY THE REGISTRY." >&2
  _platform001_coordinate_state
fi

echo "Running $script_path/main.py "
python $script_path/main.py "$@"
