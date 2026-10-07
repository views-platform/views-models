#!/usr/bin/env bash
set -euo pipefail

if [[ "${OSTYPE:-}" == "darwin"* ]]; then
  export LDFLAGS="-L/opt/homebrew/opt/libomp/lib ${LDFLAGS:-}"
  export CPPFLAGS="-I/opt/homebrew/opt/libomp/include ${CPPFLAGS:-}"
  export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${DYLD_LIBRARY_PATH:-}"
fi

script_path=$(dirname "$(realpath "$0")")
project_path="$(cd "$script_path/../../" >/dev/null 2>&1 && pwd)"
env_path="$project_path/envs/views_r2darts2"
eval "$(conda shell.bash hook)"

if [[ -d "$env_path" ]]; then
  conda activate "$env_path"
  pip install --dry-run -r "$script_path/requirements.txt" 2>&1 | grep -qv "Requirement already satisfied" && pip install -r "$script_path/requirements.txt"
else
  conda create --prefix "$env_path" python=3.11 -y
  conda activate "$env_path"
  pip install -r "$script_path/requirements.txt"
fi

python "$script_path/main.py" "$@"
