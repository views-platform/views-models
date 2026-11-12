#!/usr/bin/env bash
# GENERATED — do not edit by hand.
# Source: views_pipeline_core.templates.model.template_run_sh, applied by
# tools/scaffold/build_model_scaffold.py. A fix made here reaches one model out of
# 129; fix the template instead (views-models#310, views-pipeline-core#384).
# The only intended per-model variable is `env_path`.

if [[ "$OSTYPE" == "darwin"* ]]; then
  # libomp sits in Homebrew's prefix on macOS and is not on the default search paths.
  # Exported for THIS run only: the values are needed while the model runs, and a
  # script named "run this model" should not rewrite the user's shell profile (#384).
  export LDFLAGS="-L/opt/homebrew/opt/libomp/lib $LDFLAGS"
  export CPPFLAGS="-I/opt/homebrew/opt/libomp/include $CPPFLAGS"
  export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
fi

script_path=$(dirname "$(realpath $0)")
project_path="$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )"
env_path="$project_path/envs/views-hydranet"

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
else
  echo "Creating new Conda environment at $env_path..."
  conda create --prefix "$env_path" python=3.11 -y
  conda activate "$env_path"
  pip install -r $script_path/requirements.txt
fi

echo "Running $script_path/main.py "
python $script_path/main.py "$@"
