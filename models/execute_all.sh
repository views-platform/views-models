#!/bin/zsh

# Get a sorted list of model directories
models=($(find . -type f -name "run.sh" -exec dirname {} \; | sort))

# Loop through the sorted directories and execute run.sh with flags
for dir in "${models[@]}"; do
  script="$dir/run.sh"
  echo "Executing $script..."
  zsh "$script" "$@"
done