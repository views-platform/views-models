#!/usr/bin/env bash

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

unset ZSH
unset ZSH_VERSION
unset ZSH_NAME
export SHELL=/bin/bash

run_folder () {
    local folder="$1"
    local abs_path="$BASE_DIR/$folder"

    echo "====================================="
    echo "Running: $folder"
    echo "====================================="

    (
        cd "$abs_path"
        if [[ "$folder" == ensembles/* ]]; then
            bash run.sh -m
        else
            bash  run.sh
        fi
    )

    echo "Finished: $folder"
    echo ""
}

run_folder "ensembles/pink_ponyclub"
run_folder "ensembles/skinny_love"
run_folder "ensembles/rude_boy"
run_folder "ensembles/first_love"
run_folder "postprocessors/un_fao"

echo "All monthly runs completed."