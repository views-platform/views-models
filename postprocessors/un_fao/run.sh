#!/usr/bin/env bash
# The UN FAO delivery launcher.
#
# The delivery protocol lives in tools/launcher/postprocessor.sh (ADR-022) — registry
# first, then conda, then the #294 capability assertion, then the environment. This file
# carries only what is genuinely FAO's: which conda environment, and which
# views-postprocessing build.

# Shared with un_crafd: both install the same package into the same prefix.
POSTPROCESSOR_ENV_NAME="views-postprocessing"

# A MOVING pointer, and a known defect — views-models#364 tracks pinning it to something
# immutable. Left as-is here so that issue owns the change to a live delivery rather than
# this extraction doing it as a side effect. `un_crafd` is pinned; copy its shape.
VIEWS_POSTPROCESSING_PIN="main"

script_path=$(dirname "$(realpath "$0")")
# shellcheck source=../../tools/launcher/postprocessor.sh
. "$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )/tools/launcher/postprocessor.sh"

postprocessor_launch "$@"
