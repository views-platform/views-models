#!/usr/bin/env bash
# The CRAF'd delivery launcher.
#
# The delivery protocol lives in tools/launcher/postprocessor.sh (ADR-022) — registry
# first, then conda, then the #294 capability assertion, then the environment. This file
# carries only what is genuinely CRAF'd's: which conda environment, and which
# views-postprocessing build.

# Shared with un_fao: both install the same package into the same prefix.
POSTPROCESSOR_ENV_NAME="views-postprocessing"

# An IMMUTABLE commit, not a branch. views-models#294 is the scar: `@main` was once 208
# commits behind, carried no wire modules, and would have run green while delivering the
# legacy artifact. `3286eab` is views-postprocessing origin/main as of 2026-08-11 and is
# the first state containing views_postprocessing/crafd/ — the package this launcher
# imports.
#
# NOT a tag, because no tag carries crafd: the only tag is 1.0.0 and it predates the
# package (verified `git ls-tree -r 1.0.0 | grep -c crafd` -> 0, and there are no GitHub
# releases). A commit is immutable, which is the property the pin is for. Move it to a
# tag the day views-postprocessing cuts one that contains crafd.
VIEWS_POSTPROCESSING_PIN="3286eabeaebc74eabf4bb86e71998a79c6d507a9"

script_path=$(dirname "$(realpath "$0")")
# shellcheck source=../../tools/launcher/postprocessor.sh
. "$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )/tools/launcher/postprocessor.sh"

postprocessor_launch "$@"
