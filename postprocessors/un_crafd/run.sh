#!/usr/bin/env bash
# The CRAF'd delivery launcher.
#
# The delivery protocol lives in tools/launcher/postprocessor.sh (ADR-022) — registry
# first, then conda, then the #294 capability assertion, then the environment. This file
# carries only what is genuinely CRAF'd's: which conda environment, and which
# views-postprocessing build.

# Shared with un_fao: both install the same package into the same prefix.
POSTPROCESSOR_ENV_NAME="views-postprocessing"

# An IMMUTABLE tag. views-models#294 is the scar: `@main` was once 208 commits behind,
# carried no wire modules, and would have run green while delivering the legacy artifact.
#
# Was `3286eabe...` (views-postprocessing origin/main as of 2026-08-11, the first state
# containing views_postprocessing/crafd/) with the note "move it to a tag the day
# views-postprocessing cuts one that contains crafd". **That day was 2026-08-13**: tag
# `1.1.0` (`1e21d723`) carries crafd AND views-postprocessing#222 — the C-79 fix,
# `if success is not True:` replacing the fail-open `if success is False:`.
#
# Moving it is not optional housekeeping. Both launchers install into the SAME prefix
# (POSTPROCESSOR_ENV_NAME above, shared with un_fao). Leaving this on 3286eab means any
# un_crafd run — the views-crafdapi D4 dry run, for instance — DOWNGRADES the shared
# environment to the fail-open build, and `tools/launcher/postprocessor.sh` does not check
# that a later reinstall succeeded (no set -e, no `|| return 1`), while the #294 capability
# assertion passes on the stale build anyway. That is a live path back to C-135 on the
# armed FAO delivery. Same shared-prefix argument as the numpy pin in requirements.txt.
#
# Moved again to `1.1.1` on 2026-08-24 (#403), for the same shared-prefix reason one step
# on: 1.1.0 fixed the UPLOAD gap and still carried a DOWNLOAD one (views-postprocessing
# #268) — the defect that killed this launcher's first delivery attempt on 2026-08-13.
# Both launchers moved together; leaving either on 1.1.0 re-creates the downgrade path
# described above, now with un_crafd armed as well.
VIEWS_POSTPROCESSING_PIN="1.1.1"

script_path=$(dirname "$(realpath "$0")")
# shellcheck source=../../tools/launcher/postprocessor.sh
. "$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )/tools/launcher/postprocessor.sh"

postprocessor_launch "$@"
