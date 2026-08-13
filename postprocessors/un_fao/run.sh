#!/usr/bin/env bash
# The UN FAO delivery launcher.
#
# The delivery protocol lives in tools/launcher/postprocessor.sh (ADR-022) — registry
# first, then conda, then the #294 capability assertion, then the environment. This file
# carries only what is genuinely FAO's: which conda environment, and which
# views-postprocessing build.

# Shared with un_crafd: both install the same package into the same prefix.
POSTPROCESSOR_ENV_NAME="views-postprocessing"

# An immutable tag, not a branch (#364). Two reasons, both about the FAO delivery:
#
#   1. `main` moved on 2026-08-13 at 06:33, hours before a live delivery. A pin that can
#      change between reading it and running it is not a pin.
#   2. Tag 1.1.0 is the first merged build carrying views-postprocessing#222 — the C-79
#      fix, `if success is not True:` replacing `if success is False:`. The old form is
#      fail-OPEN: a None result passes as success, leaving an orphan in the partner bucket
#      with no metadata document, invisible to both consumer APIs, while the run exits 0.
#      That is not hypothetical — it happened here at 19:41 on 2026-07-27 (see
#      logs/views_pipeline_ERROR.log, and register C-135).
#
# `tools/launcher/postprocessor.sh` does NOT check that the install succeeded (no set -e,
# no `|| return 1` on the pip line). A failed install silently leaves the previously
# installed build in place, and the #294 capability assertion still passes because that
# build also has contract/wire. So verify the installed build, do not infer it:
#
#   python -c "import views_postprocessing, pathlib; \
#     print(pathlib.Path(views_postprocessing.__file__).parent / 'unfao/managers/unfao.py')"
#   grep -n 'success is not True' <that path>
VIEWS_POSTPROCESSING_PIN="1.1.0"

script_path=$(dirname "$(realpath "$0")")
# shellcheck source=../../tools/launcher/postprocessor.sh
. "$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )/tools/launcher/postprocessor.sh"

postprocessor_launch "$@"
