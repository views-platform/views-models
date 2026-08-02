#!/usr/bin/env bash
# platform_env.sh — the ONE writer of this platform's Appwrite environment (#309, C-48).
#
# Source it; it defines functions and does nothing on its own:
#
#     . "$project_path/tools/credentials/platform_env.sh"
#     platform_env_require_registry
#     platform_env_assert_no_env_conflicts
#     platform_env_export_coordinates
#     platform_env_export_secret
#     platform_env_validate
#
# WHAT THIS FILE IS FOR, AND WHAT IT DELIBERATELY IS NOT.
#
# The problem it solves is a data race, not untidiness. Before #314, two blocks wrote the
# same names: `source .env` and the registry loop. The registry won because it ran second.
# Reverse them and the semantics invert, silently, with no test failing. Extracting the
# logic without removing the second writer would have tidied the race rather than ended it.
#
# So the rule this file enforces, and the whole of its value:
#
#     COORDINATES COME FROM THE REGISTRY. THE SECRET COMES FROM THE OPERATOR.
#     Nothing else writes either, and a `.env` that tries is an error, not a tiebreak.
#
# NOT here, on purpose (#309 names this): conda lifecycle, pip installs, macOS libomp
# setup. `postprocessors/un_fao/run.sh` had five reasons to change; only two of them are
# about the environment, and only those two moved. One-time machine setup lives in
# `bootstrap.sh` (#311).
#
# Interpreter: reading the registry needs `tomllib`, so Python 3.11+. This file does not
# create or activate an environment — it uses whatever `python` the caller has arranged
# and fails loud if that one cannot read the registry.

# ── configuration ─────────────────────────────────────────────────────────────────────
# Resolved once, overridable. The default is a relative hop to a sibling views-appwrite
# checkout; on a different layout it is simply absent, which is fatal (#308) rather than a
# warning, because continuing moves the failure to the datastore boundary minutes later
# where it describes a symptom instead of a cause.

platform_env_repo_root() {
  # The repo containing this file, however it was sourced.
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

platform_env_registry_path() {
  local root; root="$(platform_env_repo_root)"
  echo "${APPWRITE_REGISTRY:-$root/../views-appwrite/docs/ADRs/platform/coordinate_registry.toml}"
}

platform_env_dotenv_path() {
  local root; root="$(platform_env_repo_root)"
  echo "$root/.env"
}

# The one value a `.env` may still legitimately carry.
PLATFORM_ENV_SECRET_NAME="APPWRITE_DATASTORE_API_KEY"

# ── 1. the registry must resolve ──────────────────────────────────────────────────────

platform_env_require_registry() {
  local registry; registry="$(platform_env_registry_path)"
  if [ ! -f "$registry" ]; then
    echo "FATAL: the Appwrite coordinate registry does not exist." >&2
    echo "  looked for: $registry" >&2
    echo "  override with: APPWRITE_REGISTRY=/path/to/coordinate_registry.toml" >&2
    echo "" >&2
    echo "  The default is a relative hop to a sibling checkout of views-appwrite. If this" >&2
    echo "  machine lays the repositories out differently, set APPWRITE_REGISTRY." >&2
    echo "  Fatal by design (#308): the registry is the ONLY source of coordinates." >&2
    return 1
  fi
}

# Echoes `NAME=value` lines, or fails loud. Kept separate from exporting so callers can
# inspect what the registry owns without mutating their own environment.
platform_env_coordinates() {
  local root registry err out status
  root="$(platform_env_repo_root)"
  registry="$(platform_env_registry_path)"
  err="$(mktemp "${TMPDIR:-/tmp}/platform_env.XXXXXX")"

  # Capture status directly. Do NOT wrap this in `if ! cmd; then status=$?; fi` — inside
  # that branch `$?` is the status of the NEGATION (0, because the negation succeeded),
  # not of the command. The first draft of this file did exactly that and returned 0 on a
  # failed registry read, so every caller reported success while exporting nothing. That
  # is the same shape as the pipeline-core write path that logs "uploaded successfully"
  # and uploads nothing — the defect this whole seam exists to stop.
  out="$(python "$root/tools/credentials/registry_to_env.py" "$registry" 2>"$err")"
  status=$?

  if [ "$status" -ne 0 ]; then
    echo "FATAL: the coordinate registry exists but could not be read." >&2
    echo "  registry: $registry" >&2
    echo "  python:   $(python -V 2>&1)  (registry_to_env.py needs 3.11+ for tomllib)" >&2
    sed 's/^/  /' "$err" >&2
    echo "  Fatal by design (#308): half a source is not a source." >&2
    rm -f "$err"; return 1
  fi

  if [ -z "$out" ]; then
    # A readable registry that yields nothing is not a success. Silence here would export
    # zero coordinates and let validation pass on an empty set.
    echo "FATAL: the coordinate registry parsed but declared no coordinates." >&2
    echo "  registry: $registry" >&2
    rm -f "$err"; return 1
  fi

  rm -f "$err"
  printf '%s\n' "$out"
}

# ── 2. one writer: `.env` must not declare a coordinate the registry owns ─────────────

platform_env_assert_no_env_conflicts() {
  local dotenv coords owned name conflicts=""
  dotenv="$(platform_env_dotenv_path)"
  [ -f "$dotenv" ] || return 0
  coords="$(platform_env_coordinates)" || return 1
  owned="$(echo "$coords" | cut -d= -f1)"
  for name in $owned; do
    if grep -qE "^[[:space:]]*(export[[:space:]]+)?${name}=" "$dotenv"; then
      conflicts="$conflicts $name"
    fi
  done
  if [ -n "$conflicts" ]; then
    echo "FATAL: .env declares coordinates that the registry owns (#309)." >&2
    echo "  file:     $dotenv" >&2
    echo "  registry: $(platform_env_registry_path)" >&2
    echo "  both declare:$conflicts" >&2
    echo "" >&2
    echo "  Two writers to one name is a data race decided by line order, so this is an" >&2
    echo "  error rather than a precedence question. Delete these lines from .env — they" >&2
    echo "  were never exported, so nothing has ever received them from there. Keep" >&2
    echo "  $PLATFORM_ENV_SECRET_NAME: the secret is the one value .env still carries." >&2
    return 1
  fi
}

# ── 3. export ─────────────────────────────────────────────────────────────────────────

platform_env_export_coordinates() {
  local coords line
  coords="$(platform_env_coordinates)" || return 1
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    export "${line%%=*}=${line#*=}"
  done <<< "$coords"
}

# The secret, by name. NEVER `set -a`: `.env` carries unquoted values containing spaces
# (the *_NAME coordinates), which `set -a` exports truncated at the first space (#293).
#
# #293 in full, because a removal must name what it was carrying: `source` without
# `export` never reached the python child, so between views-postprocessing's load_dotenv
# removal (2026-07-28) and the fix there was NO carrier for the secret at all. That was a
# real production failure and this comment is the only place it is written down in code.
platform_env_export_secret() {
  local dotenv; dotenv="$(platform_env_dotenv_path)"
  if [ -n "${!PLATFORM_ENV_SECRET_NAME:-}" ]; then
    return 0                      # already in the environment; the operator's own slot
  fi
  [ -f "$dotenv" ] || return 0
  # shellcheck disable=SC1090
  . "$dotenv" >/dev/null 2>&1 || true
  export "${PLATFORM_ENV_SECRET_NAME?}"
}

# ── 4. validate ───────────────────────────────────────────────────────────────────────
# Names what is missing rather than reporting a count. The value is never rendered — a
# check that prints a credential to prove it found one has published it.

platform_env_validate() {
  local coords name missing=""
  coords="$(platform_env_coordinates)" || return 1
  for name in $(echo "$coords" | cut -d= -f1) "$PLATFORM_ENV_SECRET_NAME"; do
    [ -z "${!name:-}" ] && missing="$missing $name"
  done
  if [ -n "$missing" ]; then
    echo "FATAL: the environment is incomplete — missing:$missing" >&2
    echo "  coordinates come from: $(platform_env_registry_path)" >&2
    echo "  the secret comes from: $(platform_env_dotenv_path) (or your shell)" >&2
    return 1
  fi
}

# Everything the platform needs, in the one order that is correct. Callers that want the
# whole contract use this; callers that need a step use the pieces.
platform_env_load() {
  platform_env_require_registry || return 1
  platform_env_assert_no_env_conflicts || return 1
  platform_env_export_secret || return 1
  platform_env_export_coordinates || return 1
}
