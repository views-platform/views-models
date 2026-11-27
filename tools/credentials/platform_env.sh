#!/usr/bin/env bash
# platform_env.sh — the ONE writer of this platform's Appwrite environment (#309, C-48).
#
# Governed by ADR-018 (docs/ADRs/018_environment_single_writer.md), which records WHY —
# including the two occasions this exact blind spot shipped. Read that before changing the
# contract; the reasoning is not in the commit log.
#
# Source it; it defines functions and does nothing on its own:
#
#     . "$project_path/tools/credentials/platform_env.sh"
#     platform_env_load        # the whole contract, in the one correct order
#
# or, if you genuinely need a single step, the pieces it runs, in this order:
#
#     platform_env_require_registry
#     platform_env_assert_no_env_conflicts
#     platform_env_export_coordinates
#     platform_env_export_secret
#     platform_env_validate
#
# The header, `platform_env_load` and the two callers previously disagreed about this
# order, and `platform_env_load` omitted validation while claiming to be "everything the
# platform needs". One sequence now, documented once, used by both callers.
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
# Memoised per shell. platform_env_load runs three functions that each need the registry,
# and before this each spawned its own `python registry_to_env.py` — three subprocess
# launches and three TOML parses per launcher invocation, across ~130 launchers. The read
# is pure, so caching it is safe; APPWRITE_REGISTRY changing mid-run is not a supported
# scenario, and the cache is per-process.
PLATFORM_ENV_COORDS_CACHE=""
platform_env_coordinates() {
  local root registry err out status
  if [ -n "$PLATFORM_ENV_COORDS_CACHE" ]; then
    printf '%s\n' "$PLATFORM_ENV_COORDS_CACHE"
    return 0
  fi
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
  PLATFORM_ENV_COORDS_CACHE="$out"
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
# real production failure. Also recorded in ADR-018.
#
# ──────────────────────────────────────────────────────────────────────────────────────
# THE ONLY SANCTIONED WAY TO ASK "WILL THE CHILD SEE THIS?" (C-112).
#
# `[ -n "$VAR" ]` cannot distinguish an EXPORTED variable from a shell-local one, and the
# consumer is always a child process. This blind spot has now shipped twice in four days:
# once as `_platform001_coordinate_state()` announcing "coordinates ARE present" about
# unexported values (#314), and once as this very function's first draft, whose guard
# returned early because `un_fao/run.sh` sources `.env` for GITHUB_TOKEN twenty lines
# earlier — so the secret was a shell variable, the guard saw a value, the `export` never
# ran, and the child got nothing while every check reported success.
#
# `export -p` lists exactly what a child will inherit. Nothing else answers the question.
# `compgen -e` lists exported variable NAMES, so this never parses a value or a format.
#
# The first version grepped `export -p` for `^declare -x NAME=`, which is a false negative
# for anything exported AND readonly — bash prints `declare -rx NAME=` — and would have
# reported a variable the child demonstrably receives as "will NOT reach the child". That
# is the same category of error as the bug it was written to fix: a check that answers a
# question adjacent to the one asked.
platform_env_is_exported() {
  compgen -e | grep -qxF "$1"
}

# Non-fatal probe: is the secret obtainable at all? Silent, returns 0/1, changes nothing.
#
# It exists because `bootstrap.sh` must ask this question BEFORE deciding to prompt, and
# the fatal version cannot answer it: on a first-ever machine there is no `.env`, so
# `platform_env_export_secret` printed "FATAL: ... does not exist. Run ./bootstrap.sh" —
# at the person who was running ./bootstrap.sh. A setup script whose first output is a
# false FATAL and circular advice has failed at the one job #311 gave it.
platform_env_secret_available() {
  local dotenv
  [ -n "${!PLATFORM_ENV_SECRET_NAME:-}" ] && return 0
  dotenv="$(platform_env_dotenv_path)"
  [ -f "$dotenv" ] || return 1
  # Strip quotes before judging emptiness: `NAME=""` is a declared-but-empty secret, and a
  # trailing-`.` regex would call it available — routing bootstrap into the fatal path and
  # re-creating the circular "Run ./bootstrap.sh" advice for a different input.
  local value
  value="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${PLATFORM_ENV_SECRET_NAME}=" "$dotenv" \
           | tail -1 | cut -d= -f2- | sed -e 's/^[[:space:]]*//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'\$/\1/")"
  [ -n "$value" ]
}

platform_env_export_secret() {
  local dotenv status
  dotenv="$(platform_env_dotenv_path)"

  # Already exported — the operator's own slot, or a wrapper. Nothing to do.
  if platform_env_is_exported "$PLATFORM_ENV_SECRET_NAME"; then
    return 0
  fi

  # Set but NOT exported: the trap above. Promote it rather than skipping.
  if [ -n "${!PLATFORM_ENV_SECRET_NAME:-}" ]; then
    export "${PLATFORM_ENV_SECRET_NAME?}"
    return 0
  fi

  if [ ! -f "$dotenv" ]; then
    echo "FATAL: $PLATFORM_ENV_SECRET_NAME is not set and $dotenv does not exist." >&2
    echo "  Run ./bootstrap.sh, or export it in your shell." >&2
    return 1
  fi

  # Do NOT swallow this. A hand-edited .env with an unterminated quote fails to source,
  # and `|| true` would leave the secret unset while reporting success — the same
  # report-success-deliver-nothing shape this file exists to prevent.
  # shellcheck disable=SC1090
  . "$dotenv" >/dev/null 2>&1
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "FATAL: $dotenv could not be sourced (exit $status)." >&2
    echo "  Usually an unterminated quote or a value with an unescaped space." >&2
    return 1
  fi

  if [ -z "${!PLATFORM_ENV_SECRET_NAME:-}" ]; then
    echo "FATAL: $dotenv does not declare $PLATFORM_ENV_SECRET_NAME." >&2
    echo "  It is the one value that file must carry. Run ./bootstrap.sh." >&2
    return 1
  fi
  export "${PLATFORM_ENV_SECRET_NAME?}"
}

# ── 4. validate ───────────────────────────────────────────────────────────────────────
# Names what is missing rather than reporting a count. The value is never rendered — a
# check that prints a credential to prove it found one has published it.

# Tests EXPORTED scope, not shell scope (C-112). Validating with `[ -n "$name" ]` would
# pass on values the child will never receive, which is the failure this function is the
# last line of defence against.
platform_env_validate() {
  local coords name missing=""
  coords="$(platform_env_coordinates)" || return 1
  for name in $(echo "$coords" | cut -d= -f1) "$PLATFORM_ENV_SECRET_NAME"; do
    platform_env_is_exported "$name" || missing="$missing $name"
  done
  if [ -n "$missing" ]; then
    echo "FATAL: the environment is incomplete — these will NOT reach the child" >&2
    echo "  process, either unset or set-but-not-exported:$missing" >&2
    echo "  coordinates come from: $(platform_env_registry_path)" >&2
    echo "  the secret comes from: $(platform_env_dotenv_path) (or your shell)" >&2
    return 1
  fi
}

# The whole contract, in the documented order, INCLUDING validation. A caller that trusts
# this alone must get a complete environment or a non-zero exit — the previous version
# omitted `platform_env_validate` while its comment claimed to be "everything the platform
# needs", and used a different order than the header documents. Both callers now use this,
# so there is one order rather than two.
# Populate the cache in the CALLER's shell. `coords="$(platform_env_coordinates)"` runs the
# function in a subshell, so a cache assignment made inside it evaporates — which is what
# the first version of this memoisation did: it cost a line of code and saved nothing,
# silently. Same scope-blindness as C-112, in the fix for C-112. Assigning here works
# because this function is called directly, and subshells inherit the value.
platform_env_prime_coordinate_cache() {
  local out
  out="$(platform_env_coordinates)" || return 1
  PLATFORM_ENV_COORDS_CACHE="$out"
}

platform_env_load() {
  platform_env_require_registry           || return 1
  platform_env_prime_coordinate_cache     || return 1
  platform_env_assert_no_env_conflicts || return 1
  platform_env_export_coordinates      || return 1
  platform_env_export_secret           || return 1
  platform_env_validate                || return 1
}
