#!/usr/bin/env bash
# bootstrap.sh — set this platform up on a machine that has never run it (#311).
#
# Governed by ADR-018 (docs/ADRs/018_environment_single_writer.md).
#
#     ./bootstrap.sh
#
# No arguments. No companion document. If you needed either, this script has failed at
# its actual job and the structure underneath is still wrong.
#
# WHY A SCRIPT AND NOT A PAGE. Prose describing setup rots silently: nothing fails when it
# stops being true, and whoever discovers that is the person least equipped to fix it.
# This platform's own entry point currently points at an empty link where the technical
# guide should be. A script either runs or it does not.
#
# WHAT IT ASKS YOU FOR: one secret, and zero coordinates. Coordinates come from the
# registry (`tools/credentials/platform_env.sh`, #309); asking a human to retype an address that is
# already declared somewhere is how the two copies drift.
#
# IT DOES NOT: create conda environments or install packages. Each `run.sh` still owns its
# own environment, and pretending otherwise here would mean this script quietly deciding
# which of ~130 environments you wanted.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOTENV="$REPO_ROOT/.env"
SECRET_NAME="APPWRITE_DATASTORE_API_KEY"

_step()  { printf '\n\033[1m── %s\033[0m\n' "$1"; }
_ok()    { printf '   ok    %s\n' "$1"; }
_info()  { printf '   ....  %s\n' "$1"; }
_fail()  { printf '   FAIL  %s\n' "$1" >&2; }

# ── 1. one-time machine setup ─────────────────────────────────────────────────────────
# One-time setup belongs in one-time setup (#311). NOTE: the ~130 per-run scripts still
# carry their own copy of this block — removing it from them is #310's scope, which
# touches 131 files. So this ADDS the canonical home; it has not yet replaced them, and
# saying "moved" before that lands would be a claim the tree does not support.
_step "machine setup"
if [[ "$OSTYPE" == "darwin"* ]]; then
  _added=0
  for _line in \
    'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"' \
    'export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"' \
    'export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"'
  do
    if ! grep -qF "$_line" ~/.zshrc 2>/dev/null; then
      echo "$_line" >> ~/.zshrc; _added=$((_added + 1))
    fi
  done
  if [ "$_added" -gt 0 ]; then _ok "macOS libomp flags added to ~/.zshrc ($_added new)"
  else _ok "macOS libomp flags already present in ~/.zshrc"; fi
else
  _ok "not macOS — no libomp setup needed"
fi

# ── 2. an interpreter that can read the registry ──────────────────────────────────────
# tomllib is 3.11+. Checked before anything depends on it, so the failure names the cause
# rather than surfacing as a traceback three steps later.
_step "interpreter"
if ! python -c 'import tomllib' >/dev/null 2>&1; then
  _fail "the \`python\` on your PATH cannot import tomllib (needs 3.11+)."
  printf '         found: %s\n' "$(python -V 2>&1)" >&2
  printf '         Activate an environment with Python 3.11+ and re-run. Every run.sh\n' >&2
  printf '         builds its own; `conda activate` any of them, or use the base 3.11.\n' >&2
  exit 1
fi
_ok "$(python -V 2>&1) can read the registry"

# ── 3. coordinates — from the registry, never from you ────────────────────────────────
_step "coordinates"
# shellcheck source=tools/credentials/platform_env.sh
. "$REPO_ROOT/tools/credentials/platform_env.sh"

platform_env_require_registry || exit 1
_ok "registry found: $(platform_env_registry_path)"

# Prime the cache HERE, in this shell. The steps below each call
# platform_env_coordinates through `$(...)`, and a cache written inside a command
# substitution evaporates with the subshell — so without this, bootstrap spawns the
# registry reader four times while platform_env_load alone spawns it once. The caching
# win is not automatic; it belongs to whoever primes it.
platform_env_prime_coordinate_cache || exit 1

platform_env_assert_no_env_conflicts || exit 1
_ok ".env declares no coordinate the registry owns"

platform_env_export_coordinates || exit 1
_ok "$(platform_env_coordinates | wc -l) coordinates exported from the registry"

# ── 4. the one secret ─────────────────────────────────────────────────────────────────
# SAFETY RULES, deliberately narrow because this file holds a live credential:
#   * never read, print or log the existing value — presence only
#   * never rewrite .env; APPEND only, and only when the key is absent
#   * back up before touching it at all
#   * `read -rs` so the value is never echoed to the terminal or a CI log
_step "secret"
# The NON-FATAL probe, deliberately. platform_env_export_secret is fatal when there is no
# .env — correct for a run-time launcher, wrong here, because "there is no .env yet" is the
# ordinary state of the machine this script exists to set up. Calling it here printed a
# FATAL telling the user to run ./bootstrap.sh while they were running ./bootstrap.sh.
if platform_env_secret_available; then
  platform_env_export_secret || exit 1
fi

if [ -n "${!SECRET_NAME:-}" ]; then
  # Presence only. The character count was here and is not "presence" — a length
  # narrows the search space and would be logged in plaintext CI output.
  _ok "$SECRET_NAME already present — not changed"
elif [ ! -t 0 ]; then
  # Non-interactive (CI, a pipe). Prompting would hang forever; say so instead.
  _fail "$SECRET_NAME is not set and stdin is not a terminal, so it cannot be prompted for."
  printf '         Set it in the environment, or run this script interactively.\n' >&2
  exit 1
else
  _info "$SECRET_NAME is not set. It is the ONLY value you need to supply."
  _info "Get it from the Appwrite console; it will not be echoed."
  printf '   %s: ' "$SECRET_NAME"
  read -rs _secret; echo
  if [ -z "$_secret" ]; then
    _fail "nothing entered — $SECRET_NAME is required."
    exit 1
  fi
  if [ -f "$DOTENV" ]; then
    cp -p "$DOTENV" "$DOTENV.bak-$(date +%Y%m%d%H%M%S)"
    _info "backed up existing .env before appending"
  fi
  printf '%s=%s\n' "$SECRET_NAME" "$_secret" >> "$DOTENV"
  unset _secret
  chmod 600 "$DOTENV"
  _ok "$SECRET_NAME appended to .env (mode 600); no existing line was modified"
  platform_env_export_secret || exit 1
fi

# ── 5. validate — the whole point ─────────────────────────────────────────────────────
_step "validation"
# platform_env_load is the single documented sequence; it re-runs the earlier steps
# (idempotent) and ends in validation, which tests EXPORTED scope — the thing a child
# process actually inherits — rather than shell scope (C-112).
if ! platform_env_load; then
  _fail "the environment is incomplete. Nothing above fixed it; see the names listed."
  exit 1
fi
_ok "every required variable resolves and will reach a child process"

_step "done"
printf '   This machine can now reach the Appwrite seam.\n'
printf '   Coordinates come from the registry; the secret lives only in .env.\n'
printf '   Next: run a model or postprocessor via its own run.sh, which builds its\n'
printf '   own conda environment.\n\n'
