#!/usr/bin/env bash
# verify_committed.sh — run the test suite against COMMITTED state, in a working env.
#
# Why this exists (#297).
#   Two things must both be true for a test result to mean anything: the code
#   under test must be what is in git, and the environment must actually work.
#   Today neither place has both:
#
#     GitHub Actions  clean checkout ✅   working env ❌  — `Run Tests` aborts at
#                                                          COLLECTION on the published
#                                                          views_pipeline_core skew
#                                                          (no reconciliation_port), so
#                                                          zero tests run (C-73, C-80).
#     A laptop        working env ✅      clean checkout ❌ — everyone carries uncommitted
#                                                          work, which silently supplies
#                                                          values git does not have.
#
#   The gap let `violet_visitor`'s `loss_reg` sit at 'mse' in git while every working
#   tree carried 'hurdle_nb'. `pytest` was green on every machine and committed
#   `development` was red for 12 days, reproducible nowhere (#297).
#
#   This script closes it by combining the two halves: a throwaway `git worktree` at a
#   committed ref (so the code is exactly what is in git — no artifacts, no .env, no
#   envs/, nothing untracked) run with the conda env you already have.
#
# Usage
#   bash tools/audit/verify_committed.sh                     # HEAD
#   bash tools/audit/verify_committed.sh origin/development  # any ref
#   bash tools/audit/verify_committed.sh HEAD -q tests/test_datafactory_parity.py
#   VIEWS_ENV=views_pipeline bash tools/audit/verify_committed.sh
#
# Exit code is pytest's, so it composes with the merge ritual and with CI once the
# skew that blocks `Run Tests` is resolved.

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
REF="${1:-HEAD}"
if [ $# -gt 0 ]; then shift; fi   # an `&&` here would return non-zero with no args,
                                  # which is harmless today but a trap under `set -e`
ENV_NAME="${VIEWS_ENV:-views_pipeline}"

# A hard kill (SIGKILL, power loss) skips the EXIT trap and leaves a dead worktree
# registered against the repo. Harmless but it accumulates in `git worktree list`.
git -C "$REPO" worktree prune >/dev/null 2>&1 || true

WORKTREE="$(mktemp -d "${TMPDIR:-/tmp}/views-models-committed.XXXXXX")"

cleanup() {
  git -C "$REPO" worktree remove --force "$WORKTREE" >/dev/null 2>&1
  rm -rf "$WORKTREE"
}
trap cleanup EXIT

if ! git -C "$REPO" rev-parse --verify --quiet "$REF^{commit}" >/dev/null; then
  echo "verify_committed: '$REF' is not a commit-ish in $REPO" >&2
  exit 2
fi
RESOLVED="$(git -C "$REPO" rev-parse --short "$REF")"

echo "── verifying COMMITTED state ────────────────────────────────────────────"
echo "  ref:       $REF ($RESOLVED)"
echo "  worktree:  $WORKTREE   (tracked content only)"
echo "  env:       $ENV_NAME"

DIRTY="$(git -C "$REPO" status --porcelain | wc -l | tr -d ' ')"
if [ "$DIRTY" != "0" ]; then
  echo "  note:      your working tree has $DIRTY uncommitted change(s) — deliberately"
  echo "             NOT present below. That difference is the whole point."
fi

git -C "$REPO" worktree add -q --detach "$WORKTREE" "$REF" || {
  echo "verify_committed: could not create worktree" >&2; exit 2; }

# Prove the isolation rather than asserting it: nothing untracked should be here.
STRAY="$(find "$WORKTREE" \( -name '*.npy' -o -name '*.parquet' -o -name '.env' \) 2>/dev/null | wc -l | tr -d ' ')"
echo "  isolation: $STRAY artifact/.env file(s) present (expected 0)"
echo "─────────────────────────────────────────────────────────────────────────"

if command -v conda >/dev/null 2>&1 && conda env list | grep -qE "^${ENV_NAME}\s"; then
  (cd "$WORKTREE" && conda run --no-capture-output -n "$ENV_NAME" pytest "$@")
else
  echo "verify_committed: conda env '$ENV_NAME' not found — falling back to the" >&2
  echo "  ambient interpreter. Results are only as trustworthy as that env." >&2
  (cd "$WORKTREE" && pytest "$@")
fi
STATUS=$?

echo "─────────────────────────────────────────────────────────────────────────"
if [ $STATUS -eq 0 ]; then
  echo "  COMMITTED STATE GREEN at $RESOLVED"
else
  echo "  COMMITTED STATE RED at $RESOLVED (pytest exit $STATUS)"
  echo "  These failures exist in git and are invisible in a dirty working tree."
fi
exit $STATUS
