#!/usr/bin/env bash
# shellcheck shell=bash
#
# tools/launcher/postprocessor.sh — the delivery-protocol body every postprocessor runs.
#
# Governed by ADR-022. Sourced, never executed — like tools/credentials/platform_env.sh,
# which is the same shape and the reason this file has no executable bit.
#
# WHY THIS EXISTS. `postprocessors/*/run.sh` changes for two unrelated reasons: because a
# partner is different (which environment, which pin), and because the delivery protocol
# is different (registry first, then conda, then the capability assertion, then the
# environment). The first varies per launcher; the second must not. Copying the second
# means a protocol fix has to be hand-applied once per partner, and the first one that is
# missed fails silently — which is exactly the scar views-postprocessing recorded when it
# cloned `unfao/` into `crafd/` (their #211: "every partner-scoped guard was scoped to ONE
# partner").
#
# WHAT A CALLER SUPPLIES, before sourcing this file:
#
#   script_path                 the launcher's own directory
#   POSTPROCESSOR_ENV_NAME      conda env under envs/ (partners may share one)
#   VIEWS_POSTPROCESSING_PIN    git ref for the views-postprocessing install
#
# and then calls `postprocessor_launch "$@"`.
#
# EVERY STEP BELOW IS A SCAR. Read the comment before reordering anything.

postprocessor_launch() {
  local project_path env_path wire_declared config_meta

  project_path="$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )"
  env_path="$project_path/envs/$POSTPROCESSOR_ENV_NAME"
  config_meta="$script_path/configs/config_meta.py"

  if [[ "$OSTYPE" == "darwin"* ]]; then
    # libomp sits in Homebrew's prefix on macOS and is not on the default search paths.
    # Exported for THIS run only: the values are needed while the delivery runs, and a
    # script named "run this postprocessor" should not rewrite the user's shell profile.
    # Persisting them is bootstrap.sh's job (#311), not a launcher's (#310, #384).
    export LDFLAGS="-L/opt/homebrew/opt/libomp/lib $LDFLAGS"
    export CPPFLAGS="-I/opt/homebrew/opt/libomp/include $CPPFLAGS"
    export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
  fi

  # ── the environment contract lives in ONE place (#309) ─────────────────────────────
  # tools/credentials/platform_env.sh is the only writer of Appwrite coordinates and the
  # secret. This file keeps what is genuinely the launcher's own — macOS notes, conda
  # lifecycle, pip install — and borrows nothing else.
  # shellcheck source=../credentials/platform_env.sh
  . "$project_path/tools/credentials/platform_env.sh"

  # Registry existence FIRST — before conda, before pip. Continuing without it does not
  # cost a warning, it moves the failure to the datastore boundary minutes later,
  # describing a symptom rather than a cause, in front of whoever is least equipped to
  # trace it back to a checkout layout (#308). Parsing needs 3.11, so that half is
  # checked after conda below.
  platform_env_require_registry || return 1

  # GITHUB_TOKEN only, and only because the pip install below needs it before anything
  # else runs. The Appwrite secret is NOT exported here: `platform_env_export_secret`
  # owns it, and doing it in both places would reinstate the second writer #309 exists
  # to remove.
  #
  # Sourcing sets SHELL variables; only what is `export`ed reaches a child process. Do
  # NOT replace this with `set -a` — .env carries unquoted values containing spaces (the
  # *_NAME coordinates), which `set -a` would export truncated at the first space (#293).
  if [ -f "$project_path/.env" ]; then
    source "$project_path/.env"
    export GITHUB_TOKEN
  fi

  eval "$(conda shell.bash hook)"

  if [ -d "$env_path" ]; then
    echo "Conda environment already exists at $env_path. Checking dependencies..."
    # Checked here too, not only on the create path (#392) — this is the branch that
    # actually runs. A silent activate failure leaves python/pip on the base interpreter
    # (3.10, no tomllib), so every install below lands in the wrong prefix and the first
    # visible symptom is platform_env_load failing to parse the registry, hundreds of
    # lines later and blaming the wrong thing.
    conda activate "$env_path" || {
      echo "FATAL: could not activate the existing environment at $env_path (#392)." >&2
      echo "  Continuing would install into, and run from, the base interpreter." >&2
      return 1
    }
    echo "$env_path is activated"

    missing_packages=$(pip install --dry-run -r "$script_path/requirements.txt" 2>&1 | grep -v "Requirement already satisfied" | wc -l)
    if [ "$missing_packages" -gt 0 ]; then
      echo "Installing missing or outdated packages..."
      # ── #392: a failed dependency install STOPS the run ───────────────────────────
      # There is no `set -e` here (see the header: this body is sourced, and set -e in a
      # sourced function would change the caller's shell). So every install that matters
      # carries its own check.
      #
      # Without this the launcher printed pip's error and carried on — through the #294
      # capability assertion, into main.py — with the data source simply absent. Observed
      # 2026-08-12 on un_crafd: "No matching distribution found for views-datafactory",
      # then "Capability check: ... passed", then the run. The queryset declares
      # `"source": "views-datafactory"`, so the historical leg had no data at all.
      pip install -r "$script_path/requirements.txt" || {
        echo "FATAL: could not install $script_path/requirements.txt (#392)." >&2
        echo "  The run is stopped here rather than continuing without the dependency." >&2
        echo "  A postprocessor whose queryset declares a source it cannot import has no" >&2
        echo "  historical leg, and every check after this point would still pass." >&2
        return 1
      }
    else
      echo "All packages are up-to-date."
    fi
  else
    echo "Creating new Conda environment at $env_path..."
    conda create --prefix "$env_path" python=3.11 -y || {
      echo "FATAL: could not create the conda environment at $env_path (#392)." >&2
      return 1
    }
    conda activate "$env_path" || {
      echo "FATAL: could not activate $env_path (#392). Without this, python and pip stay" >&2
      echo "  on the base interpreter and everything below installs into the wrong place." >&2
      return 1
    }
    pip install -r "$script_path/requirements.txt" || {
      echo "FATAL: could not install $script_path/requirements.txt into the new" >&2
      echo "  environment at $env_path (#392)." >&2
      return 1
    }
  fi
  echo "Installing views-postprocessing @ $VIEWS_POSTPROCESSING_PIN ..."
  # Reported where it happens, not inferred two steps later (ADR-020). The #385 check
  # below would also catch a failed install — but only when the stale build's ref differs
  # from the pin, and it would blame the pin rather than the install.
  pip install "git+https://${GITHUB_TOKEN}@github.com/views-platform/views-postprocessing.git@${VIEWS_POSTPROCESSING_PIN}" || {
    echo "FATAL: could not install views-postprocessing @ $VIEWS_POSTPROCESSING_PIN (#392)." >&2
    echo "  Not continuing on whatever build happens to be installed: the #294 capability" >&2
    echo "  assertion passes on a stale build too, so this would otherwise run green." >&2
    return 1
  }

  # ── #385: the pin is VERIFIED, not assumed ──────────────────────────────────────────
  # views-postprocessing declares a STATIC version in pyproject.toml, so pip treats the
  # requirement as satisfied whenever any build of that version is installed and skips the
  # rebuild. Moving the pin to a different COMMIT of the same version is therefore a
  # silent no-op on any machine that has run this launcher before. Found during the first
  # CRAF'd delivery (views-crafdapi#44).
  #
  # Pinning a TAG largely dodges this — a tag maps 1:1 to a version, so the two move
  # together — and both launchers now do (#391/#364). But that is a property of today's
  # pins, not of the mechanism, and the next person to pin a raw commit gets the old
  # behaviour with no warning.
  #
  # pip records what it actually installed in direct_url.json. Compare it. This is exactly
  # the manual check performed before the 2026-08-13 FAO delivery; a check you have to
  # remember to run by hand is one you will one day forget.
  installed_ref="$(python - <<'PY' 2>/dev/null
import glob, json, sysconfig
hits = glob.glob(f"{sysconfig.get_paths()['purelib']}/views_postprocessing-*.dist-info/direct_url.json")
print(json.load(open(hits[0]))["vcs_info"].get("requested_revision", "") if hits else "")
PY
)"
  if [ -z "$installed_ref" ]; then
    echo "WARNING: could not read the installed views-postprocessing ref from" >&2
    echo "  direct_url.json — cannot confirm the pin was applied. Continuing, because a" >&2
    echo "  missing provenance file is not itself evidence of a wrong build (#385)." >&2
  elif [ "$installed_ref" != "$VIEWS_POSTPROCESSING_PIN" ]; then
    echo "FATAL: the pin was NOT applied (#385)." >&2
    echo "  requested: $VIEWS_POSTPROCESSING_PIN" >&2
    echo "  installed: $installed_ref" >&2
    echo "  pip skipped the rebuild because the version string did not change. Force it:" >&2
    echo "    pip install --force-reinstall --no-deps \\" >&2
    echo "      'git+https://github.com/views-platform/views-postprocessing.git@${VIEWS_POSTPROCESSING_PIN}'" >&2
    echo "  Do NOT proceed: the build about to run is not the one this launcher declares." >&2
    return 1
  else
    echo "Pin verified: views-postprocessing @ $installed_ref is what is installed."
  fi

  # ── #294: does the build we just installed do what config_meta DECLARES? ────────────
  # Whether the installed package can produce the artifact this postprocessor declares is
  # a fact about someone else's repository on the day you run, and nothing above checked
  # it.
  #
  # That is not hypothetical. On 2026-07-31 `@main` was 208 commits behind, carried zero
  # wire modules, and still ran to completion: it would have ignored `wire_contract: True`
  # and delivered the LEGACY parquet instead of the ADR-013 contract dialect — green run,
  # wrong artifact, and the partner API then serving the previous month.
  #
  # So: read what config_meta declares, then assert the installed package can honour it.
  # Declaration read through importlib, never grep — a commented-out key looks identical
  # to a live one to a regex, which is register entry C-57.
  wire_declared="$(python - "$config_meta" <<'PY' 2>/dev/null || echo unknown
import importlib.util, sys
spec = importlib.util.spec_from_file_location("_launcher_config_meta", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("yes" if module.get_meta_config().get("wire_contract") else "no")
PY
)"

  if [ "$wire_declared" = "yes" ]; then
    if ! python -c "import views_postprocessing.contract.wire" >/dev/null 2>&1; then
      echo "ERROR: config_meta declares wire_contract: True, but the installed" >&2
      echo "       views-postprocessing cannot import views_postprocessing.contract.wire." >&2
      echo "       This build CANNOT produce the ADR-013 contract dialect. Left to run, it" >&2
      echo "       would exit 0 having delivered the legacy artifact, and the partner API" >&2
      echo "       would keep being served the previous month (#294)." >&2
      echo "       Installed from: github.com/views-platform/views-postprocessing @${VIEWS_POSTPROCESSING_PIN}" >&2
      echo "       Fix: point the pin at a build that carries the wire, or set" >&2
      echo "       wire_contract: False if the legacy artifact is genuinely what you want." >&2
      return 1
    fi
    echo "Capability check: wire_contract declared and views_postprocessing.contract.wire importable."
  elif [ "$wire_declared" = "unknown" ]; then
    echo "Capability check SKIPPED: could not read wire_contract from config_meta." >&2
    echo "  Not fatal — this check only ever ADDS a failure mode, it must not invent one." >&2
  fi

  # ── the environment, from the one writer (#287 -> #309) ─────────────────────────────
  # AFTER conda activate, deliberately: the registry parse needs tomllib (3.11+) and the
  # box's base interpreter is 3.10.
  #
  # REMOVED in #314 and kept out deliberately: `_platform001_coordinate_state()`, which on
  # a missing registry announced "Coordinates ARE present in the environment (exported
  # outside this script)". That was FALSE — it tested SHELL variables, which `source .env`
  # sets and nothing exports, so the python child never saw them. If a coordinate-state
  # reporter ever returns, it must read the EXPORTED environment (C-112).
  #
  # One call, one order: platform_env_load is the single sequence and it ends in
  # validation, which tests EXPORTED scope rather than shell scope. Calling the pieces
  # here and in a different order in bootstrap.sh is how the two drifted.
  platform_env_load || return 1
  echo "Appwrite environment loaded: coordinates from the registry, secret from the operator slot."

  echo "Running $script_path/main.py "
  python "$script_path/main.py" "$@"
}
