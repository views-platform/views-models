#!/usr/bin/env bash
set -uo pipefail

# Audit all .sh scripts in the repo for portability and correctness.
# Skips envs/ (conda internals) and .git/.
#
# Checks:
#   1. Shebang present and portable (bash or /usr/bin/env bash)
#   2. No zsh-specific builtins or syntax
#   3. Executable permission set
#   4. Trailing newline present
#   5. No shellcheck errors (if shellcheck is installed)
#   6. bash --noexec parse check (syntax valid under bash)
#   7. Explicit zsh invocations (e.g., "zsh script.sh")
#   8. Parity test: zsh vs bash --help output (optional, --parity flag)
#
# Usage:
#   bash scripts/audit_shell_health.sh             # full audit
#   bash scripts/audit_shell_health.sh --parity    # also run zsh/bash parity diff
#   bash scripts/audit_shell_health.sh --fix       # report what --fix would change (dry-run)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PARITY=false
FIX_MODE=false
for arg in "$@"; do
    case "$arg" in
        --parity) PARITY=true ;;
        --fix)    FIX_MODE=true ;;
    esac
done

PASS=0
WARN=0
FAIL=0
TOTAL=0

pass()  { PASS=$((PASS + 1)); }
warn()  { WARN=$((WARN + 1)); echo "  WARN: $1"; }
fail()  { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

ZSH_BUILTINS="setopt|zparseopts|zmodload|autoload|whence|zstyle|zle|bindkey|vared|zformat|zcompile|compadd|compdef"
ZSH_SYNTAX='=\(|print -P|print -l|<<<.*\<\(|typeset -A'

SCRIPTS=()
while IFS= read -r f; do
    SCRIPTS+=("$f")
done < <(find . -name "*.sh" -type f \
    -not -path "./envs/*" \
    -not -path "./.git/*" \
    -not -path "./node_modules/*" \
    | sort)

echo "Shell Health Audit — $(date -I)"
echo "Repository: $REPO_ROOT"
echo "Scripts found: ${#SCRIPTS[@]}"
echo ""
echo "========================================"

for script in "${SCRIPTS[@]}"; do
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "--- $script ---"
    script_ok=true

    # 1. Shebang check
    first_line=$(head -1 "$script")
    case "$first_line" in
        "#!/bin/bash"|"#!/usr/bin/env bash")
            pass
            ;;
        "#!/bin/zsh"|"#!/usr/bin/env zsh")
            fail "shebang is '$first_line' — not portable to Linux servers/CI/Docker"
            script_ok=false
            if $FIX_MODE; then
                echo "         FIX: change to #!/usr/bin/env bash"
            fi
            ;;
        "#!"*)
            warn "unusual shebang: $first_line"
            ;;
        *)
            fail "no shebang line — first line: '$first_line'"
            script_ok=false
            ;;
    esac

    # 2. zsh-specific builtins (skip self)
    if [ "$(realpath "$script")" != "$(realpath "$0")" ]; then
    if grep -Pn "\\b($ZSH_BUILTINS)\\b" "$script" 2>/dev/null | grep -v "^[0-9]*:#" | grep -q .; then
        fail "uses zsh-specific builtins:"
        grep -Pn "\\b($ZSH_BUILTINS)\\b" "$script" 2>/dev/null | grep -v "^[0-9]*:#" | sed 's/^/         /'
        script_ok=false
    else
        pass
    fi

    # 2b. Explicit zsh invocations
    if grep -Pn '\bzsh\b' "$script" 2>/dev/null | grep -v "^[0-9]*:#" | grep -v "#!/bin/zsh" | grep -q .; then
        fail "explicitly invokes zsh:"
        grep -Pn '\bzsh\b' "$script" 2>/dev/null | grep -v "^[0-9]*:#" | grep -v "#!/bin/zsh" | sed 's/^/         /'
        script_ok=false
        if $FIX_MODE; then
            echo "         FIX: replace 'zsh' invocations with the script path (let shebang decide)"
        fi
    else
        pass
    fi
    fi # end self-skip

    # 3. Executable permission
    if [ -x "$script" ]; then
        pass
    else
        warn "not executable (chmod +x needed)"
    fi

    # 4. Trailing newline
    if [ -s "$script" ] && [ "$(tail -c1 "$script" | xxd -p)" != "0a" ]; then
        warn "no trailing newline"
        if $FIX_MODE; then
            echo "         FIX: append newline"
        fi
    else
        pass
    fi

    # 5. bash syntax check
    if bash -n "$script" 2>/dev/null; then
        pass
    else
        fail "bash -n parse error:"
        bash -n "$script" 2>&1 | sed 's/^/         /'
        script_ok=false
    fi

    # 6. shellcheck (if available)
    if command -v shellcheck >/dev/null 2>&1; then
        sc_output=$(shellcheck -S warning -f gcc "$script" 2>/dev/null)
        sc_count=$(echo "$sc_output" | grep -c "warning\|error" || true)
        if [ "$sc_count" -gt 0 ]; then
            warn "shellcheck: $sc_count warnings/errors"
            echo "$sc_output" | head -5 | sed 's/^/         /'
            if [ "$sc_count" -gt 5 ]; then
                echo "         ... and $((sc_count - 5)) more"
            fi
        else
            pass
        fi
    fi

    if $script_ok; then
        echo "  OK"
    fi
done

# Parity testing
if $PARITY; then
    echo ""
    echo "========================================"
    echo "PARITY TEST: zsh vs bash --help output"
    echo "========================================"

    if ! command -v zsh >/dev/null 2>&1; then
        echo "SKIP: zsh not installed — parity test requires both shells"
    else
        PARITY_PASS=0
        PARITY_FAIL=0
        PARITY_SKIP=0

        for script in "${SCRIPTS[@]}"; do
            # Only test scripts that accept --help (model/ensemble run.sh files)
            case "$script" in
                ./models/*/run.sh|./ensembles/*/run.sh)
                    ;;
                *)
                    continue
                    ;;
            esac

            zsh_out=$(timeout 30 zsh "$script" --help 2>&1) || true
            bash_out=$(timeout 30 bash "$script" --help 2>&1) || true

            if [ -z "$zsh_out" ] && [ -z "$bash_out" ]; then
                PARITY_SKIP=$((PARITY_SKIP + 1))
                continue
            fi

            if [ "$zsh_out" = "$bash_out" ]; then
                PARITY_PASS=$((PARITY_PASS + 1))
            else
                PARITY_FAIL=$((PARITY_FAIL + 1))
                echo ""
                echo "  DIFFERS: $script"
                diff <(echo "$zsh_out") <(echo "$bash_out") | head -10 | sed 's/^/    /'
            fi
        done

        echo ""
        echo "Parity: $PARITY_PASS pass, $PARITY_FAIL differ, $PARITY_SKIP skip"
    fi
fi

# Summary
echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Scripts scanned: $TOTAL"
echo "Pass:  $PASS checks"
echo "Warn:  $WARN"
echo "Fail:  $FAIL"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "VERDICT: FAIL — $FAIL issues must be fixed for Linux/CI portability"
    exit 1
elif [ "$WARN" -gt 0 ]; then
    echo "VERDICT: WARN — $WARN minor issues"
    exit 0
else
    echo "VERDICT: CLEAN"
    exit 0
fi
