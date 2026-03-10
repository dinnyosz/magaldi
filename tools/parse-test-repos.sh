#!/usr/bin/env bash
# Parse all cloned test repositories one by one.
# Each repo uses scope=test-repo and repository=<dirname>.
# Ensures magaldi.yaml exists with the right values before parsing.
# After all repos are parsed, extracts JSON summaries from parse logs
# and writes a comparison report to test_repos/_results/.
#
# Usage:
#   ./tools/parse-test-repos.sh                  # Parse all cloned repos
#   ./tools/parse-test-repos.sh --tier 1         # Smoke test only
#   ./tools/parse-test-repos.sh --lang rust      # Single language
#   ./tools/parse-test-repos.sh --dry-run        # In-memory, no DB needed
#   ./tools/parse-test-repos.sh --skip-ai        # Skip summarization/embedding
#   ./tools/parse-test-repos.sh --list           # Just list what would be parsed
#   ./tools/parse-test-repos.sh click ripgrep    # Parse specific repos by name
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEST="$PROJECT_ROOT/test_repos"
RESULTS_DIR="$DEST/_results"
SCOPE="test-repo"
USER="main"

# ── Repo definitions (mirrors clone-test-repos.sh) ─────────────────
# Format: "dirname|tier|language"

REPOS=(
  "click|1|python"
  "requests|2|python"
  "httpx|2|python"
  "fastapi|3|python"
  "pydantic|3|python"
  "full-stack-fastapi-template|3|python"
  "got|1|javascript"
  "express|2|javascript"
  "lodash|2|javascript"
  "axios|3|javascript"
  "date-fns|3|javascript"
  "Ghost|3|javascript"
  "nodebestpractices|3|javascript"
  "zod|1|typescript"
  "trpc|2|typescript"
  "drizzle-orm|2|typescript"
  "prisma|3|typescript"
  "typeorm|3|typescript"
  "immich|3|typescript"
  "guzzle|1|php"
  "composer|2|php"
  "PHPMailer|2|php"
  "framework|3|php"
  "console|3|php"
  "firefly-iii|3|php"
  "fd|1|rust"
  "ripgrep|2|rust"
  "bat|2|rust"
  "tokio|3|rust"
  "serde|3|rust"
  "zellij|3|rust"
  "gson|1|java"
  "spring-petclinic|2|java"
  "okhttp|2|java"
  "commons-lang|3|java"
  "junit5|3|java"
  "java-design-patterns|3|java"
  "kafka|3|java"
  "neofetch|1|bash"
  "rbenv|2|bash"
  "nvm|2|bash"
  "ohmyzsh|3|bash"
  "asdf|3|bash"
  "pi-hole|3|bash"
  "dokku|3|bash"
  "nickel|2|polyglot"
)

# ── Argument parsing ────────────────────────────────────────────────

TIER_FILTER=""
LANG_FILTER=""
LIST_ONLY=false
EXTRA_ARGS=()
SPECIFIC_REPOS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)
      TIER_FILTER="$2"
      shift 2
      ;;
    --lang|--language)
      LANG_FILTER="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
      shift 2
      ;;
    --list)
      LIST_ONLY=true
      shift
      ;;
    --dry-run|--skip-ai|--skip-tests|--skip-resolve|--force-clean)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --workers|-w)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --llm-url)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS] [REPO_NAME...]"
      echo ""
      echo "Options:"
      echo "  --tier 1|2       Filter by tier (1=smoke, 2=all)"
      echo "  --lang LANG      Filter by language"
      echo "  --list           Print repos without parsing"
      echo "  --dry-run        Use in-memory storage (no DB needed)"
      echo "  --skip-ai        Skip summarization and embedding"
      echo "  --skip-tests     Exclude test directories"
      echo "  --skip-resolve   Skip call resolution"
      echo "  --force-clean    Delete existing index data before parsing"
      echo "  --workers N      Max parallel workers"
      echo "  --llm-url URL    Override LLM API URL"
      echo ""
      echo "Positional args are repo directory names to parse (e.g., click ripgrep)."
      echo "If none given, all matching repos are parsed."
      echo ""
      echo "Results saved to: $RESULTS_DIR"
      echo "Run ./tools/clone-test-repos.sh first to clone repos."
      exit 0
      ;;
    -*)
      echo "Unknown option: $1 (try --help)"
      exit 1
      ;;
    *)
      # Positional arg = specific repo name
      SPECIFIC_REPOS+=("$1")
      shift
      ;;
  esac
done

# ── Ensure magaldi.yaml ────────────────────────────────────────────

ensure_config() {
  local repo_dir="$1"
  local name="$2"
  local config_file="$repo_dir/magaldi.yaml"

  # Always overwrite to ensure scope=test-repo
  cat > "$config_file" <<EOF
scope: $SCOPE
repository: $name
EOF
}

# ── Get JSON summary ───────────────────────────────────────────────

get_json_summary() {
  # Read the _last_run.json written by ParseRunLogger after each parse
  local repo_dir="$1"
  local json_file="$repo_dir/logs/_last_run.json"

  if [[ -f "$json_file" ]]; then
    cat "$json_file"
  else
    return 1
  fi
}

# ── Main ───────────────────────────────────────────────────────────

if [[ ! -d "$DEST" ]]; then
  echo "ERROR: test_repos/ directory not found."
  echo "Run ./tools/clone-test-repos.sh first."
  exit 1
fi

MATCHING=()
for entry in "${REPOS[@]}"; do
  IFS='|' read -r name tier lang <<< "$entry"

  # Filter by specific repos
  if [[ ${#SPECIFIC_REPOS[@]} -gt 0 ]]; then
    found=false
    for specific in "${SPECIFIC_REPOS[@]}"; do
      [[ "$name" == "$specific" ]] && found=true && break
    done
    $found || continue
  fi

  # Apply tier/lang filters
  [[ -n "$TIER_FILTER" && "$tier" -gt "$TIER_FILTER" ]] && continue
  [[ -n "$LANG_FILTER" && "$lang" != "$LANG_FILTER" ]] && continue

  MATCHING+=("$entry")
done

if [[ ${#MATCHING[@]} -eq 0 ]]; then
  echo "No repos match the given filters."
  exit 0
fi

if $LIST_ONLY; then
  printf "\n%-20s  %-6s  %-12s  %s\n" "REPO" "TIER" "LANGUAGE" "STATUS"
  printf "%-20s  %-6s  %-12s  %s\n" "────" "────" "────────" "──────"
  for entry in "${MATCHING[@]}"; do
    IFS='|' read -r name tier lang <<< "$entry"
    if [[ -d "$DEST/$name" ]]; then
      status="cloned"
    else
      status="NOT CLONED"
    fi
    printf "%-20s  %-6s  %-12s  %s\n" "$name" "$tier" "$lang" "$status"
  done
  echo ""
  exit 0
fi

# Set up results directory
RUN_TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS_DIR"

total=${#MATCHING[@]}
passed=0
failed=0
skipped=0
failed_repos=()
parsed_repos=()

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Magaldi Test Repo Parser                                   ║"
echo "║  Scope: $SCOPE                                         ║"
echo "║  Repos: $total                                              ║"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
echo "║  Flags: ${EXTRA_ARGS[*]}"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

start_time=$SECONDS

for i in "${!MATCHING[@]}"; do
  entry="${MATCHING[$i]}"
  IFS='|' read -r name tier lang <<< "$entry"
  idx=$((i + 1))

  repo_dir="$DEST/$name"

  if [[ ! -d "$repo_dir" ]]; then
    echo "[$idx/$total] SKIP $name (not cloned)"
    ((skipped++))
    continue
  fi

  echo "────────────────────────────────────────────────────────────"
  echo "[$idx/$total] PARSE $name ($lang, tier $tier)"
  echo "────────────────────────────────────────────────────────────"

  # Ensure magaldi.yaml has the right scope/repo
  ensure_config "$repo_dir" "$name"

  repo_start=$SECONDS

  if magaldi parse "$repo_dir" --user "$USER" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
    elapsed=$((SECONDS - repo_start))
    echo ""
    echo "[$idx/$total] PASS $name (${elapsed}s)"
    ((passed++))
    parsed_repos+=("$name|$lang|$tier|pass|$elapsed")
  else
    elapsed=$((SECONDS - repo_start))
    echo ""
    echo "[$idx/$total] FAIL $name (${elapsed}s)"
    ((failed++))
    failed_repos+=("$name")
    parsed_repos+=("$name|$lang|$tier|fail|$elapsed")
  fi

  # Extract JSON summary from parse log regardless of pass/fail
  if json=$(get_json_summary "$repo_dir"); then
    echo "$json" > "$RESULTS_DIR/${name}.json"
  fi

  echo ""
done

total_time=$((SECONDS - start_time))

# ── Generate comparison report ─────────────────────────────────────

REPORT="$RESULTS_DIR/run_${RUN_TS}.txt"

{
  echo "Magaldi Parse Test Results — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "═══════════════════════════════════════════════════════════"
  echo ""

  # Header
  printf "%-16s %-12s %5s %7s %7s %8s %8s %7s %7s %9s\n" \
    "REPO" "LANG" "TIER" "STATUS" "TIME" "FILES" "ELEMS" "INDEXD" "ERRORS" "TOKENS"
  printf "%-16s %-12s %5s %7s %7s %8s %8s %7s %7s %9s\n" \
    "────" "────" "────" "──────" "─────" "─────" "─────" "─────" "──────" "──────"

  for entry in "${parsed_repos[@]}"; do
    IFS='|' read -r name lang tier status elapsed <<< "$entry"

    json_file="$RESULTS_DIR/${name}.json"
    if [[ -f "$json_file" ]]; then
      # Extract key metrics using python (available in magaldi venv)
      read -r files elems indexed errors tokens < <(
        python3 -c "
import json, sys
with open('$json_file') as f:
    d = json.load(f)
p = d.get('processing', {})
t = d.get('token_usage', {})
print(
    p.get('processed', 0) + p.get('skipped', 0),
    p.get('processed', 0),
    p.get('indexed', 0),
    d.get('error_count', 0),
    t.get('total', 0),
)
" 2>/dev/null || echo "- - - - -"
      )
    else
      files="-"; elems="-"; indexed="-"; errors="-"; tokens="-"
    fi

    printf "%-16s %-12s %5s %7s %5ss %8s %8s %7s %7s %9s\n" \
      "$name" "$lang" "$tier" "$status" "$elapsed" "$files" "$elems" "$indexed" "$errors" "$tokens"
  done

  echo ""
  echo "───────────────────────────────────────────────────────────"
  echo "TOTAL: $passed passed, $failed failed, $skipped skipped (${total_time}s)"
  if [[ ${#failed_repos[@]} -gt 0 ]]; then
    echo "FAILED: ${failed_repos[*]}"
  fi
  echo ""

  # Phase timing breakdown per repo (from JSON files)
  echo ""
  echo "Phase Timing Breakdown (seconds)"
  echo "═══════════════════════════════════════════════════════════"
  printf "%-16s %8s %8s %8s %8s %8s %8s %8s\n" \
    "REPO" "DISCOV" "CHANGE" "PARSE" "SCORE" "PROCESS" "RESOLVE" "TOTAL"
  printf "%-16s %8s %8s %8s %8s %8s %8s %8s\n" \
    "────" "──────" "──────" "─────" "─────" "───────" "───────" "─────"

  for entry in "${parsed_repos[@]}"; do
    IFS='|' read -r name lang tier status elapsed <<< "$entry"

    json_file="$RESULTS_DIR/${name}.json"
    if [[ -f "$json_file" ]]; then
      read -r discov change parse score process resolve total_e < <(
        python3 -c "
import json, sys
with open('$json_file') as f:
    d = json.load(f)
pt = d.get('phase_timings', {})
def g(prefix):
    for k, v in pt.items():
        if prefix.lower() in k.lower():
            return round(v, 1)
    return 0
print(
    g('discovery'), g('change'), g('parsing'), g('scoring'),
    g('processing'), g('call resolution'), round(d.get('total_elapsed_seconds', 0), 1)
)
" 2>/dev/null || echo "- - - - - - -"
      )
    else
      discov="-"; change="-"; parse="-"; score="-"; process="-"; resolve="-"; total_e="-"
    fi

    printf "%-16s %8s %8s %8s %8s %8s %8s %8s\n" \
      "$name" "$discov" "$change" "$parse" "$score" "$process" "$resolve" "$total_e"
  done

  echo ""
} > "$REPORT"

# Print summary to terminal too
cat "$REPORT"

echo "Results saved to:"
echo "  Report:     $REPORT"
echo "  Per-repo:   $RESULTS_DIR/<name>.json"
echo ""

# Also save a combined JSON for programmatic comparison
COMBINED="$RESULTS_DIR/run_${RUN_TS}.json"
python3 -c "
import json, glob, os

results = {}
for f in sorted(glob.glob('$RESULTS_DIR/*.json')):
    name = os.path.splitext(os.path.basename(f))[0]
    if name.startswith('run_'):
        continue
    with open(f) as fh:
        results[name] = json.load(fh)

with open('$COMBINED', 'w') as fh:
    json.dump({
        'timestamp': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
        'repos': results,
    }, fh, indent=2)
" 2>/dev/null && echo "  Combined:   $COMBINED" || true

# Symlink latest
ln -sf "run_${RUN_TS}.txt" "$RESULTS_DIR/latest.txt" 2>/dev/null || true
ln -sf "run_${RUN_TS}.json" "$RESULTS_DIR/latest.json" 2>/dev/null || true

# Exit with failure if any repo failed
[[ $failed -eq 0 ]]
